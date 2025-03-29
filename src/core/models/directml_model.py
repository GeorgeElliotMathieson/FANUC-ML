"""
Custom model implementation for DirectML compatibility.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

class CustomDirectMLModel:
    """
    Utility class to evaluate a DirectML model.
    
    This provides compatibility with AMD GPUs via DirectML, implementing a subset
    of the Stable Baselines 3 API for model loading and prediction.
    """
    def __init__(self, observation_space, action_space, device=None):
        """
        Initialize the DirectML model.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            device: DirectML device (if None, will be obtained automatically)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Try to get default device if none provided
        if device is None:
            try:
                import torch_directml
                device = torch_directml.device()
            except ImportError:
                raise ImportError("DirectML is required but not available")
        
        self.device = device
        print(f"CustomDirectMLModel initialized with device: {self.device}")
        
        # Initialize model architecture
        self.policy = None
        self.feature_extractor = None
        self.actor = None
        self.critic = None
        
        # Model state
        self.loaded = False
        self.eval_mode = True
        
    def predict(self, observation, deterministic=True):
        """
        Get model prediction.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, None) following SB3 API
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            # Handle batch vs single observation
            if len(observation.shape) == 1:
                observation = observation.reshape(1, -1)
                
            observation = torch.as_tensor(observation, dtype=torch.float32)
            
        # Move to device
        observation = observation.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(observation)
            
            # Get action distribution parameters
            action_mean, _ = self.actor(features)
            
            # Use mean for deterministic or sample for stochastic
            if deterministic:
                action = action_mean
            else:
                # In a full implementation, we would sample from distribution
                # For simplicity, just add some noise
                noise = torch.randn_like(action_mean) * 0.1
                action = action_mean + noise
                
            # Convert to numpy
            action = action.cpu().numpy()
            
            # If single observation, return first action
            if action.shape[0] == 1:
                action = action[0]
                
        return action, None
        
    def get_features(self, observation):
        """
        Extract features from an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Extracted features
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation, dtype=torch.float32)
            
        # Move to device
        observation = observation.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(observation)
            
        return features
            
    def _infer_architecture(self, state_dict):
        """
        Infer the model architecture from the state dictionary.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Whether the architecture was successfully inferred
        """
        # Extract architecture information from state dict keys
        layers = {}
        for key in state_dict.keys():
            parts = key.split('.')
            
            # Skip optimizer states if present
            if parts[0] == 'optimizer':
                continue
                
            # Count layer dimensions from weights
            if len(parts) >= 2 and parts[-1] == 'weight':
                module_path = '.'.join(parts[:-1])
                shape = state_dict[key].shape
                layers[module_path] = shape
                
        # Check for feature extractor
        feature_extractor_keys = [k for k in layers.keys() if 'features_extractor' in k]
        if not feature_extractor_keys:
            print("Could not find feature extractor in state dict")
            return False
            
        # Infer feature dimension from last layer
        last_feature_layer = sorted(feature_extractor_keys)[-1]
        self.features_dim = layers[last_feature_layer][0]
        print(f"Inferred feature dimension: {self.features_dim}")
        
        # Check for actor and critic
        actor_keys = [k for k in layers.keys() if 'actor' in k]
        critic_keys = [k for k in layers.keys() if 'critic' in k]
        
        if not actor_keys or not critic_keys:
            print("Could not find actor or critic in state dict")
            return False
            
        # Infer action dimension from actor output
        self.action_dim = self.action_space.shape[0]
        print(f"Action dimension: {self.action_dim}")
        
        # Success
        return True
        
    def _build_networks(self):
        """
        Build the neural network architecture.
        
        Returns:
            Whether the networks were successfully built
        """
        if not hasattr(self, 'features_dim') or not hasattr(self, 'action_dim'):
            print("Cannot build networks: Architecture inference failed")
            return False
            
        try:
            # Create feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(np.prod(self.observation_space.shape), 128),
                nn.ReLU(),
                nn.Linear(128, self.features_dim),
                nn.ReLU()
            )
            
            # Create actor (policy) network - mean and log_std branches
            self.actor_shared = nn.Sequential(
                nn.Linear(self.features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            
            self.actor_mean = nn.Linear(64, self.action_dim)
            self.actor_logstd = nn.Linear(64, self.action_dim)
            
            # Combine into actor function
            def actor_fn(x):
                shared = self.actor_shared(x)
                mean = self.actor_mean(shared)
                log_std = self.actor_logstd(shared)
                # Clamp log_std for numerical stability
                log_std = torch.clamp(log_std, -20, 2)
                return mean, log_std
                
            self.actor = actor_fn
            
            # Create critic (value) network
            self.critic = nn.Sequential(
                nn.Linear(self.features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            # Move networks to device
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.actor_shared = self.actor_shared.to(self.device)
            self.actor_mean = self.actor_mean.to(self.device)
            self.actor_logstd = self.actor_logstd.to(self.device)
            self.critic = self.critic.to(self.device)
            
            # Set to evaluation mode
            self.feature_extractor.eval()
            self.actor_shared.eval()
            self.actor_mean.eval()
            self.actor_logstd.eval()
            self.critic.eval()
            
            return True
            
        except Exception as e:
            print(f"Error building networks: {e}")
            return False
            
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        
        if self.loaded:
            # Move networks to device
            self.feature_extractor = self.feature_extractor.to(device)
            self.actor_shared = self.actor_shared.to(device)
            self.actor_mean = self.actor_mean.to(device)
            self.actor_logstd = self.actor_logstd.to(device)
            self.critic = self.critic.to(device)
            
        return self
            
    def load(self, path):
        """
        Load model from path.
        
        Args:
            path: Path to model file
            
        Returns:
            Self for chaining
        """
        print(f"Loading model from: {path}")
        
        try:
            # Load the state dict
            if path.endswith('.pt'):
                state_dict = torch.load(path, map_location=self.device)
            else:
                # Try with .pt extension
                state_dict = torch.load(f"{path}.pt", map_location=self.device)
                
            # Infer architecture
            if not self._infer_architecture(state_dict):
                print("Failed to infer architecture from state dict")
                return False
                
            # Build networks
            if not self._build_networks():
                print("Failed to build networks")
                return False
                
            # Create parameter mapping
            param_mapping = self._create_parameter_mapping(state_dict)
            
            # Load parameters
            with torch.no_grad():
                for target_name, source_name in param_mapping.items():
                    parts = target_name.split('.')
                    if parts[0] == 'feature_extractor':
                        layer_idx = int(parts[1])
                        param_type = parts[2]
                        self.feature_extractor[layer_idx].__dict__[param_type].copy_(
                            state_dict[source_name]
                        )
                    elif parts[0] == 'actor_shared':
                        layer_idx = int(parts[1])
                        param_type = parts[2]
                        self.actor_shared[layer_idx].__dict__[param_type].copy_(
                            state_dict[source_name]
                        )
                    elif parts[0] == 'actor_mean':
                        param_type = parts[1]
                        self.actor_mean.__dict__[param_type].copy_(
                            state_dict[source_name]
                        )
                    elif parts[0] == 'actor_logstd':
                        param_type = parts[1]
                        self.actor_logstd.__dict__[param_type].copy_(
                            state_dict[source_name]
                        )
                    elif parts[0] == 'critic':
                        layer_idx = int(parts[1])
                        param_type = parts[2]
                        self.critic[layer_idx].__dict__[param_type].copy_(
                            state_dict[source_name]
                        )
            
            self.loaded = True
            return self
            
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            return False
            
    def _create_parameter_mapping(self, state_dict):
        """
        Create a mapping between state dict keys and model parameters.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Dictionary mapping target parameter names to source names
        """
        mapping = {}
        
        # Extract all parameter names from state dict
        param_names = [k for k in state_dict.keys() if 'optimizer' not in k]
        
        # Feature extractor parameter mapping
        for name in param_names:
            if 'features_extractor.net' in name:
                # Parse layer index
                parts = name.split('.')
                if len(parts) >= 5 and parts[3].isdigit():
                    layer_idx = int(parts[3])
                    param_type = parts[4]  # weight or bias
                    
                    target_name = f"feature_extractor.{layer_idx}.{param_type}"
                    mapping[target_name] = name
        
        # Actor parameter mapping
        for name in param_names:
            if 'policy.actor.shared' in name:
                # Parse layer index
                parts = name.split('.')
                if len(parts) >= 5 and parts[4].isdigit():
                    layer_idx = int(parts[4])
                    param_type = parts[5]  # weight or bias
                    
                    target_name = f"actor_shared.{layer_idx}.{param_type}"
                    mapping[target_name] = name
            elif 'policy.actor.mean' in name:
                parts = name.split('.')
                param_type = parts[-1]  # weight or bias
                
                target_name = f"actor_mean.{param_type}"
                mapping[target_name] = name
            elif 'policy.actor.log_std' in name:
                parts = name.split('.')
                param_type = parts[-1]  # weight or bias
                
                target_name = f"actor_logstd.{param_type}"
                mapping[target_name] = name
        
        # Critic parameter mapping
        for name in param_names:
            if 'policy.critic.net' in name:
                # Parse layer index
                parts = name.split('.')
                if len(parts) >= 5 and parts[4].isdigit():
                    layer_idx = int(parts[4])
                    param_type = parts[5]  # weight or bias
                    
                    target_name = f"critic.{layer_idx}.{param_type}"
                    mapping[target_name] = name
        
        return mapping 