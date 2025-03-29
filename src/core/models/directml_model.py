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
        self.loaded = False  # Initialize loaded flag to prevent reference before assignment
        
        # Try to get default device if none provided
        if device is None:
            try:
                from src.dml import get_device
                device = get_device()
                if device is None:
                    raise ImportError("DirectML device could not be initialized")
            except ImportError as e:
                print(f"WARNING: DirectML initialization failed: {e}")
                print("Falling back to CPU. This will be very slow.")
                device = torch.device("cpu")
            except Exception as e:
                print(f"WARNING: Unexpected error initializing DirectML: {e}")
                print("Falling back to CPU. This will be very slow.")
                device = torch.device("cpu")
                import traceback
                print(traceback.format_exc())
        
        self.device = device
        print(f"CustomDirectMLModel initialized with device: {self.device}")
        
        # Initialize model architecture
        self.policy = None
        self.feature_extractor = None
        self.actor = None
        self.critic = None
        
        # Model state
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
        # Check if model is loaded before proceeding
        if not hasattr(self, 'loaded') or not self.loaded:
            print("WARNING: Model not loaded. Call load() first.")
            # Return zeros as a safe default
            return self._get_default_action(), None
                
        try:
            # Convert observation to tensor
            processed_observation = self._preprocess_observation(observation)
            if processed_observation is None:
                return self._get_default_action(), None
                
            # Extract features
            with torch.no_grad():
                features = None
                try:
                    features = self.feature_extractor(processed_observation)
                except Exception as e:
                    print(f"ERROR in feature extraction: {e}")
                    return self._get_default_action(), None
                
                # Get action distribution parameters
                try:
                    action_mean, _ = self.actor(features)
                except Exception as e:
                    print(f"ERROR in actor network: {e}")
                    return self._get_default_action(), None
                
                # Use mean for deterministic or sample for stochastic
                if deterministic:
                    action = action_mean
                else:
                    # In a full implementation, we would sample from distribution
                    # For simplicity, just add some noise
                    noise = torch.randn_like(action_mean) * 0.1
                    action = action_mean + noise
                    
                # Convert to numpy
                try:
                    action = action.cpu().numpy()
                except Exception as e:
                    print(f"ERROR converting action to numpy: {e}")
                    return self._get_default_action(), None
                
                # If single observation, return first action
                if action.shape[0] == 1:
                    action = action[0]
                    
            return action, None
            
        except Exception as e:
            print(f"ERROR in predict method: {e}")
            import traceback
            print(traceback.format_exc())
            return self._get_default_action(), None
            
    def _preprocess_observation(self, observation):
        """
        Preprocess observation to tensor format.
        
        Args:
            observation: Raw observation
            
        Returns:
            Preprocessed observation tensor or None if preprocessing failed
        """
        try:
            if isinstance(observation, np.ndarray):
                # Handle batch vs single observation
                if len(observation.shape) == 1:
                    observation = observation.reshape(1, -1)
                    
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            elif isinstance(observation, list):
                # Handle list observations
                observation_tensor = torch.as_tensor(np.array(observation), dtype=torch.float32)
                if len(observation_tensor.shape) == 1:
                    observation_tensor = observation_tensor.reshape(1, -1)
            elif isinstance(observation, torch.Tensor):
                observation_tensor = observation
                if len(observation_tensor.shape) == 1:
                    observation_tensor = observation_tensor.reshape(1, -1)
            else:
                print(f"WARNING: Unexpected observation type: {type(observation)}")
                return None
                
            # Move to device
            try:
                observation_tensor = observation_tensor.to(self.device)
            except Exception as e:
                print(f"WARNING: Error moving observation to device: {e}")
                # Fall back to CPU if needed
                observation_tensor = observation_tensor.to(torch.device("cpu"))
                
            return observation_tensor
            
        except Exception as e:
            print(f"ERROR preprocessing observation: {e}")
            return None
            
    def _get_default_action(self):
        """
        Get default action when prediction fails.
        
        Returns:
            Default action as numpy array
        """
        try:
            if hasattr(self, 'action_dim'):
                return np.zeros(self.action_dim)
            elif hasattr(self, 'action_space') and hasattr(self.action_space, 'shape') and self.action_space.shape:
                return np.zeros(self.action_space.shape[0])
            else:
                # Ultra-safe fallback
                return np.zeros(4)  # Default reasonable size for robot control
        except Exception:
            return np.zeros(4)  # Ultimate fallback
        
    def get_features(self, observation):
        """
        Extract features from an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Extracted features
        """
        # Check if model is loaded before proceeding
        if not hasattr(self, 'loaded') or not self.loaded:
            print("WARNING: Model not loaded. Call load() first.")
            return self._get_default_features()
            
        try:
            # Use the shared preprocessing method
            processed_observation = self._preprocess_observation(observation)
            if processed_observation is None:
                return self._get_default_features()
            
            # Extract features
            with torch.no_grad():
                try:
                    features = self.feature_extractor(processed_observation)
                    return features
                except Exception as e:
                    print(f"ERROR in feature extraction: {e}")
                    return self._get_default_features()
                
        except Exception as e:
            print(f"ERROR in get_features method: {e}")
            import traceback
            print(traceback.format_exc())
            return self._get_default_features()
            
    def _get_default_features(self):
        """
        Get default features when extraction fails.
        
        Returns:
            Default features tensor
        """
        try:
            if hasattr(self, 'features_dim'):
                return torch.zeros(1, self.features_dim, device=torch.device("cpu"))
            else:
                return torch.zeros(1, 64, device=torch.device("cpu"))  # Default reasonable size
        except Exception:
            # Ultra-safe fallback
            return torch.zeros(1, 64, device=torch.device("cpu"))
            
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
            
        # Infer action dimension from action space or try to determine from state dict
        try:
            if self.action_space is not None and hasattr(self.action_space, 'shape') and self.action_space.shape is not None and len(self.action_space.shape) > 0:
                self.action_dim = self.action_space.shape[0]
            else:
                # Try to infer from state dict - find actor mean weights
                actor_mean_keys = [k for k in state_dict.keys() if 'actor.mean.weight' in k]
                if actor_mean_keys:
                    # Get output dimension from the actor mean weights
                    self.action_dim = state_dict[actor_mean_keys[0]].shape[0]
                else:
                    print("WARNING: Could not infer action dimension from state dict")
                    self.action_dim = 4  # Default to a reasonable value for robot control
            
            print(f"Action dimension: {self.action_dim}")
        except Exception as e:
            print(f"Error inferring action dimension: {e}. Using default.")
            self.action_dim = 4  # Default to a reasonable value for robot control
        
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
            # Determine input dimension from observation space
            input_dim = 64  # Default reasonable size
            if hasattr(self, 'observation_space') and self.observation_space is not None:
                try:
                    if hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None:
                        # Calculate product of all dimensions for flattened input
                        input_dim = int(np.prod(self.observation_space.shape))
                        print(f"Using input dimension from observation space: {input_dim}")
                    else:
                        print("WARNING: Observation space has no shape attribute, using default input dimension")
                except (TypeError, ValueError, AttributeError) as e:
                    print(f"WARNING: Could not determine input dimension from observation space: {e}")
            else:
                print("WARNING: No observation space available, using default input dimension")
                
            # Create feature extractor
            if input_dim <= 0:
                print(f"WARNING: Invalid input dimension {input_dim}, using default value")
                input_dim = 64  # Default reasonable size
                
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, 128),
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
            import traceback
            print(traceback.format_exc())
            return False
            
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            self for chaining
        """
        if device is None:
            print("Warning: Requested device is None, staying on current device")
            return self
            
        # Store original device for rollback in case of error
        original_device = self.device
            
        try:
            # First try to move to the device
            self.device = device
            
            # Move networks to the device
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                self.feature_extractor = self.feature_extractor.to(device)
                
            if hasattr(self, 'actor_shared') and self.actor_shared is not None:
                self.actor_shared = self.actor_shared.to(device)
                
            if hasattr(self, 'actor_mean') and self.actor_mean is not None:
                self.actor_mean = self.actor_mean.to(device)
                
            if hasattr(self, 'actor_logstd') and self.actor_logstd is not None:
                self.actor_logstd = self.actor_logstd.to(device)
                
            if hasattr(self, 'critic') and self.critic is not None:
                self.critic = self.critic.to(device)
                
            print(f"Successfully moved model to device: {device}")
            return self
            
        except Exception as e:
            print(f"ERROR: Failed to move model to device {device}: {e}")
            # Roll back to original device
            self.device = original_device
            print(f"Rolled back to original device: {original_device}")
            return self
            
    def load(self, path):
        """
        Load model from path.
        
        Args:
            path: Path to model file
            
        Returns:
            Self for chaining
        """
        if path is None:
            print("ERROR: Model path cannot be None")
            return False
            
        print(f"Loading model from: {path}")
        
        try:
            # Check various file extensions
            possible_paths = [
                path,
                path + ".pt",
                path + ".pth",
                path + ".model"
            ]
            
            path_to_try = None
            for p in possible_paths:
                if os.path.exists(p):
                    path_to_try = p
                    if p != path:
                        print(f"Using model file with extension: {p}")
                    break
                    
            if path_to_try is None:
                print(f"ERROR: Model file not found at {path} or with standard extensions (.pt/.pth/.model)")
                return False
                
            # Load the state dict
            try:
                state_dict = torch.load(path_to_try, map_location=self.device)
            except Exception as e:
                print(f"ERROR: Failed to load model file: {e}")
                import traceback
                print(traceback.format_exc())
                return False
                
            # Try to handle different formats of saved models
            if isinstance(state_dict, dict):
                # Check if this is a complete model or just state_dict
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
            else:
                print(f"WARNING: Expected dictionary for state_dict, got {type(state_dict)}")
                
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
            
            # Check if we have valid mappings
            if not param_mapping:
                print("WARNING: No parameter mappings found. Model may not be loaded correctly.")
            
            # Load parameters
            with torch.no_grad():
                for target_name, source_name in param_mapping.items():
                    try:
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
                    except Exception as e:
                        print(f"WARNING: Error loading parameter {target_name} from {source_name}: {e}")
            
            self.loaded = True
            print(f"Model successfully loaded from {path_to_try}")
            return self
            
        except Exception as e:
            import traceback
            print(f"ERROR: Exception while loading model: {e}")
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
        
        if not isinstance(state_dict, dict):
            print(f"ERROR: Expected state_dict to be a dictionary, got {type(state_dict)}")
            return mapping
        
        # Extract all parameter names from state dict
        try:
            param_names = [k for k in state_dict.keys() if isinstance(k, str) and 'optimizer' not in k]
        except Exception as e:
            print(f"ERROR: Failed to extract parameter names from state_dict: {e}")
            return mapping
        
        # Feature extractor parameter mapping
        for name in param_names:
            try:
                if 'features_extractor.net' in name:
                    # Parse layer index
                    parts = name.split('.')
                    if len(parts) >= 5 and parts[3].isdigit():
                        layer_idx = int(parts[3])
                        param_type = parts[4]  # weight or bias
                        
                        target_name = f"feature_extractor.{layer_idx}.{param_type}"
                        mapping[target_name] = name
            except Exception as e:
                print(f"WARNING: Error mapping feature extractor parameter {name}: {e}")
                continue
        
        # Actor parameter mapping
        for name in param_names:
            try:
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
            except Exception as e:
                print(f"WARNING: Error mapping actor parameter {name}: {e}")
                continue
        
        # Critic parameter mapping
        for name in param_names:
            try:
                if 'policy.critic.net' in name:
                    # Parse layer index
                    parts = name.split('.')
                    if len(parts) >= 5 and parts[4].isdigit():
                        layer_idx = int(parts[4])
                        param_type = parts[5]  # weight or bias
                        
                        target_name = f"critic.{layer_idx}.{param_type}"
                        mapping[target_name] = name
            except Exception as e:
                print(f"WARNING: Error mapping critic parameter {name}: {e}")
                continue
        
        if not mapping:
            print("WARNING: No parameter mappings created. State dict may have an unexpected format.")
            
        return mapping 