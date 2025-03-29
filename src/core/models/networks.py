"""
Neural network models for the policy and value functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core.models.features import CustomFeatureExtractor

class CustomActorNetwork(nn.Module):
    """
    Custom actor network for PPO that outputs mean and log standard deviation
    for a Gaussian policy.
    """
    def __init__(self, feature_dim, action_dim):
        """
        Initialize the actor network.
        
        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of action space
        """
        super().__init__()
        
        # Create a multi-layer actor network
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Mean and log standard deviation heads
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        
        # Initialize with small values for better stability
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)
        
    def forward(self, features):
        """
        Forward pass through the actor network.
        
        Args:
            features: Input features extracted by the feature extractor
            
        Returns:
            Tuple of (mean, log_std) for the Gaussian policy
        """
        # First layer
        x = self.shared(features)
        
        # Output mean and log_std
        mean = self.mean(x)
        
        # Bound log_std for numerical stability
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

class CustomCriticNetwork(nn.Module):
    """
    Custom critic network for PPO that outputs a state value estimate.
    """
    def __init__(self, feature_dim):
        """
        Initialize the critic network.
        
        Args:
            feature_dim: Dimension of input features
        """
        super().__init__()
        
        # Create a multi-layer critic network
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single value estimate
        )
        
        # Initialize with appropriate gain for value function
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
                
    def forward(self, features):
        """
        Forward pass through the critic network.
        
        Args:
            features: Input features extracted by the feature extractor
            
        Returns:
            Value estimate for the input state
        """
        # First layer
        value = self.net(features)
        return value

class CustomActorCriticPolicy(nn.Module):
    """
    Custom actor-critic policy for PPO with our improved architectures.
    Combines the feature extractor, actor, and critic networks.
    """
    def __init__(self, observation_space, action_space):
        """
        Initialize the actor-critic policy.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
        """
        super().__init__()
        
        # Feature extractor
        self.features_extractor = CustomFeatureExtractor(observation_space)
        feature_dim = self.features_extractor.features_dim
        
        # Actor network
        self.actor = CustomActorNetwork(feature_dim, action_space.shape[0])
        
        # Critic network
        self.critic = CustomCriticNetwork(feature_dim)
    
    def forward(self, obs):
        """
        Forward pass through the actor-critic policy.
        
        Args:
            obs: Observations from the environment
            
        Returns:
            Tuple of (mean, log_std, value) for the policy and value function
        """
        # Extract features
        features = self.features_extractor(obs)
        
        # Get action distribution parameters
        mean, log_std = self.actor(features)
        
        # Get value estimate
        value = self.critic(features)
        
        return mean, log_std, value 