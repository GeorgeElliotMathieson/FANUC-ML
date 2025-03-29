"""
Feature extraction classes for neural network models.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the robot policy network.
    
    Extracts meaningful features from raw observations, reducing the dimension
    while preserving important information.
    """
    def __init__(self, observation_space: spaces.Box):
        # Use a smaller feature dimension for efficiency
        features_dim = 64  # Final features dimension
        super().__init__(observation_space, features_dim)
        
        # Input dimension from observation space
        n_input = int(np.prod(observation_space.shape))
        
        # Create a two-layer feature extraction network
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Initialize the weights with orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Raw observations from the environment
            
        Returns:
            Tensor of extracted features
        """
        # Flatten if needed
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)
            
        # Extract features
        features = self.net(observations)
        return features 