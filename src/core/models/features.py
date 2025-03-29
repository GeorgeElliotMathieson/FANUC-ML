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
    
    Optimized for GPU performance with DirectML using a balanced architecture
    for best performance.
    """
    def __init__(self, observation_space: spaces.Box, features_dim=384):
        # Use the provided features dimension or default to 384 (optimal balance)
        super().__init__(observation_space, features_dim)
        
        # Input dimension from observation space
        n_input = int(np.prod(observation_space.shape))
        
        # Create optimally balanced feature extraction network
        self.fc1 = nn.Linear(n_input, 512)  # Larger first layer captures more input patterns
        self.relu1 = nn.ReLU(inplace=True)  # Use in-place ReLU to save memory
        self.fc2 = nn.Linear(512, features_dim)  # Output at optimal 384-dimension feature space
        self.relu2 = nn.ReLU(inplace=True)
        
        # Use batch normalization for improved training stability
        self.bn1 = nn.BatchNorm1d(512)
        
        # Initialize the weights with orthogonal initialization for better gradient flow
        nn.init.orthogonal_(self.fc1.weight, gain=1.41)  # Increased gain for better initialization
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.41)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Raw observations from the environment
            
        Returns:
            Tensor of extracted features
        """
        # Reshape while preserving batch dimension for efficiency
        if len(observations.shape) > 2:
            observations = observations.reshape(observations.shape[0], -1)

        # Skip batch norm during inference or if batch size is 1
        training_mode = self.training and observations.shape[0] > 1
            
        # Extract features with GPU-optimized operations
        features = self.fc1(observations)
        if training_mode:
            features = self.bn1(features)
        features = self.relu1(features)
        
        features = self.fc2(features)
        features = self.relu2(features)
        
        return features 