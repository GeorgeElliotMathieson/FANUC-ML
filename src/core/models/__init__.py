"""
Models module for FANUC Robot ML Platform.

Contains neural network models, feature extractors, and policy implementations.
"""

# Import feature extractor
from src.core.models.features import CustomFeatureExtractor

# Import network models
from src.core.models.networks import (
    CustomActorNetwork,
    CustomCriticNetwork,
    CustomActorCriticPolicy
)

# Import PPO implementation
from src.core.models.ppo import CustomPPO

# Import DirectML model
from src.core.models.directml_model import CustomDirectMLModel

# Define exports
__all__ = [
    # Feature extractor
    'CustomFeatureExtractor',
    
    # Network models
    'CustomActorNetwork',
    'CustomCriticNetwork',
    'CustomActorCriticPolicy',
    
    # PPO implementation
    'CustomPPO',
    
    # DirectML model
    'CustomDirectMLModel'
] 