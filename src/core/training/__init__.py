"""
Training module for FANUC Robot ML Platform.

Contains training functions, callbacks, and training utilities.
"""

# Import callbacks
from src.core.training.callbacks import (
    SaveModelCallback,
    TrainingMonitorCallback,
    JointLimitMonitorCallback
)

# Import training functionality
from src.core.training.train import (
    create_revamped_envs,
    train_model,
    train_revamped_robot
)

# Define exports
__all__ = [
    # Callbacks
    'SaveModelCallback',
    'TrainingMonitorCallback',
    'JointLimitMonitorCallback',
    
    # Training functionality
    'create_revamped_envs',
    'train_model',
    'train_revamped_robot'
] 