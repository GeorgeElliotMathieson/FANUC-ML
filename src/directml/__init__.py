"""
DirectML-specific implementations for FANUC robot control using AMD GPUs.

This module provides optimized versions of the robot training and evaluation
code that leverages DirectML to accelerate computations on AMD GPUs.
"""

# Import key symbols for direct access
try:
    # Only import if torch_directml is available
    import torch_directml
    DIRECTML_AVAILABLE = True
    
    # Import core functionality
    from .train_robot_rl_ppo_directml import (
        DirectMLPPO,
        train_robot_with_ppo_directml
    )
    
    # Import the main entry point
    from .directml_train import main as directml_main
    
except ImportError:
    DIRECTML_AVAILABLE = False

def is_available():
    """Check if DirectML is available in the current environment."""
    return DIRECTML_AVAILABLE

def get_device():
    """Get a DirectML device if available, otherwise return None."""
    if DIRECTML_AVAILABLE:
        try:
            return torch_directml.device()
        except:
            pass
    return None
