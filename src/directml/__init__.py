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
    """
    Get a DirectML device or raise an exception if not available.
    
    Returns:
        torch_directml.Device: A DirectML device instance.
        
    Raises:
        RuntimeError: If DirectML is not available or fails to initialize.
    """
    if DIRECTML_AVAILABLE:
        try:
            return torch_directml.device()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DirectML device: {e}")
    else:
        raise RuntimeError(
            "DirectML is not available. Please install torch-directml package: "
            "pip install torch-directml"
        )
