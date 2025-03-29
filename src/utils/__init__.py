"""
Utility functions for the FANUC robot ML framework.

Contains helper functions, visualization tools, and other utilities
to support the robot training and evaluation processes.
"""

import random
import os
import numpy as np

def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): The random seed to use
    """
    if seed is None:
        return
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Try to set PyTorch's random seed if available
    try:
        import torch
        torch.manual_seed(seed)
        
        # Try to set DirectML-specific settings if available
        try:
            import torch_directml  # type: ignore
            # DirectML doesn't have specific seed controls like CUDA,
            # but we can still set the PyTorch seeds
            torch.manual_seed(seed)
        except ImportError:
            # DirectML not available
            pass
    except ImportError:
        pass
    
    # Try to set TensorFlow's random seed if available
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set environment variables for libraries that check them
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set other libraries' seeds if needed
    try:
        import gym  # type: ignore
        gym.utils.seeding.np_random(seed)
    except (ImportError, AttributeError):
        pass
    
    print(f"Random seed set to {seed} for all libraries")

# Import other utility functions
from .pybullet_utils import (
    get_pybullet_client,
    get_shared_pybullet_client,
    configure_visualization,
    visualize_target,
    visualize_ee_position,
    visualize_target_line,
    load_workspace_data,
    determine_reachable_workspace,
    adjust_camera_for_robots
)
