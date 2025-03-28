#!/usr/bin/env python3
"""
Seed utilities for consistent randomization across different libraries.

This module provides functions to set random seeds consistently across
NumPy, PyTorch, PyBullet, and other libraries to ensure reproducible results.
"""

import random
import numpy as np
import os

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Try to set TensorFlow's random seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set environment variables for libraries that check them
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set other libraries' seeds if needed
    try:
        import gym
        gym.utils.seeding.np_random(seed)
    except (ImportError, AttributeError):
        pass
    
    print(f"Random seed set to {seed} for all libraries") 