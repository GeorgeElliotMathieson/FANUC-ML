"""
Training functionality for FANUC Robot ML Platform.
"""

import os
import time
import traceback
from src.core.utils import print_banner

def print_train_usage():
    """Print usage instructions for the train command."""
    print("Usage: python fanuc_platform.py train [model_path] [steps] [options]")
    print("\nArguments:")
    print("  model_path    - Path to save the model (optional)")
    print("  steps         - Number of training steps (default: 500000)")
    print("\nOptions:")
    print("  --no-gui       - Disable visualization")
    print("  --eval-after   - Run evaluation after training")
    print("  --verbose      - Show detailed output")

def train_model(model_path=None, steps=500000, visualize=True, eval_after=False, verbose=False):
    """
    Train a new FANUC robot model with DirectML acceleration.
    
    This is a wrapper for the main implementation in src.core.training.train.
    
    Args:
        model_path: Path to save the model (default: auto-generated)
        steps: Number of training steps
        visualize: Whether to use the PyBullet GUI
        eval_after: Whether to run evaluation after training
        verbose: Whether to show detailed progress
        
    Returns:
        0 if successful, 1 otherwise
    """
    import warnings
    warnings.warn(
        "This train_model function is deprecated. "
        "Please use src.core.training.train.train_model instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import the real implementation
    from src.core.training.train import train_model as train_model_impl
    
    # Delegate to the real implementation
    return train_model_impl(
        model_path=model_path,
        steps=steps,
        visualize=visualize,
        eval_after=eval_after,
        verbose=verbose
    ) 