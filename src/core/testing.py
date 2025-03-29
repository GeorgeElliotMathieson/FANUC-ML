"""
Testing functionality for FANUC Robot ML Platform.
"""

import os
import traceback
from src.core.utils import print_banner, ensure_model_file_exists

def print_test_usage():
    """Print usage instructions for the test command."""
    print("Usage: python fanuc_platform.py test <model_path> [episodes] [options]")
    print("\nArguments:")
    print("  model_path    - Path to trained model")
    print("  episodes      - Number of episodes (default: 1)")
    print("\nOptions:")
    print("  --no-gui       - Disable GUI")
    print("  --verbose      - Show detailed output")
    print("  --speed=<val>  - Visualization speed (default: 0.02)")

def run_test(model_path, episodes=1, use_gui=True, verbose=False):
    """
    Run a quick test of a model with DirectML acceleration.
    
    Args:
        model_path: Path to the model to test
        episodes: Number of test episodes
        use_gui: Whether to use the PyBullet GUI
        verbose: Whether to show detailed progress
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Ensure model file exists
    model_path = ensure_model_file_exists(model_path)
    
    # Print banner with settings
    print_banner(f"Testing Model: {model_path} with DirectML")
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Episodes: {episodes}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print()
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    try:
        from src.dml import test_model_directml
        
        print("Loading model and environment...")
        
        # Run test
        test_model_directml(
            model_path=model_path,
            num_episodes=episodes
        )
        
        print("\nTest completed!")
        return 0
    
    except Exception as e:
        print(f"\nERROR: Testing failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1 