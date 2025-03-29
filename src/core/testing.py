"""
Testing functionality for FANUC Robot ML Platform.
"""

import os
import traceback
from src.core.utils import print_banner, ensure_model_file_exists

def print_test_usage():
    """Print usage instructions for the test command."""
    print("Usage: python fanuc_platform.py test <model_path> [num_episodes] [options]")
    print("\nArguments:")
    print("  model_path    - Path to trained model")
    print("  num_episodes  - Number of episodes (default: 1)")
    print("\nOptions:")
    print("  --no-gui       - Disable visualization")
    print("  --verbose      - Show detailed output")
    print("  --speed=<val>  - Visualization speed (default: 0.02)")
    print("  --max-steps    - Maximum steps per episode before timeout (default: 1000)")

def run_test(model_path, num_episodes=1, visualize=True, verbose=False, max_steps=1000, viz_speed=0.02):
    """
    Run a quick test of a model with DirectML acceleration.
    
    Args:
        model_path: Path to the model to test
        num_episodes: Number of test episodes
        visualize: Whether to use visualization
        verbose: Whether to show detailed progress
        max_steps: Maximum number of steps per episode before timing out
        viz_speed: Visualization speed (seconds per step)
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Ensure model file exists
    model_path = ensure_model_file_exists(model_path)
    
    # Print banner with settings
    print_banner(f"Testing Model: {model_path} with DirectML")
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Num Episodes: {num_episodes}")
    print(f"  Max Steps Per Episode: {max_steps}")
    print(f"  Visualization: {'Enabled' if visualize else 'Disabled'}")
    print(f"  Visualization Speed: {viz_speed if visualize else 0.0}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print()
    
    # Set environment variables for child processes and components that read them
    # These are used by functions like get_visualization_settings_from_env
    os.environ['FANUC_VISUALIZE'] = '1' if visualize else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    try:
        # First check if DirectML is available
        from src.dml import is_available
        if not is_available():
            print("\nERROR: DirectML is not available for testing.")
            print("This test implementation requires an AMD GPU with DirectML support.")
            print("Please verify your DirectML installation with: python fanuc_platform.py install")
            return 1
            
        # Import the test function
        from src.dml import test_model_directml
        
        print("Loading model and environment...")
        
        # Run test
        test_model_directml(
            model_path=model_path,
            num_episodes=num_episodes,
            visualize=visualize,
            verbose=verbose,
            max_steps=max_steps,
            viz_speed=viz_speed
        )
        
        print("\nTest completed!")
        return 0
    
    except ImportError as e:
        print(f"\nERROR: Required modules not found: {e}")
        print("This implementation requires DirectML. Please run:")
        print("  python fanuc_platform.py install")
        if verbose:
            print(traceback.format_exc())
        return 1
    except Exception as e:
        print(f"\nERROR: Testing failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1 