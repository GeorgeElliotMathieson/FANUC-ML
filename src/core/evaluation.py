"""
Evaluation functionality for FANUC Robot ML Platform.
"""

import os
import traceback
from src.core.utils import print_banner, ensure_model_file_exists

def print_eval_usage():
    """Print usage information for evaluation mode"""
    print("Evaluation Mode Usage:")
    print("  fanuc-platform eval <model-path> [episodes] [options]")
    print("")
    print("Arguments:")
    print("  model-path  - Path to the trained model file")
    print("  episodes    - Number of episodes to evaluate (default: 10)")
    print("")
    print("Options:")
    print("  --no-gui     - Disable visualization")
    print("  --verbose    - Enable verbose output")
    print("  --speed      - Set visualization speed (default: 0.02)")
    print("")

def run_evaluation(model_path, episodes=10, use_gui=True, verbose=False):
    """
    Run a thorough evaluation of a model with DirectML acceleration.
    
    Args:
        model_path: Path to the model to evaluate
        episodes: Number of evaluation episodes
        use_gui: Whether to use the PyBullet GUI
        verbose: Whether to show detailed progress
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Ensure model file exists
    model_path = ensure_model_file_exists(model_path)
    
    # Print banner with settings
    print_banner(f"Evaluating Model: {model_path} with DirectML")
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Episodes: {episodes}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print()
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Load DirectML components
    try:
        from src.dml import evaluate_model_directml
        
        # Check that the required module exists
        if 'evaluate_model_directml' not in globals() and not hasattr(evaluate_model_directml, '__call__'):
            print("ERROR: DirectML evaluation function not found.")
            print("This implementation requires AMD GPU with DirectML support.")
            return 1
        
        print("Loading model and environment...")
        
        # Run evaluation
        results = evaluate_model_directml(
            model_path=model_path,
            num_episodes=episodes
        )
        
        if results:
            print("\nEvaluation Results:")
            print(f"  Success Rate: {results['success_rate']:.1f}%")
            print(f"  Average Distance: {results['avg_distance']:.4f} meters")
            print(f"  Average Reward: {results['avg_reward']:.1f}")
            print(f"  Average Steps: {results['avg_steps']:.1f}")
        
        print("\nEvaluation completed!")
        return 0
    
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1 