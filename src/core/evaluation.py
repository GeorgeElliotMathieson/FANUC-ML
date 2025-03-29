"""
Evaluation functionality for FANUC Robot ML Platform.
"""

import os
import traceback
from src.core.utils import print_banner, ensure_model_file_exists

def print_eval_usage():
    """Print usage information for evaluation mode"""
    print("Evaluation Mode Usage:")
    print("  fanuc-platform eval <model-path> [num_episodes] [options]")
    print("")
    print("Arguments:")
    print("  model-path    - Path to the trained model file")
    print("  num_episodes  - Number of episodes to evaluate (default: 10)")
    print("")
    print("Options:")
    print("  --no-gui       - Disable visualization")
    print("  --verbose      - Enable verbose output")
    print("  --speed        - Set visualization speed (default: 0.02)")
    print("  --max-steps    - Maximum steps per episode before timeout (default: 1000)")
    print("")

def run_evaluation(model_path, num_episodes=10, visualize=True, verbose=False, max_steps=1000, viz_speed=0.02):
    """
    Run a thorough evaluation of a model with DirectML acceleration.
    
    Args:
        model_path: Path to the model to evaluate
        num_episodes: Number of evaluation episodes
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
    print_banner(f"Evaluating Model: {model_path} with DirectML")
    
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
    
    # Load DirectML components
    try:
        # First check if DirectML is available
        from src.dml import is_available
        if not is_available():
            print("\nERROR: DirectML is not available for evaluation.")
            print("This implementation requires an AMD GPU with DirectML support.")
            print("Please install DirectML with: pip install torch-directml")
            print("Then verify your installation with: python fanuc_platform.py install")
            return 1
            
        # Import the function from evaluation module
        from src.core.evaluation.evaluate import evaluate_model_directml
        
        # Run evaluation with max_steps parameter
        results = evaluate_model_directml(
            model_path=model_path,
            num_episodes=num_episodes,
            visualize=visualize,
            verbose=verbose,
            max_steps=max_steps,
            viz_speed=viz_speed
        )
        
        if results:
            print("\nEvaluation Results:")
            print(f"  Success Rate: {results['success_rate']:.1f}%")
            print(f"  Average Distance: {results['avg_distance']:.4f} meters")
            print(f"  Average Reward: {results['avg_reward']:.1f}")
            print(f"  Average Steps: {results['avg_steps']:.1f}")
        
        print("\nEvaluation completed!")
        return 0
    
    except ImportError as e:
        print(f"\nERROR: Required modules not found: {e}")
        print("This implementation requires DirectML. Please run:")
        print("  python fanuc_platform.py install")
        if verbose:
            print(traceback.format_exc())
        return 1
    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1 