"""
Training functionality for FANUC Robot ML Platform.
"""

import os
import time
import traceback
from src.core.utils import print_banner
from src.core.evaluation import run_evaluation

def print_train_usage():
    """Print usage instructions for the train command."""
    print("Usage: python fanuc_platform.py train [model_path] [steps] [options]")
    print("\nArguments:")
    print("  model_path    - Path to save the model (optional)")
    print("  steps         - Number of training steps (default: 500000)")
    print("\nOptions:")
    print("  --no-gui       - Disable GUI")
    print("  --eval         - Run evaluation after training")
    print("  --verbose      - Show detailed output")

def train_model(model_path=None, steps=500000, use_gui=True, eval_after=False, verbose=False):
    """
    Train a new FANUC robot model with DirectML acceleration.
    
    Args:
        model_path: Path to save the model (default: auto-generated)
        steps: Number of training steps
        use_gui: Whether to use the PyBullet GUI
        eval_after: Whether to run evaluation after training
        verbose: Whether to show detailed progress
        
    Returns:
        0 if successful, 1 otherwise
    """
    # Generate a default model path if not provided
    if not model_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/fanuc-{timestamp}-directml"
    
    # Create models directory if it doesn't exist
    if "/" in model_path or "\\" in model_path:
        directory = os.path.dirname(model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    else:
        # If there's no directory in the path, ensure the models directory exists
        os.makedirs("models", exist_ok=True)
    
    # Print banner with settings
    print_banner("Training FANUC Robot Model with DirectML")
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Training Steps: {steps}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Evaluation After: {'Yes' if eval_after else 'No'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print()
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Train with DirectML
    try:
        # Use DirectML for training
        from src.dml import train_robot_with_ppo_directml
        
        # Check if DirectML is available
        from src.dml import is_available
        if not is_available():
            print("ERROR: DirectML is not available in this environment.")
            print("Install DirectML with: pip install torch-directml")
            return 1
        
        print("Starting training with DirectML...")
        model = train_robot_with_ppo_directml(
            total_timesteps=steps,
            model_path=model_path,
            verbose=verbose
        )
        
        print(f"\nTraining completed! Model saved to: {model_path}")
        
        # Run evaluation if requested
        if eval_after:
            print("\nRunning evaluation on trained model...")
            return run_evaluation(
                model_path=model_path,
                episodes=10,
                use_gui=use_gui,
                verbose=verbose
            )
        
        return 0
    
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        if verbose:
            print(traceback.format_exc())
        return 1 