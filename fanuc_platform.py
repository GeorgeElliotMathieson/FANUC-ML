#!/usr/bin/env python3
"""
FANUC Robot ML Platform - Unified Implementation (DirectML Edition)

This script provides the implementation for all FANUC Robot ML Platform operations,
with a single entry point through fanuc.bat:

1. Training models with DirectML acceleration
2. Evaluating models with DirectML acceleration
3. Testing models with DirectML acceleration
4. Installation verification and DirectML troubleshooting

This implementation is optimized specifically for AMD GPUs with DirectML acceleration.
"""

import os
import sys
import time
import argparse
import traceback
import importlib
from typing import Optional, Any, Dict, List, Tuple, Union

# Add the project root directory to sys.path
project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Set environment variables for DirectML
os.environ['FANUC_DIRECTML'] = '1'
os.environ['USE_DIRECTML'] = '1'
os.environ['USE_GPU'] = '1'

#########################################
# Common Utilities
#########################################

def print_banner(title):
    """Print a formatted banner with a title."""
    print("\n" + "="*80)
    print(f"FANUC Robot with DirectML - {title}")
    print("="*80 + "\n")

def print_usage():
    """Print usage instructions for the script."""
    print("\nFANUC Robot ML Platform - DirectML CLI")
    print("\nAvailable commands:")
    print("  python fanuc_platform.py train    - Train a model with DirectML")
    print("  python fanuc_platform.py eval     - Evaluate a model thoroughly with DirectML")
    print("  python fanuc_platform.py test     - Run a quick test of a model with DirectML")
    print("  python fanuc_platform.py install  - Test DirectML installation")
    print("\nFor help on a specific command:")
    print("  python fanuc_platform.py train --help")
    print("  python fanuc_platform.py eval --help")
    print("  python fanuc_platform.py test --help")
    print("  python fanuc_platform.py install --help")

def ensure_model_file_exists(model_path: str) -> str:
    """
    Check if model file exists and handle .pt extension if needed.
    Returns the corrected path.
    """
    if not model_path:
        print("ERROR: No model path specified.")
        return model_path
        
    # Check if the file exists as is
    if os.path.exists(model_path):
        return model_path
        
    # Try with .pt extension
    if os.path.exists(model_path + ".pt"):
        model_path += ".pt"
        print(f"Using model file with .pt extension: {model_path}")
        return model_path
    
    # Try looking in the models directory
    model_dir_path = os.path.join("models", model_path)
    if os.path.exists(model_dir_path):
        return model_dir_path
    if os.path.exists(model_dir_path + ".pt"):
        return model_dir_path + ".pt"
    
    # Try with timestamp directories
    model_files = []
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pt") or file.endswith(".zip"):
                if model_path in os.path.join(root, file):
                    model_files.append(os.path.join(root, file))
    
    if model_files:
        # Return the most recent match
        print(f"Found {len(model_files)} possible model files matching {model_path}")
        print(f"Using: {model_files[0]}")
        return model_files[0]
    
    # Neither exists
    print(f"Warning: Model file not found at {model_path}")
    return model_path

#########################################
# Installation Testing Functionality
#########################################

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"Cannot import {module_name}: {e}")
        return False

def test_install():
    """
    Test the installation of FANUC-ML package and DirectML dependencies.
    
    Returns:
        0 if successful, 1 otherwise
    """
    print_banner("DirectML Installation Test")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Required core modules
    core_modules = [
        "src",
        "src.core",
        "src.envs",
        "src.utils",
        "src.dml"
    ]
    
    # Dependencies
    dependencies = [
        "torch",
        "numpy",
        "pybullet",
        "gymnasium",
        "stable_baselines3",
        "matplotlib"
    ]
    
    # DirectML is required
    directml_dependencies = [
        "torch_directml"
    ]
    
    # Check core modules
    print("Checking core modules:")
    core_ok = True
    for module in core_modules:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            core_ok = False
    
    if not core_ok:
        print("\nWARNING: Some core modules are not installed correctly.")
        print("Try reinstalling the package with: pip install -e .")
    else:
        print("\nAll core modules are installed correctly!")
    
    # Check required dependencies
    print("\nChecking required dependencies:")
    deps_ok = True
    for module in dependencies:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            deps_ok = False
    
    if not deps_ok:
        print("\nWARNING: Some dependencies are missing.")
        print("Try reinstalling with: pip install -r requirements.txt")
    else:
        print("\nAll required dependencies are installed correctly!")
    
    # Check DirectML
    directml_ok = True
    print("\nChecking DirectML support:")
    for module in directml_dependencies:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            directml_ok = False
    
    if not directml_ok:
        print("\nERROR: DirectML support is not available.")
        print("This implementation requires AMD GPU with DirectML support.")
        print("Install with: pip install torch-directml")
        return 1
    
    # Test DirectML
    print("\nTesting DirectML AMD GPU detection:")
    try:
        import torch
        import torch_directml
        
        # Check for DirectML devices
        device_count = torch_directml.device_count()
        print(f"  ✓ Found {device_count} DirectML device(s)")
        
        # Try to initialize a DirectML device
        device = torch_directml.device()
        print(f"  ✓ Successfully initialized DirectML device: {device}")
        
        # Check for AMD GPU info
        from src.dml import setup_directml
        dml_device = setup_directml()
        
        print("\nDirectML setup successful!")
        
    except Exception as e:
        print(f"\nERROR: DirectML initialization failed: {e}")
        if "trace" in locals():
            print(traceback.format_exc())
        return 1

    if all([core_ok, deps_ok, directml_ok]):
        print("\n✅ All checks passed! DirectML support is ready to use.")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above before proceeding.")
        return 1

#########################################
# Evaluation Functionality
#########################################

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

#########################################
# Testing Functionality
#########################################

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

#########################################
# Training Functionality
#########################################

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

#########################################
# CLI Parser and Main Entry Point
#########################################

def parse_args():
    """
    Parse command line arguments based on the mode (train, eval, test, install).
    
    Returns:
        Parsed arguments namespace
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="FANUC Robot ML Platform - DirectML CLI"
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Train mode parser
    train_parser = subparsers.add_parser("train", 
                                         help="Train a model with DirectML",
                                         description="Train a FANUC robot model using DirectML acceleration")
    train_parser.add_argument("model_path", nargs="?", help="Path to save the model (optional)")
    train_parser.add_argument("steps", nargs="?", type=int, default=500000, 
                             help="Number of training steps")
    train_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    train_parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    train_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Eval mode parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model with DirectML")
    eval_parser.add_argument("model_path", help="Path to trained model")
    eval_parser.add_argument("episodes", nargs="?", type=int, default=10, 
                           help="Number of evaluation episodes")
    eval_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    eval_parser.add_argument("--speed", type=float, default=0.02, 
                          help="Visualization speed")
    
    # Test mode parser
    test_parser = subparsers.add_parser("test", help="Test a model with DirectML")
    test_parser.add_argument("model_path", help="Path to trained model")
    test_parser.add_argument("episodes", nargs="?", type=int, default=1, 
                           help="Number of test episodes")
    test_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    test_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    test_parser.add_argument("--speed", type=float, default=0.02, 
                          help="Visualization speed")
    
    # Install mode parser
    install_parser = subparsers.add_parser("install", help="Test DirectML installation")
    
    # Parse args
    args = parser.parse_args()
    
    # If no mode specified, print usage and exit
    if not hasattr(args, 'mode') or not args.mode:
        parser.print_help()
        sys.exit(0)
    
    return args

def main():
    """
    Main entry point for the DirectML-focused FANUC Robot ML Platform.
    
    Parses command line arguments and dispatches to the appropriate functions.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Handle each mode
    if args.mode == "install":
        return test_install()
    
    elif args.mode == "train":
        return train_model(
            model_path=args.model_path,
            steps=args.steps,
            use_gui=not args.no_gui,
            eval_after=args.eval,
            verbose=args.verbose
        )
    
    elif args.mode == "eval":
        return run_evaluation(
            model_path=args.model_path,
            episodes=args.episodes,
            use_gui=not args.no_gui,
            verbose=args.verbose
        )
    
    elif args.mode == "test":
        return run_test(
            model_path=args.model_path,
            episodes=args.episodes,
            use_gui=not args.no_gui,
            verbose=args.verbose
        )
    
    else:
        print(f"Unknown mode: {args.mode}")
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 