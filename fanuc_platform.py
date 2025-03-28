#!/usr/bin/env python3
"""
Unified Entry Point for FANUC Robot ML Platform

This consolidated script provides a single entry point for all FANUC Robot ML Platform operations:
1. Training models (with or without DirectML acceleration)
2. Evaluating models (thorough evaluation)
3. Testing models (quick tests)
4. Installation verification and troubleshooting

This file combines functionality from:
- main.py (general framework entry point)
- directml_tools.py (DirectML-specific operations)
- tools/fanuc_tools.py (utility tools)
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

#########################################
# Common Utilities
#########################################

def print_banner(title):
    """Print a formatted banner with a title."""
    print("\n" + "="*80)
    print(f"FANUC Robot - {title}")
    print("="*80 + "\n")

def print_usage():
    """Print usage instructions for the script."""
    print("\nFANUC Robot ML Platform - Unified CLI")
    print("\nAvailable commands:")
    print("  python fanuc_platform.py train    - Train a model")
    print("  python fanuc_platform.py eval     - Evaluate a model thoroughly")
    print("  python fanuc_platform.py test     - Run a quick test of a model")
    print("  python fanuc_platform.py install  - Test installation")
    print("\nFor help on a specific command:")
    print("  python fanuc_platform.py train --help")
    print("  python fanuc_platform.py eval --help")
    print("  python fanuc_platform.py test --help")
    print("  python fanuc_platform.py install --help")
    print("\nDirectML acceleration:")
    print("  To use DirectML acceleration, add the --directml flag to any command")
    print("  Example: python fanuc_platform.py train --directml")

def is_directml_model(model_path):
    """
    Check if a model was trained with DirectML by looking at the file name pattern.
    """
    if model_path and 'directml' in model_path.lower():
        return True
    return False

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

def test_install(use_directml=False):
    """
    Test the installation of FANUC-ML package and its dependencies.
    
    Args:
        use_directml: Whether to include DirectML-specific checks
    
    Returns:
        0 if successful, 1 otherwise
    """
    print_banner("Installation Test" + (" with DirectML" if use_directml else ""))
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Required core modules
    core_modules = [
        "src",
        "src.core",
        "src.envs",
        "src.utils"
    ]
    
    # Add directml_core.py for DirectML mode
    if use_directml:
        core_modules.append("src.directml_core")
    
    # Dependencies
    dependencies = [
        "torch",
        "numpy",
        "pybullet",
        "gymnasium",
        "stable_baselines3",
        "matplotlib"
    ]
    
    # DirectML is required in DirectML mode, optional otherwise
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
    if use_directml:
        print("\nChecking DirectML support:")
        for module in directml_dependencies:
            if check_module(module):
                print(f"  ✓ {module}")
            else:
                print(f"  ✗ {module}")
                directml_ok = False
        
        if not directml_ok:
            print("\nERROR: DirectML dependencies are not installed.")
            print("This project requires AMD GPU acceleration. Install DirectML with:")
            print("pip install torch-directml")
            return 1
        
        # Test AMD GPU detection with DirectML
        try:
            print("\nTesting DirectML AMD GPU detection:")
            import torch
            import torch_directml
            
            # Check for DirectML devices
            device_count = torch_directml.device_count()
            if device_count > 0:
                dml_device = torch_directml.device()
                print(f"  ✓ Found {device_count} DirectML devices")
                print(f"  ✓ Using DirectML device: {dml_device}")
                
                # Create a test tensor on device
                test_tensor = torch.ones((2, 3), device=dml_device)
                print(f"  ✓ Successfully created tensor on DirectML device")
                
                # Access tensor to force execution
                _ = test_tensor.cpu().numpy()
                print(f"  ✓ Successfully executed operation on DirectML device")
            else:
                print(f"  ✗ No DirectML devices found")
                directml_ok = False
        except Exception as e:
            print(f"  ✗ DirectML initialization failed: {e}")
            directml_ok = False
        
        # Try importing key DirectML class
        print("\nTesting DirectML implementation access:")
        try:
            from src.directml_core import is_available, DirectMLPPO
            if is_available():
                print("  ✓ Successfully imported DirectML implementation")
            else:
                print("  ✗ DirectML is not available in the environment")
                directml_ok = False
        except ImportError:
            print("  ✗ Failed to import DirectML implementation")
            print(traceback.format_exc())
            directml_ok = False
    else:
        # In non-DirectML mode, check if DirectML is available as optional
        print("\nChecking optional DirectML support:")
        for module in directml_dependencies:
            if check_module(module):
                print(f"  ✓ {module} (optional)")
            else:
                print(f"  ✗ {module} (optional)")
                directml_ok = False
        
        if not directml_ok:
            print("\nNOTE: DirectML support is not available.")
            print("For AMD GPU acceleration, install: pip install torch-directml")
        else:
            print("\nDirectML support is available (optional)!")
    
    # Print summary
    print("\nInstallation test summary:")
    if core_ok and deps_ok:
        print("  ✓ Basic installation looks good!")
        if use_directml:
            if directml_ok:
                print("  ✓ DirectML support is available and working")
            else:
                print("  ✗ DirectML support is not working properly")
                return 1
        else:
            if directml_ok:
                print("  ✓ DirectML support is available (optional)")
            else:
                print("  ✗ DirectML support is not available (optional)")
    else:
        print("  ✗ Installation has issues that need to be fixed")
        return 1
    
    return 0

#########################################
# Evaluation Functionality
#########################################

def print_eval_usage():
    """Print usage instructions for evaluation."""
    print("\nEvaluation Usage:")
    print("  fanuc.bat eval <model_path> [--episodes N] [--no-gui] [--verbose]")
    print("\nRequired Arguments:")
    print("  model_path      Path to a trained model file (.pt or .zip)")
    print("\nOptional Arguments:")
    print("  --episodes N    Number of evaluation episodes (default: 10)")
    print("  --no-gui        Run without PyBullet GUI visualization")
    print("  --verbose, -v   Show detailed episode results")
    print("\nExamples:")
    print("  fanuc.bat eval models/fanuc-ppo-model")
    print("  fanuc.bat eval models/fanuc-directml-model --episodes 20 --no-gui")

def block_argparse():
    """Block argparse from parsing sys.argv and exiting."""
    saved_parse = argparse.ArgumentParser.parse_args
    saved_exit = sys.exit
    argparse.ArgumentParser.parse_args = lambda self, *args, **kwargs: None
    sys.exit = lambda *args, **kwargs: None
    return saved_parse, saved_exit

def restore_argparse(saved_parse, saved_exit):
    """Restore argparse functionality."""
    argparse.ArgumentParser.parse_args = saved_parse
    sys.exit = saved_exit

def run_evaluation(model_path, episodes=10, use_gui=True, verbose=False, use_directml=False):
    """
    Run evaluation on a trained model.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of episodes to run
        use_gui: Whether to use the PyBullet GUI
        verbose: Whether to print detailed results
        use_directml: Whether to use DirectML for evaluation
    
    Returns:
        0 if successful, 1 otherwise
    """
    # Ensure that a valid model file exists
    model_path = ensure_model_file_exists(model_path)
    if not model_path:
        return 1
    
    # Print banner with settings
    title = f"Evaluating Model: {os.path.basename(model_path)}"
    if use_directml:
        title += " with DirectML"
    print_banner(title)
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Episodes: {episodes}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print(f"  DirectML: {'Enabled' if use_directml else 'Disabled'}")
    print()
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Use the appropriate evaluation function based on DirectML flag
    try:
        print("Loading model and environment...")
        if use_directml:
            # Use DirectML evaluation
            from src.directml_core import evaluate_model_directml
            
            # Check if DirectML is available
            from src.directml_core import is_available
            if not is_available():
                print("ERROR: DirectML is not available in this environment.")
                print("Install DirectML with: pip install torch-directml")
                return 1
            
            # Evaluate with DirectML
            results = evaluate_model_directml(model_path, num_episodes=episodes)
        else:
            # Use regular evaluation
            from src.envs import FanucReachEnv
            from src.core import evaluate_model
            
            # Evaluate with standard PyTorch
            results = evaluate_model(model_path, env_class=FanucReachEnv, num_episodes=episodes)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"  Success Rate: {results['success_rate']:.2f}%")
        print(f"  Average Distance: {results['avg_distance']:.4f} m")
        print(f"  Average Reward: {results['avg_reward']:.2f}")
        print(f"  Average Steps: {results['avg_steps']:.1f}")
        
        if verbose:
            print("\nEpisode details:")
            for i, (success, distance, reward, steps) in enumerate(zip(
                    results['successes'], results['distances'],
                    results['rewards'], results['steps'])):
                print(f"  Episode {i+1}: {'✓' if success else '✗'} "
                      f"Distance={distance:.4f}m, Reward={reward:.2f}, Steps={steps}")
        
        # Provide success/failure summary
        print("\nSummary:")
        if results['success_rate'] >= 80:
            print("  ✓ Model performs well with high success rate!")
        elif results['success_rate'] >= 50:
            print("  ⚠ Model performs moderately well.")
        else:
            print("  ✗ Model performs poorly.")
        
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
    """Print usage instructions for testing."""
    print("\nTesting Usage:")
    print("  fanuc.bat test <model_path> [--episodes N] [--no-gui] [--verbose]")
    print("\nRequired Arguments:")
    print("  model_path      Path to a trained model file (.pt or .zip)")
    print("\nOptional Arguments:")
    print("  --episodes N    Number of test episodes (default: 1)")
    print("  --no-gui        Run without PyBullet GUI visualization")
    print("  --verbose, -v   Show detailed episode results")
    print("\nExamples:")
    print("  fanuc.bat test models/fanuc-ppo-model")
    print("  fanuc.bat test models/fanuc-directml-model --episodes 5 --no-gui")
 
def run_test(model_path, episodes=1, use_gui=True, verbose=False, use_directml=False):
    """
    Run testing on a trained model (interactive visualization).
    
    Args:
        model_path: Path to the trained model
        episodes: Number of episodes to run
        use_gui: Whether to use the PyBullet GUI
        verbose: Whether to print detailed results
        use_directml: Whether to use DirectML for testing
    
    Returns:
        0 if successful, 1 otherwise
    """
    # Ensure that a valid model file exists
    model_path = ensure_model_file_exists(model_path)
    if not model_path:
        return 1
    
    # Print banner with settings
    title = f"Testing Model: {os.path.basename(model_path)}"
    if use_directml:
        title += " with DirectML"
    print_banner(title)
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Episodes: {episodes}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print(f"  DirectML: {'Enabled' if use_directml else 'Disabled'}")
    print()
    
    if not use_gui:
        print("WARNING: Testing without GUI may not be helpful.")
        print("Consider running with GUI enabled for better visualization.")
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Use the appropriate test function based on DirectML flag
    try:
        print("Loading model and environment...")
        if use_directml:
            # Use DirectML for testing
            from src.directml_core import test_model_directml
            
            # Check if DirectML is available
            from src.directml_core import is_available
            if not is_available():
                print("ERROR: DirectML is not available in this environment.")
                print("Install DirectML with: pip install torch-directml")
                return 1
            
            # Test with DirectML
            test_model_directml(model_path, num_episodes=episodes)
        else:
            # Use regular PyTorch for testing
            from src.envs import FanucReachEnv
            from src.core import test_model
            
            # Test with standard PyTorch
            test_model(model_path, env_class=FanucReachEnv, num_episodes=episodes)
        
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
    """Print usage instructions for training."""
    print("\nTraining Usage:")
    print("  fanuc.bat train [options]")
    print("\nOptions:")
    print("  --model_path NAME   Custom name/path for the trained model")
    print("  --steps N           Number of training steps (default: 500000)")
    print("  --no-gui            Run without PyBullet GUI visualization")
    print("  --eval              Run evaluation after training")
    print("  --verbose, -v       Show detailed training progress")
    print("\nExamples:")
    print("  fanuc.bat train")
    print("  fanuc.bat train --model_path my_model --steps 1000000 --no-gui")
    print("  fanuc.bat train --directml  # Train with DirectML for AMD GPUs")

def train_model(model_path=None, steps=500000, use_gui=True, eval_after=False, 
                verbose=False, use_directml=False):
    """
    Train a new FANUC robot model.
    
    Args:
        model_path: Path to save the model (default: auto-generated)
        steps: Number of training steps
        use_gui: Whether to use the PyBullet GUI
        eval_after: Whether to run evaluation after training
        verbose: Whether to show detailed progress
        use_directml: Whether to use DirectML for training
    
    Returns:
        0 if successful, 1 otherwise
    """
    # Generate a default model path if not provided
    if not model_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/fanuc-{timestamp}"
        if use_directml:
            model_path += "-directml"
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Print banner with settings
    title = "Training FANUC Robot Model"
    if use_directml:
        title += " with DirectML"
    print_banner(title)
    
    print("Settings:")
    print(f"  Model Path: {model_path}")
    print(f"  Training Steps: {steps}")
    print(f"  GUI: {'Enabled' if use_gui else 'Disabled'}")
    print(f"  Evaluation After: {'Yes' if eval_after else 'No'}")
    print(f"  Verbose: {'Enabled' if verbose else 'Disabled'}")
    print(f"  DirectML: {'Enabled' if use_directml else 'Disabled'}")
    print()
    
    # Set environment variables
    os.environ['FANUC_GUI'] = '1' if use_gui else '0'
    os.environ['FANUC_VERBOSE'] = '1' if verbose else '0'
    
    # Train with the appropriate function based on DirectML flag
    try:
        if use_directml:
            # Use DirectML for training
            from src.directml_core import train_robot_with_ppo_directml
            
            # Check if DirectML is available
            from src.directml_core import is_available
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
        else:
            # Use regular PyTorch for training
            from src.envs import FanucReachEnv
            from src.core import train_robot_with_ppo
            
            print("Starting training with PyTorch...")
            model = train_robot_with_ppo(
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
                verbose=verbose,
                use_directml=use_directml
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
    # Define the base parser with shared arguments
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--directml", action="store_true", 
                             help="Use DirectML acceleration for AMD GPUs")
    
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="FANUC Robot ML Platform - Unified CLI"
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Train mode parser
    train_parser = subparsers.add_parser("train", parents=[base_parser],
                                        help="Train a model")
    train_parser.add_argument("model_path", nargs="?", help="Path to load existing model (optional)")
    train_parser.add_argument("steps", nargs="?", type=int, default=1000000, 
                             help="Number of training steps")
    train_parser.add_argument("--save", type=str, help="Path to save the trained model")
    train_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    train_parser.add_argument("--viz-speed", type=float, default=0.02, 
                            help="Visualization speed")
    train_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--parallel", type=int, default=1, 
                            help="Number of parallel environments")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, 
                            help="Learning rate")
    
    # Eval mode parser
    eval_parser = subparsers.add_parser("eval", parents=[base_parser],
                                       help="Evaluate a model thoroughly")
    eval_parser.add_argument("model_path", help="Path to trained model")
    eval_parser.add_argument("episodes", nargs="?", type=int, default=5, 
                           help="Number of evaluation episodes")
    eval_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    eval_parser.add_argument("--speed", type=float, default=0.02, 
                          help="Visualization speed")
    eval_parser.add_argument("--save-video", action="store_true", 
                           help="Save a video of the evaluation")
    
    # Test mode parser
    test_parser = subparsers.add_parser("test", parents=[base_parser],
                                       help="Run a quick test of a model")
    test_parser.add_argument("model_path", help="Path to trained model")
    test_parser.add_argument("episodes", nargs="?", type=int, default=1, 
                           help="Number of test episodes")
    test_parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    test_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    test_parser.add_argument("--speed", type=float, default=0.02, 
                          help="Visualization speed")
    test_parser.add_argument("--save-video", action="store_true", 
                           help="Save a video of the test")
    
    # Install mode parser
    install_parser = subparsers.add_parser("install", parents=[base_parser],
                                         help="Test installation")
    
    # Parse args
    args = parser.parse_args()
    
    # If no mode specified, print usage and exit
    if not hasattr(args, 'mode') or not args.mode:
        parser.print_help()
        sys.exit(0)
    
    return args

def main():
    """
    Main entry point for the unified FANUC Robot ML Platform.
    
    Parses command line arguments and dispatches to the appropriate functions
    based on mode and DirectML availability.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Handle each mode
    if args.mode == "install":
        return test_install(use_directml=args.directml)
    
    elif args.mode == "train":
        return train_model(
            model_path=args.model_path,
            steps=args.steps,
            use_gui=not args.no_gui,
            eval_after=args.eval,
            verbose=args.verbose,
            use_directml=args.directml
        )
    
    elif args.mode == "eval":
        return run_evaluation(
            model_path=args.model_path,
            episodes=args.episodes,
            use_gui=not args.no_gui,
            verbose=args.verbose,
            use_directml=args.directml
        )
    
    elif args.mode == "test":
        return run_test(
            model_path=args.model_path,
            episodes=args.episodes,
            use_gui=not args.no_gui,
            verbose=args.verbose,
            use_directml=args.directml
        )
    
    else:
        print(f"Unknown mode: {args.mode}")
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 