#!/usr/bin/env python3
"""
Main entry point for the FANUC Robot Control and Training Platform.

This script serves as the unified interface for training, evaluating, and
running demonstrations using the reinforcement learning framework for 
FANUC robots. It handles command line arguments, sets up the environment,
and dispatches to the appropriate functions based on user input.
"""

import os
import sys
import time
import argparse

def print_banner(title):
    """Print a formatted banner with a title."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"| {title} |")
    print(f"{border}\n")

def print_usage():
    """Print usage instructions"""
    print("\nUsage:")
    print("  python main.py [options]")
    print("\nFor help:")
    print("  python main.py --help")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FANUC Robot Training and Control Platform"
    )
    
    # Mode selection arguments
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument("--train", action="store_true", help="Train a new model")
    mode_group.add_argument("--eval", action="store_true", help="Evaluate an existing model")
    mode_group.add_argument("--demo", action="store_true", help="Run a demo of an existing model")
    
    # Model loading/saving arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--load", type=str, help="Path to load an existing model from")
    model_group.add_argument("--save", type=str, default="./models/fanuc-ml", 
                         help="Path to save the trained model to")
    
    # Training parameters
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--steps", type=int, default=1000000, 
                          help="Number of training timesteps")
    train_group.add_argument("--eval-episodes", type=int, default=10,
                          help="Number of episodes to evaluate on")
    
    # Environment options
    env_group = parser.add_argument_group("Environment Options")
    env_group.add_argument("--no-gui", action="store_true", help="Disable GUI")
    env_group.add_argument("--viz-speed", type=float, default=0.02,
                        help="Visualization speed (smaller is faster)")
    
    # Miscellaneous options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--verbose", action="store_true", help="Enable verbose output")
    misc_group.add_argument("--seed", type=int, help="Random seed")
    
    # DirectML is now always enabled - no need for an option
    
    return parser.parse_args()

def main():
    """
    Main entry point for the application.
    
    Parses command line arguments and dispatches to the appropriate functions.
    """
    # Add the project directory to Python path to ensure imports work
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Parse command line arguments
    args = parse_args()
    
    # Seed if requested
    if args.seed is not None:
        from src.utils import set_seed
        set_seed(args.seed)
    
    # Check if we're in evaluation or demo mode
    if args.eval or args.demo:
        if args.load is None:
            print("Error: Must specify a model path with --load when using --eval or --demo")
            return 1
    
    try:
        # Import DirectML-specific functionality
        from src.directml import DIRECTML_AVAILABLE, directml_main
        
        if not DIRECTML_AVAILABLE:
            print("Error: DirectML support is required but torch_directml is not installed.")
            print("Please install with 'pip install torch-directml'")
            return 1
        
        # Run with DirectML
        print_banner("FANUC Robot Control with DirectML Acceleration")
        return directml_main(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 