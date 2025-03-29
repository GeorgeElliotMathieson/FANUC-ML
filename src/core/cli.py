"""
Command-line interface functionality for FANUC Robot ML Platform.
"""

import sys
import argparse
from src.core.utils import print_usage

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
    subparsers = parser.add_subparsers(dest="command", help="Operation mode")
    
    # Train mode parser
    train_parser = subparsers.add_parser("train", 
                                        help="Train a model with DirectML",
                                        description="Train a FANUC robot model using DirectML acceleration")
    train_parser.add_argument("model_path", nargs="?", help="Path to save the model (optional)")
    train_parser.add_argument("steps", nargs="?", type=int, default=500000, 
                            help="Number of training steps")
    train_parser.add_argument("--no-gui", action="store_true", help="Disable visualization")
    train_parser.add_argument("--eval-after", action="store_true", help="Run evaluation after training")
    train_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Eval mode parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model with DirectML")
    eval_parser.add_argument("model_path", help="Path to trained model")
    eval_parser.add_argument("num_episodes", nargs="?", type=int, default=10, 
                          help="Number of evaluation episodes")
    eval_parser.add_argument("--no-gui", action="store_true", help="Disable visualization")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    eval_parser.add_argument("--speed", type=float, default=0.02, 
                         help="Visualization speed")
    eval_parser.add_argument("--max-steps", type=int, default=1000,
                         help="Maximum steps per episode before timing out")
    
    # Test mode parser
    test_parser = subparsers.add_parser("test", help="Test a model with DirectML")
    test_parser.add_argument("model_path", help="Path to trained model")
    test_parser.add_argument("num_episodes", nargs="?", type=int, default=1, 
                          help="Number of test episodes")
    test_parser.add_argument("--no-gui", action="store_true", help="Disable visualization")
    test_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    test_parser.add_argument("--speed", type=float, default=0.02, 
                         help="Visualization speed")
    test_parser.add_argument("--max-steps", type=int, default=1000,
                          help="Maximum steps per episode before timing out")
    
    # Install mode parser
    install_parser = subparsers.add_parser("install", help="Test DirectML installation")
    
    # Parse args
    args = parser.parse_args()
    
    # If no mode specified, print usage and exit
    if not hasattr(args, 'command') or not args.command:
        print_usage()
        sys.exit(0)
    
    return args 