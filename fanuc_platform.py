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
import argparse

# Add the project root directory to sys.path
project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import the utility first so we can use it to set environment variables
from src.core.utils import get_directml_settings_from_env

# Set environment variables for DirectML if not already set
if not get_directml_settings_from_env():
    os.environ['FANUC_DIRECTML'] = '1'
    os.environ['USE_DIRECTML'] = '1'
    os.environ['USE_GPU'] = '1'

# Import functionality from refactored modules
from src.core.utils import print_banner, print_usage
from src.core.install import test_install
from src.core.evaluation.evaluate import evaluate_model_wrapper
from src.core.testing import run_test, print_test_usage
from src.core.training.train import train_model
from src.core.cli import parse_args

# Import print usage functions
from src.core.evaluation.evaluate import print_eval_usage
from src.core.training.train import print_train_usage

def main():
    """Main entry point for the FANUC Robot ML Platform."""
    args = parse_args()
    
    if args.command == 'train':
        # Import the implementation directly
        from src.core.training.train import train_model as train_model_impl
        
        # Call the implementation directly with the correct parameter names
        return train_model_impl(
            model_dir=args.model_path,
            num_timesteps=args.steps,
            viz_speed=0.02 if not args.no_gui else 0.0,
            eval_freq=10000 if args.eval_after else 0,
            verbose=args.verbose
        )
    elif args.command == 'eval':
        results = evaluate_model_wrapper(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            visualize=not args.no_gui,
            verbose=args.verbose,
            max_steps=getattr(args, 'max_steps', 1000),
            viz_speed=getattr(args, 'speed', 0.02)
        )
        # Return 0 if successful (results is not None), 1 otherwise
        return 0 if results is not None else 1
    elif args.command == 'test':
        return run_test(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            visualize=not args.no_gui,
            verbose=args.verbose,
            max_steps=getattr(args, 'max_steps', 1000),
            viz_speed=getattr(args, 'speed', 0.02)
        )
    elif args.command == 'install':
        return test_install()
    else:
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 