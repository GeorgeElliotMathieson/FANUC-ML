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

# Set environment variables for DirectML
os.environ['FANUC_DIRECTML'] = '1'
os.environ['USE_DIRECTML'] = '1'
os.environ['USE_GPU'] = '1'

# Import functionality from refactored modules
from src.core.utils import print_banner, print_usage
from src.core.install import test_install
from src.core.evaluation import evaluate_model_wrapper as run_evaluation
from src.core.testing import run_test, print_test_usage
from src.core.training import train_model
from src.core.cli import parse_args

# Import print usage functions
from src.core.evaluation.evaluate import print_eval_usage
from src.core.training.train import print_train_usage

def main():
    """Main entry point for the FANUC Robot ML Platform."""
    args = parse_args()
    
    if args.command == 'train':
        return train_model(
            model_path=args.model_path,
            steps=args.steps,
            use_gui=not args.no_gui,
            eval_after=args.eval_after,
            verbose=args.verbose
        )
    elif args.command == 'eval':
        return run_evaluation(
            model_path=args.model_path,
            num_episodes=args.episodes,
            visualize=not args.no_gui,
            verbose=args.verbose
        )
    elif args.command == 'test':
        return run_test(
            model_path=args.model_path,
            episodes=args.episodes,
            use_gui=not args.no_gui,
            verbose=args.verbose
        )
    elif args.command == 'install':
        return test_install()
    else:
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 