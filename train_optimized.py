#!/usr/bin/env python3
"""
High-performance training script for FANUC robot using DirectML acceleration.
This script optimizes for maximum training performance on AMD GPUs.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set environment variables for DirectML and training optimization
os.environ["FANUC_DIRECTML"] = "1"
os.environ["USE_DIRECTML"] = "1"
os.environ["DIRECTML_ENABLE_TENSOR_CORES"] = "1"
os.environ["DIRECTML_GPU_TRANSFER_OPTIMIZATION"] = "1"
os.environ["DIRECTML_ENABLE_OPTIMIZATION"] = "1"
os.environ["DIRECTML_DISABLE_TRACING"] = "0"
os.environ["DIRECTML_DISABLE_PARALLELIZATION"] = "0"
os.environ["TORCH_COMPILE_MODE"] = "max-autotune"
os.environ["PYBULLET_FORCE_NOGL"] = "1"
os.environ["TRAINING_MODE"] = "1"

# Import PyTorch and set thread count
import torch
torch.set_num_threads(4)  # Limit CPU thread usage to prevent competition with GPU

# Import DirectML model
from src.core.training.train import train_model

# Helper function to check training mode
def is_training_mode():
    """Check if we're in training mode from environment variables"""
    return os.environ.get('TRAINING_MODE', '0').lower() in ('1', 'true', 'yes')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train FANUC robot model with optimized DirectML acceleration"
    )
    
    # Model and training parameters
    parser.add_argument(
        "model_dir", 
        type=str, 
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "timesteps", 
        type=int, 
        default=10000,
        help="Number of timesteps to train for"
    )
    
    # Performance parameters
    parser.add_argument(
        "--n-envs", 
        type=int, 
        default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-steps", 
        type=int, 
        default=2048,
        help="Number of steps per rollout"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-directml", 
        action="store_true",
        help="Disable DirectML acceleration and use CPU"
    )
    parser.add_argument(
        "--viz-speed", 
        type=float, 
        default=0.0,
        help="Visualization speed (0.0 for no visualization)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Print system info
    print("\n--- System Information ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Workspace directory: {os.getcwd()}")
    print(f"Is training mode: {is_training_mode()}")
    
    try:
        import torch_directml  # type: ignore
        print(f"DirectML version: {torch_directml.__version__ if hasattr(torch_directml, '__version__') else 'Unknown'}")
    except ImportError:
        print("DirectML not available")
    
    # Check for AMD GPU
    try:
        import subprocess
        gpu_info = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode("utf-8")
        print("Detected GPUs:")
        for line in gpu_info.strip().split("\n")[1:]:
            if line.strip():
                print(f"- {line.strip()}")
    except:
        print("Could not detect GPU information")
    
    # Print training parameters
    print("\n--- Training Parameters ---")
    print(f"Model directory: {args.model_dir}")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"N steps: {args.n_steps}")
    print(f"DirectML acceleration: {'Disabled' if args.no_directml else 'Enabled'}")
    print(f"Visualization speed: {args.viz_speed}")
    print(f"Verbose output: {args.verbose}")
    
    # Create output directory if needed
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up environment kwargs
    env_kwargs = {
        'training_mode': True,
        'viz_speed': args.viz_speed,
        'max_episode_steps': 150,
        'verbose': args.verbose
    }
    
    # Train the model
    print("\n--- Starting Training ---")
    start_time = time.time()
    
    model = train_model(
        model_dir=args.model_dir,
        num_timesteps=args.timesteps,
        seed=args.seed,
        use_directml=not args.no_directml,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        n_envs=args.n_envs,
        env_kwargs=env_kwargs,
        save_freq=1000,
        verbose=args.verbose,
        viz_speed=args.viz_speed
    )
    
    # Calculate training statistics
    total_time = time.time() - start_time
    steps_per_second = args.timesteps / total_time
    
    print("\n--- Training Complete ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Training speed: {steps_per_second:.2f} steps/second")
    print(f"Final model saved to: {os.path.join(args.model_dir, 'final_model')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 