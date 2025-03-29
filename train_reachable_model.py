#!/usr/bin/env python3
"""
Train a new model with optimized DirectML acceleration and improved reachable target sampling.
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

# Set environment variables for DirectML and optimization
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
# Add additional GPU optimization settings
os.environ["DIRECTML_ENABLE_ADVANCED_OPTIMIZATION"] = "1"
os.environ["DIRECTML_MEMORY_LIMIT_MB"] = "8192"  # Reserve 8GB GPU memory 
os.environ["TORCH_CUDNN_BENCHMARK"] = "1"  # Enable cuDNN benchmark mode
# Add settings to minimize CPU fallbacks
os.environ["TORCH_AUTOCAST_ENABLED"] = "1"  # Enable automatic mixed precision
os.environ["DIRECTML_FALLBACK_WARNING_LEVEL"] = "0"  # Suppress fallback warnings
os.environ["DIRECTML_ENABLE_MEMORY_OPTIMIZATIONS"] = "1"  # Enable memory optimizations
os.environ["DIRECTML_ENABLE_BATCH_OPTIMIZATION"] = "1"  # Enable batch optimizations
os.environ["TORCH_COMPILE_OFFLOAD"] = "1"  # Try to offload more operations to GPU
# Add settings for larger models and improved memory management
os.environ["DIRECTML_ENABLE_TILED_RESOURCES"] = "1"  # Enable tiled memory resources
os.environ["DIRECTML_ENABLE_DEFERRED_RELEASES"] = "1"  # Defer memory releases
os.environ["DIRECTML_WORKGROUP_SIZE"] = "256"  # Optimize thread workgroup size
os.environ["TORCH_ALLOW_TF32"] = "1"  # Allow TF32 precision for better performance

# Suppress warnings for known CPU fallbacks
import warnings
warnings.filterwarnings("ignore", message=".*not currently supported on the DML backend.*")
warnings.filterwarnings("ignore", message=".*will fall back to run on the CPU.*")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FANUC robot model with DirectML acceleration and better target sampling"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=10000,
        help="Number of timesteps to train (default: 10000)"
    )
    parser.add_argument(
        "--n-envs", 
        type=int, 
        default=1,  # Changed from 8 to 1 as default
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--no-directml", 
        action="store_true",
        help="Disable DirectML acceleration"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models/ppo_reachable_optimized",
        help="Directory to save model"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model after training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,  # Best balance between memory usage and performance
        help="Batch size for training (default: 1024)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout (default: 2048)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create model directory
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n=== Training with Optimized DirectML and Reachable Target Sampling ===")
    print(f"Model directory: {args.model_dir}")
    print(f"Training for {args.timesteps} timesteps with {args.n_envs} parallel environments")
    print(f"DirectML acceleration: {'Disabled' if args.no_directml else 'Enabled'}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Steps per rollout: {args.n_steps}")
    
    # Import here to allow environment variables to be set first
    try:
        from src.core.training.train import train_model
        
        # Special environment settings for better target sampling
        env_kwargs = {
            'training_mode': True,
            'max_episode_steps': 150,
            'verbose': args.verbose,
            'use_curriculum': True,  # Enable curriculum learning
            'viz_speed': 0.0  # Disable visualization by default for faster training
        }
        
        # GPU-optimized policy settings - OPTIMAL CONFIGURATION
        policy_kwargs = {
            'net_arch': dict(pi=[512, 384, 256], vf=[512, 384, 256]),  # Optimal balance for GPU utilization
            'activation_fn': torch.nn.ReLU,  # Explicitly use ReLU for better GPU performance
            'features_extractor_kwargs': {'features_dim': 384}  # Optimal feature dimension
        }
        
        # Start training
        print("\nStarting training...")
        start_time = time.time()
        
        model = train_model(
            model_dir=args.model_dir,
            num_timesteps=args.timesteps,
            use_directml=not args.no_directml,
            n_envs=args.n_envs,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            save_freq=1000,
            eval_freq=2000 if args.evaluate else 0,
            eval_episodes=5 if args.evaluate else 0
        )
        
        # Calculate training statistics
        total_time = time.time() - start_time
        fps = args.timesteps / total_time
        
        print("\n=== Training Complete ===")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Training throughput: {fps:.2f} steps/second")
        print(f"Final model saved to: {os.path.join(args.model_dir, 'final_model')}")
        
        # Run evaluation if requested
        if args.evaluate:
            print("\n=== Evaluating Model ===")
            
            from fanuc_platform import evaluate_model
            
            eval_args = argparse.Namespace(
                model_path=os.path.join(args.model_dir, "final_model"),
                num_episodes=10,
                verbose=args.verbose,
                seed=42,
                render=False
            )
            
            evaluate_model(eval_args)
            
        return 0
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 