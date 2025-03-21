#!/usr/bin/env python3
# run_optimized_directml_ppo.py - Script to run training with optimized DirectML PPO implementation

import os
import sys
import argparse
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run optimized DirectML PPO training for robot positioning')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=300000, help='Total number of training steps')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--load', type=str, help='Path to pre-trained model to continue training')
    
    # DirectML parameters
    parser.add_argument('--memory-limit', type=int, default=512, help='Memory limit for DirectML in MB')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for PPO')
    
    # Visualization
    parser.add_argument('--viz-speed', type=float, default=0.0, help='Visualization speed (0 to disable)')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI visualization')
    
    # Demo and evaluation
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    
    # Misc options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Detect if we're running in demo or eval mode
    demo_mode = args.demo
    eval_mode = args.eval
    
    # Convert these arguments to the format expected by the DirectML script
    cmd_args = [
        f"--steps {args.steps}",
        f"--learning-rate {args.learning_rate}",
        f"--parallel {args.parallel}",
        f"--memory-limit {args.memory_limit}",
    ]
    
    # Add optional arguments
    if args.load:
        cmd_args.append(f"--load {args.load}")
    
    if args.viz_speed > 0:
        cmd_args.append(f"--viz-speed {args.viz_speed}")
    
    if args.no_gui:
        cmd_args.append("--no-gui")
    
    if args.verbose:
        cmd_args.append("--verbose")
    
    if args.seed is not None:
        cmd_args.append(f"--seed {args.seed}")
    
    # Always add optimized DirectML flag
    cmd_args.append("--optimized-directml")
    
    # Add mode-specific flags
    if demo_mode:
        cmd_args.append("--demo")
    elif eval_mode:
        cmd_args.append("--eval-only")
        cmd_args.append(f"--eval-episodes {args.eval_episodes}")
    
    # Construct the final command
    cmd = f"python train_robot_rl_demo_directml.py {' '.join(cmd_args)}"
    
    # Print the command
    print(f"Executing: {cmd}")
    
    # Execute the command
    start_time = time.time()
    return_code = os.system(cmd)
    elapsed_time = time.time() - start_time
    
    # Print execution information
    print(f"\nCommand completed with return code: {return_code}")
    print(f"Execution time: {elapsed_time:.1f} seconds")
    
    if return_code != 0:
        print("ERROR: Command execution failed")
        sys.exit(return_code)
    
    # Print summary for training
    if not demo_mode and not eval_mode:
        print("\n=== Training Summary ===")
        print(f"Trained for {args.steps} steps with optimized DirectML PPO")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Steps per second: {args.steps / elapsed_time:.1f}")
        
        # Calculate approximate speedup compared to CPU
        estimated_cpu_speed = args.steps / 200  # Rough estimate: 200 steps/sec on CPU
        estimated_cpu_time = args.steps / estimated_cpu_speed
        speedup = estimated_cpu_time / elapsed_time
        print(f"Estimated speedup vs CPU: {speedup:.1f}x")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 