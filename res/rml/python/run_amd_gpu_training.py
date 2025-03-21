#!/usr/bin/env python3
# run_amd_gpu_training.py
# Simple launcher for DirectML-accelerated robot training on AMD GPUs

import os
import sys
import argparse
import platform
import shutil
import subprocess
import time
import psutil

# Set required environment variables for AMD GPU with DirectML
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256'
os.environ['DML_PREFETCH_BUFFERS'] = '1'

# Set thread count based on system (use half of logical cores)
cpu_count = psutil.cpu_count(logical=True)
thread_count = max(4, cpu_count // 2)
os.environ['DML_THREAD_COUNT'] = str(thread_count)

def check_directml_available():
    """Check if torch-directml is available."""
    try:
        import torch_directml
        return True
    except ImportError:
        return False

def get_system_info():
    """Get system information."""
    import platform
    import torch
    
    system_info = {
        'os': f"{platform.system()} {platform.release()}",
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'cpu_count': psutil.cpu_count(logical=True),
        'ram_total': round(psutil.virtual_memory().total / (1024**3), 1),
        'ram_available': round(psutil.virtual_memory().available / (1024**3), 1),
    }
    
    # Check for DirectML
    has_directml = check_directml_available()
    system_info['directml'] = has_directml
    
    if has_directml:
        import torch_directml
        # Try to get AMD GPU info
        try:
            dml_device = torch_directml.device()
            system_info['gpu'] = "AMD GPU detected (DirectML backend)"
        except Exception as e:
            system_info['gpu'] = f"DirectML available but error initializing device: {e}"
    else:
        system_info['gpu'] = "DirectML not available - install with: pip install torch-directml"
    
    return system_info

def print_system_info():
    """Print system information in a nice format."""
    system_info = get_system_info()
    
    print("=" * 60)
    print("Robot Training with DirectML for AMD GPU")
    print("=" * 60)
    print(f"System: {system_info['os']}")
    print(f"Python: {system_info['python']}")
    print(f"PyTorch: {system_info['pytorch']}")
    print(f"CPU Threads: {system_info['cpu_count']}")
    print(f"Memory: {system_info['ram_total']} GB total, {system_info['ram_available']} GB available")
    print(f"DirectML: {'Available' if system_info['directml'] else 'Not Available'}")
    print(f"GPU: {system_info['gpu']}")
    print("=" * 60)

def install_directml():
    """Install torch-directml package."""
    print("Installing torch-directml...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-directml"])
    print("torch-directml installed successfully!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run robot training with AMD GPU acceleration')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true', help='Train a new model (default)')
    mode_group.add_argument('--demo', action='store_true', help='Run demo with existing model')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate existing model')
    
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps (default: 10000)')
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel environments (default: 8)')
    parser.add_argument('--model', type=str, help='Path to model (required for --demo and --eval)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--viz-speed', type=float, default=0.05, help='Visualization speed (default: 0.05)')
    parser.add_argument('--save-video', action='store_true', help='Save video of demo/evaluation')
    parser.add_argument('--install-directml', action='store_true', help='Install torch-directml if not available')
    
    args = parser.parse_args()
    
    # Set default mode to train if none specified
    if not (args.train or args.demo or args.eval):
        args.train = True
    
    # Check required arguments
    if (args.demo or args.eval) and not args.model:
        parser.error("--model is required when using --demo or --eval")
    
    return args

def main():
    """Main function."""
    args = parse_args()
    
    # Check if DirectML is available
    if not check_directml_available():
        if args.install_directml:
            install_directml()
        else:
            print("ERROR: torch-directml is not installed.")
            print("Install it with: pip install torch-directml")
            print("Or run this script with --install-directml flag")
            return 1
    
    # Print system information
    print_system_info()
    
    # Construct command to run the appropriate script
    cmd = [sys.executable, "train_robot_rl_demo_directml.py"]
    
    if args.train:
        print(f"Starting training for {args.steps} steps with {args.parallel} parallel environments...")
        cmd.extend(["--steps", str(args.steps), "--parallel", str(args.parallel)])
    elif args.demo:
        print(f"Starting demo with model: {args.model}")
        cmd.extend(["--demo", "--load", args.model, "--viz-speed", str(args.viz_speed)])
    elif args.eval:
        print(f"Starting evaluation of model: {args.model} with {args.episodes} episodes")
        cmd.extend(["--eval-only", "--load", args.model, "--eval-episodes", str(args.episodes)])
    
    if args.save_video:
        cmd.append("--save-video")
    
    # Run the command
    start_time = time.time()
    try:
        subprocess.check_call(cmd)
        end_time = time.time()
        print(f"Operation completed successfully in {end_time - start_time:.1f} seconds.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main()) 