"""
Script to run curriculum training with optimal settings
"""
import os
import subprocess
import platform
import sys

def main():
    print("Starting curriculum training with optimal settings...")
    
    # Check if we're on Windows
    if platform.system() == "Windows":
        # Run with CPU for MLP policies (recommended based on performance testing)
        cmd = [sys.executable, "curriculum_training.py", "--cpu"]
    else:
        # On Linux, try GPU first
        cmd = [sys.executable, "curriculum_training.py"]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training script
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        
    print("\nTraining completed or interrupted.")
    print("Trained models are saved in the ./models directory.")
    print("You can visualize training progress using TensorBoard:")
    print("tensorboard --logdir=./logs")

if __name__ == "__main__":
    main() 