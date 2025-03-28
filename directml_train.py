#!/usr/bin/env python3
# directml_train.py
# Direct script to train a robot model with DirectML acceleration

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a robot model with DirectML acceleration")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel robots")
    parser.add_argument("--load", type=str, help="Load a pre-trained model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model")
    parser.add_argument("--viz-speed", type=float, default=0.0, help="Visualization speed")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    return parser.parse_args()

def main():
    """
    Main function to train a robot model with DirectML acceleration.
    """
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot Training with DirectML Acceleration")
    print("Optimized for AMD RX 6700S GPU")
    print("="*80 + "\n")
    
    # Add the project directory to the Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Setup DirectML
    try:
        import torch
        import torch_directml
        
        # Check for available DirectML devices
        device_count = torch_directml.device_count()
        if device_count == 0:
            raise RuntimeError("No DirectML devices detected")
        
        # Create a DirectML device
        dml_device = torch_directml.device()
        print(f"DirectML devices available: {device_count}")
        print(f"Using DirectML device: {dml_device}")
        
        # Create a test tensor on the DirectML device to verify it works
        test_tensor = torch.ones((2, 3), device=dml_device)
        # Access the tensor to force execution on GPU
        _ = test_tensor.cpu().numpy()
        
        print("✓ DirectML acceleration active and verified")
        print("✓ Test tensor created successfully on GPU")
        print("="*80 + "\n")
        
        # Configure environment variables for DirectML
        os.environ["PYTORCH_DIRECTML_VERBOSE"] = "1"
        os.environ["DIRECTML_ENABLE_OPTIMIZATION"] = "1"
        os.environ["USE_DIRECTML"] = "1"
        os.environ["USE_GPU"] = "1"
        
        # Configure GUI
        if args.no_gui:
            os.environ["NO_GUI"] = "1"
            print("Running in headless mode (no GUI)")
        else:
            print(f"Running with GUI visualization (speed: {args.viz_speed})")
        
        # Import the DirectML implementation
        import train_robot_rl_ppo_directml
        
        # Create and populate the args object
        class DirectMLArgs:
            def __init__(self):
                self.gpu = True  # This is what DirectML uses to check for GPU
                self.steps = args.steps
                self.save_freq = 50000
                self.parallel = args.parallel
                self.seed = None
                self.load = args.load
                self.eval_only = args.eval_only
                self.viz_speed = args.viz_speed
                self.verbose = args.verbose
                self.learning_rate = 3e-4
                self.n_steps = 2048
                self.batch_size = args.batch_size
                self.n_epochs = 4  # Reduced for AMD GPU optimization
                self.gamma = 0.99
                self.gae_lambda = 0.95
                self.clip_range = 0.2
                self.ent_coef = 0.0
                self.vf_coef = 0.5
                self.max_grad_norm = 0.5
                # Demo-related attributes
                self.eval_episodes = 5
                # GUI attribute
                self.gui = not args.no_gui
        
        # Create the args object
        directml_args = DirectMLArgs()
        
        # Print training details
        print(f"Training with:")
        print(f"  Steps: {directml_args.steps}")
        print(f"  Parallel environments: {directml_args.parallel}")
        print(f"  Batch size: {directml_args.batch_size}")
        print(f"  GUI: {'Enabled' if directml_args.gui else 'Disabled'}")
        if directml_args.load:
            print(f"  Loading model from: {directml_args.load}")
        if directml_args.eval_only:
            print(f"  Mode: Evaluation only")
        else:
            print(f"  Mode: Training")
        print("\n" + "="*80 + "\n")
        
        # Call the training function
        print("Starting training with DirectML acceleration...")
        train_robot_rl_ppo_directml.train_robot_with_ppo_directml(directml_args)
        
    except ImportError as e:
        print(f"ERROR: DirectML package not found: {e}")
        print("AMD GPU acceleration will not be available.")
        print("To install DirectML support, run: pip install torch-directml")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 