#!/usr/bin/env python3
# evaluate_directml.py - Enhanced evaluation script for DirectML models

import os
import sys
import argparse
import torch
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a DirectML-trained robot model with enhanced visualizations")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI visualization")
    parser.add_argument("--speed", type=float, default=0.02, help="Visualization speed (seconds between frames)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-video", action="store_true", help="Save video of the evaluation")
    return parser.parse_args()

def setup_directml():
    """Set up DirectML environment"""
    try:
        import torch_directml
        print("\n" + "="*80)
        print("FANUC Robot - DirectML Model Evaluation")
        print("="*80 + "\n")
        
        # Check for available DirectML devices
        device_count = torch_directml.device_count()
        if device_count == 0:
            print("WARNING: No DirectML devices detected, falling back to CPU")
            return torch.device("cpu")
        
        # Create a DirectML device
        dml_device = torch_directml.device()
        print(f"DirectML devices available: {device_count}")
        print(f"Using DirectML device: {dml_device}")
        
        # Create a test tensor on the DirectML device to verify it works
        try:
            test_tensor = torch.ones((2, 3), device=dml_device)
            # Access the tensor to force execution on GPU
            _ = test_tensor.cpu().numpy()
            print("✓ DirectML acceleration active and verified")
            print("✓ Test tensor created successfully on GPU")
            print("="*80 + "\n")
            return dml_device
        except Exception as e:
            print(f"WARNING: Could not create tensor on DirectML device: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")
            
    except ImportError:
        print("WARNING: torch_directml not found, falling back to CPU")
        print("To enable DirectML acceleration, install: pip install torch-directml")
        return torch.device("cpu")

def main():
    # Parse arguments
    args = parse_args()
    
    # Ensure the model path is valid
    model_path = args.model
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path += ".pt"
        else:
            print(f"ERROR: Model file not found at {model_path}")
            sys.exit(1)
    
    # Set up DirectML
    device = setup_directml()
    
    # Configure environment variables
    if args.no_gui:
        os.environ["NO_GUI"] = "1"
        print("Running in headless mode (no GUI)")
    else:
        print(f"Running with GUI visualization (speed: {args.speed})")
    
    # Add project directory to Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Now import our modules
    from res.rml.python.train_robot_rl_positioning_revamped import (
        evaluate_model_wrapper,
        CustomDirectMLModel
    )
    
    # Run the evaluation
    print(f"Evaluating model: {model_path}")
    print(f"Running {args.episodes} episodes")
    
    result = evaluate_model_wrapper(
        model_path=model_path,
        num_episodes=args.episodes,
        visualize=not args.no_gui,
        verbose=args.verbose
    )
    
    if result:
        # Print success message
        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"Average reward: {result['avg_reward']:.2f}")
        print(f"Average distance: {result['avg_distance']:.2f} cm")
        print(f"Best distance: {result['best_distance']:.2f} cm")
        print("="*80)
    else:
        print("\nEvaluation failed or returned no results")
    
if __name__ == "__main__":
    main() 