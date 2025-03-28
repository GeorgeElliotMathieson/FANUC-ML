#!/usr/bin/env python3
# run_directml_eval.py - Simple script to evaluate DirectML models

import os
import sys
import time
import torch
import numpy as np

def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"FANUC Robot - {title}")
    print("="*80 + "\n")

def print_usage():
    """Print usage information"""
    print("Usage: python run_directml_eval.py <model_path> [episodes] [options]")
    print("  model_path: Path to the DirectML model to evaluate")
    print("  episodes: Number of episodes to evaluate (default: 5)")
    print("  options: Additional options")
    print("    --no-gui: Run without visualization")
    print("    --verbose: Show detailed output")
    print("    --speed=X: Set visualization speed in seconds (default: 0.02)")
    print("")
    print("Example: python run_directml_eval.py ./models/ppo_directml_20250326_202801/final_model 3 --verbose")
    
def main():
    # Check for minimum arguments
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    # Parse arguments
    model_path = sys.argv[1]
    episodes = 5  # Default
    
    # Check if second arg is episodes number
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        episodes = int(sys.argv[2])
    
    # Parse options
    no_gui = "--no-gui" in sys.argv
    verbose = "--verbose" in sys.argv
    
    # Check for speed option
    viz_speed = 0.02  # Default
    for arg in sys.argv:
        if arg.startswith("--speed="):
            try:
                viz_speed = float(arg.split("=")[1])
            except:
                pass
    
    # Print banner
    print_banner("DirectML Model Evaluation")
    
    # Print configuration
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"GUI: {'Disabled' if no_gui else 'Enabled'}")
    print(f"Visualization speed: {viz_speed}")
    print(f"Verbose: {'Yes' if verbose else 'No'}")
    print("")
    
    # Ensure model file exists
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path += ".pt"
            print(f"Using model file with .pt extension: {model_path}")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            return 1
    
    # Configure environment variables
    if no_gui:
        os.environ["NO_GUI"] = "1"
    
    # Import necessary modules directly
    try:
        from res.rml.python.train_robot_rl_positioning_revamped import evaluate_model_wrapper
        
        # Run the evaluation
        print("Starting evaluation...")
        result = evaluate_model_wrapper(
            model_path=model_path,
            num_episodes=episodes,
            visualize=not no_gui,
            verbose=verbose
        )
        
        if result:
            print("\n" + "="*50)
            print("Evaluation completed successfully!")
            print(f"Success rate: {result['success_rate']:.1%}")
            print(f"Average reward: {result['avg_reward']:.2f}")
            print(f"Average distance: {result['avg_distance']:.2f} cm")
            print(f"Best distance: {result['best_distance']:.2f} cm")
            print("="*50)
            return 0
        else:
            print("\nEvaluation returned no results")
            return 1
            
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 