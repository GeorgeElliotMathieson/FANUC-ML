#!/usr/bin/env python3
# simple_directml_demo.py
# Simple demonstration script for DirectML-trained models

import os
import sys
import time
import argparse
import numpy as np
import torch

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import our utilities
from directml_utils import (
    block_argparse, 
    restore_argparse, 
    setup_directml, 
    patch_directml_ppo, 
    ensure_model_file_exists,
    safe_import_robot_env,
    initialize_pybullet
)

def main():
    """
    Main function to demonstrate a DirectML-trained model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate a DirectML-trained robot model")
    parser.add_argument("model", nargs="?", help="Path to the trained model file")
    parser.add_argument("--model", dest="model_flag", help="Path to the trained model file (alternative)")
    parser.add_argument("--viz-speed", type=float, default=0.02, help="Visualization speed")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    
    args = parser.parse_args()
    
    # Use either positional or --model flag
    model_path = args.model if args.model else args.model_flag
    
    # Ensure we have a model path
    if not model_path:
        print("ERROR: No model path specified. Use --model or provide it as a positional argument.")
        sys.exit(1)
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Demonstration")
    print("Showcasing trained model behavior")
    print("="*80 + "\n")
    
    # Check if model file exists and adjust path if needed
    model_path = ensure_model_file_exists(model_path)
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Setup DirectML
    dml_device = setup_directml()
    if dml_device is None:
        sys.exit(1)
    
    print("="*80 + "\n")
    
    # Configure GUI
    render = not args.no_gui
    if args.no_gui:
        os.environ["NO_GUI"] = "1"
        print("Running in headless mode (no GUI)")
    else:
        print(f"Running with GUI visualization (speed: {args.viz_speed})")
    
    # Initialize PyBullet (sets up shared client)
    initialize_pybullet(render=render)
    
    # Import required modules safely
    print("Importing required modules...")
    RobotEnv, pybullet = safe_import_robot_env(render=render, viz_speed=args.viz_speed)
    if RobotEnv is None:
        sys.exit(1)
    
    # Block argument parsing in imported modules
    original_argv = block_argparse()
    
    try:
        # Import DirectML modules
        print("Importing DirectML modules...")
        import train_robot_rl_ppo_directml
        from train_robot_rl_ppo_directml import DirectMLPPO
        
        # Restore original arguments
        restore_argparse(original_argv)
        
        # Patch the DirectMLPPO class with the predict method
        patch_directml_ppo()
        
        # Create the environment manually to avoid arg parsing
        print("\nCreating robot environment...")
        env = RobotEnv(
            gui=render,
            gui_delay=0.0,
            workspace_size=0.7,
            clean_viz=True,
            viz_speed=args.viz_speed,
            verbose=True,
            parallel_viz=False,
            rank=0,
            offset_x=0.0,
            training_mode=False
        )
        
        # Create model
        print(f"\nLoading model from {model_path}...")
        try:
            # Create a DirectML model
            model = DirectMLPPO(
                env=env,
                device=dml_device,
                verbose=True
            )
            
            # Load the saved model
            model.load(model_path)
            print("Model loaded successfully!")
            
            # Run demonstrations
            episodes = args.episodes
            print(f"\nRunning {episodes} demonstration episodes...")
            
            # Track metrics
            success_count = 0
            distances = []
            rewards = []
            episode_lengths = []
            
            for episode in range(episodes):
                print(f"\nEpisode {episode+1}/{episodes}")
                
                # Reset environment
                state, _ = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                # Run episode
                while not done:
                    # Get action from model
                    action, _ = model.predict(state)
                    
                    # Take step in environment
                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                    
                    # Add delay for visualization
                    if render and args.viz_speed > 0:
                        time.sleep(args.viz_speed)
                
                # Track results
                distances.append(info.get('distance', float('inf')))
                rewards.append(total_reward)
                episode_lengths.append(steps)
                if info.get('success', False):
                    success_count += 1
                
                print(f"  Episode finished after {steps} steps")
                print(f"  Distance to target: {info.get('distance', 'unknown')} cm")
                print(f"  Success: {'Yes' if info.get('success', False) else 'No'}")
                print(f"  Total reward: {total_reward:.2f}")
            
            # Print overall results
            print("\n" + "="*50)
            print("Demonstration Results:")
            print(f"  Success rate: {success_count/episodes:.1%} ({success_count}/{episodes})")
            if distances:
                print(f"  Average distance: {np.mean(distances):.2f} cm")
                print(f"  Best distance: {min(distances):.2f} cm")
            print(f"  Average steps: {np.mean(episode_lengths):.1f}")
            print(f"  Average reward: {np.mean(rewards):.2f}")
            print("="*50)
            
            # Disconnect PyBullet
            pybullet.disconnect()
            
        except Exception as e:
            import traceback
            print(f"Error loading or running model: {e}")
            print(traceback.format_exc())
            sys.exit(1)
            
    except ImportError as e:
        print(f"Failed to import DirectML modules: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 