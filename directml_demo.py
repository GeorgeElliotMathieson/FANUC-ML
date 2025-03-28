#!/usr/bin/env python3
# directml_demo.py - Demo script for DirectML models

import os
import sys
import argparse
import numpy as np
import torch

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import our utilities first
from directml_utils import (
    setup_directml, 
    patch_directml_ppo, 
    load_directml_model, 
    ensure_model_file_exists,
    safe_import_robot_env,
    initialize_pybullet
)

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstrate a DirectML-trained robot model")
    parser.add_argument("--model", required=True, help="Path to the DirectML model")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI visualization")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--speed", type=float, default=0.02, help="Visualization speed")
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Demo")
    print("="*80)
    
    # Ensure model file exists
    model_path = ensure_model_file_exists(args.model)
    print(f"Model file: {model_path}")
    
    # Setup DirectML
    dml_device = setup_directml()
    if dml_device is None:
        sys.exit(1)
    
    # Configure GUI
    render = not args.no_gui
    if args.no_gui:
        os.environ["NO_GUI"] = "1"
        print("Running in headless mode (no GUI)")
    else:
        print(f"Running with GUI visualization (speed: {args.speed})")
    
    # Initialize PyBullet (sets up shared client)
    initialize_pybullet(render=render)
    
    # Import robot environment safely
    print("Importing environment...")
    RobotEnv, pybullet = safe_import_robot_env(render=render, viz_speed=args.speed)
    if RobotEnv is None:
        sys.exit(1)
    
    # Create environment
    env = RobotEnv(
        gui=render,
        gui_delay=0.0,
        workspace_size=0.7,
        clean_viz=True,
        viz_speed=args.speed,
        verbose=True,
        parallel_viz=False,
        rank=0,
        offset_x=0.0,
        training_mode=False
    )
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_directml_model(model_path, env, dml_device)
    
    if model is None:
        print("Failed to load model")
        sys.exit(1)
    
    # Run demonstrations
    print(f"\nRunning {args.episodes} demonstration episodes...")
    success_count = 0
    distances = []
    rewards = []
    steps_list = []
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode+1}/{args.episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        # Run episode
        while not done:
            # Get action
            action, _ = model.predict(obs)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Optional delay for visualization
            if render and args.speed > 0:
                import time
                time.sleep(args.speed)
        
        # Record results
        success = info.get('success', False)
        distance = info.get('distance', float('inf'))
        
        if success:
            success_count += 1
        
        distances.append(distance)
        rewards.append(total_reward)
        steps_list.append(steps)
        
        print(f"  Steps: {steps}")
        print(f"  Distance: {distance:.2f} cm")
        print(f"  Success: {'Yes' if success else 'No'}")
        print(f"  Reward: {total_reward:.2f}")
    
    # Print summary
    print("\n" + "="*50)
    print("Demonstration Results:")
    print(f"  Success rate: {success_count/args.episodes:.1%} ({success_count}/{args.episodes})")
    if distances:
        print(f"  Average distance: {np.mean(distances):.2f} cm")
        print(f"  Best distance: {min(distances):.2f} cm")
    print(f"  Average steps: {np.mean(steps_list):.1f}")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print("="*50)
    
    # Disconnect
    pybullet.disconnect()

if __name__ == "__main__":
    main() 