#!/usr/bin/env python3
# Simple script to test loading and evaluating a DirectML model

import os
import sys
import time
import numpy as np
import pybullet as p
import torch

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

def test_directml_model(model_path, gui=True, viz_speed=0.02, episodes=1):
    print("\n" + "="*50)
    print("DirectML Model Test")
    print("="*50)
    
    # Try importing DirectML
    try:
        import torch_directml
        print("DirectML available, initializing device...")
        device_count = torch_directml.device_count()
        if device_count == 0:
            raise RuntimeError("No DirectML devices detected")
            
        print(f"DirectML devices available: {device_count}")
        dml_device = torch_directml.device()
        print(f"Using DirectML device: {dml_device}")
        
        # Create a test tensor to verify DirectML works
        test_tensor = torch.ones((2, 3), device=dml_device)
        _ = test_tensor.cpu().numpy()
        print("✓ DirectML test tensor created successfully")
    except Exception as e:
        print(f"WARNING: Could not initialize DirectML: {e}")
        print("Falling back to CPU")
        dml_device = "cpu"
    
    # Now try to load the model
    try:
        print(f"\nLoading model from {model_path}...")
        # First check if the model file exists
        if not os.path.exists(model_path):
            if os.path.exists(model_path + ".pt"):
                model_path = model_path + ".pt"
            else:
                print(f"ERROR: Model file not found at {model_path}")
                return
                
        # Now try to setup a simple environment
        print("\nCreating a simple environment to test observations...")
        from src.envs.robot_sim import FANUCRobotEnv
        from src.core.train_robot_rl_positioning_revamped import RobotPositioningRevampedEnv, CustomDirectMLModel
        
        # Create a simplified environment
        env = RobotPositioningRevampedEnv(
            gui=gui,
            viz_speed=viz_speed,
            verbose=True,
            rank=0,
            offset_x=0.0,
            training_mode=False
        )
        
        # Reset the environment and get an observation
        obs, _ = env.reset()
        print(f"\nObservation shape: {obs.shape}")
        
        # Create the DirectML model
        print("\nCreating and loading DirectML model...")
        model = CustomDirectMLModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=dml_device
        )
        
        # Load the model
        print(f"Loading weights from {model_path}...")
        model.load(model_path)
        print("Model loaded successfully!")
        
        # Now run some test episodes
        print(f"\nRunning {episodes} test episodes...")
        
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
                if gui and viz_speed > 0:
                    time.sleep(viz_speed)
            
            # Track results
            distances.append(info.get('distance', float('inf')))
            rewards.append(total_reward)
            episode_lengths.append(steps)
            if info.get('success', False):
                success_count += 1
            
            print(f"  Episode finished after {steps} steps")
            print(f"  Distance to target: {info.get('distance', 'unknown'):.2f} cm")
            print(f"  Success: {'Yes' if info.get('success', False) else 'No'}")
            print(f"  Total reward: {total_reward:.2f}")
        
        # Print overall results
        print("\n" + "="*50)
        print("DirectML Model Evaluation Results:")
        print(f"  Success rate: {success_count/episodes:.1%} ({success_count}/{episodes})")
        if distances:
            print(f"  Average distance: {np.mean(distances):.2f} cm")
            print(f"  Best distance: {min(distances):.2f} cm")
        if rewards:
            print(f"  Average reward: {np.mean(rewards):.2f}")
        if episode_lengths:
            print(f"  Average steps: {np.mean(episode_lengths):.1f}")
        print("="*50)
        
    except Exception as e:
        import traceback
        print(f"\nERROR testing model: {e}")
        print(traceback.format_exc())
        
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test a DirectML model")
    parser.add_argument("--model", type=str, default="../../models/ppo_directml_20250326_202801/final_model", help="Path to model file")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to run")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI visualization")
    parser.add_argument("--viz-speed", type=float, default=0.02, help="Visualization speed")
    
    args = parser.parse_args()
    
    test_directml_model(
        model_path=args.model,
        gui=not args.no_gui,
        viz_speed=args.viz_speed,
        episodes=args.episodes
    ) 