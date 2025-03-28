#!/usr/bin/env python3
# evaluate_directml_model.py - Dedicated script for evaluating DirectML models
# This version avoids argument parsing conflicts with the main training script

import os
import sys
import torch
import numpy as np
from datetime import datetime

def main():
    """
    Main function for evaluating DirectML models with enhanced visualizations.
    This script bypasses the standard argument parser to avoid conflicts.
    """
    # Handle arguments manually to avoid conflicts
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        return 0
    
    # Get model path (required argument)
    model_path = sys.argv[1]
    
    # Parse other arguments
    episodes = 5  # Default number of episodes
    no_gui = False
    verbose = False
    viz_speed = 0.02  # Default visualization speed
    
    # Process remaining arguments
    for arg in sys.argv[2:]:
        if arg.isdigit():
            episodes = int(arg)
        elif arg == "--no-gui":
            no_gui = True
        elif arg == "--verbose":
            verbose = True
        elif arg.startswith("--speed="):
            try:
                viz_speed = float(arg.split("=")[1])
            except:
                pass
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Evaluation")
    print("Enhanced visualization and flexible architecture support")
    print("="*80 + "\n")
    
    # Print configuration
    print(f"Model path: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"GUI enabled: {not no_gui}")
    print(f"Visualization speed: {viz_speed}")
    print(f"Verbose mode: {verbose}")
    print("")
    
    # Verify model path
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path = model_path + ".pt"
            print(f"Found model at {model_path}")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            return 1
    
    # Set up environment variables
    if no_gui:
        os.environ["NO_GUI"] = "1"
    
    # Add the project root to the Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Import only the specific functions needed, avoiding argument parsing code
    try:
        # Import DirectML support
        try:
            import torch_directml
            dml_available = True
            
            # Try to create DirectML device
            try:
                dml_device = torch_directml.device()
                print(f"Using DirectML device: {dml_device}")
                
                # Create test tensor to verify DirectML works
                test_tensor = torch.ones((2, 3), device=dml_device)
                _ = test_tensor.cpu().numpy()  # Force execution
                print("✓ DirectML acceleration active")
            except Exception as e:
                print(f"Warning: Could not initialize DirectML device: {e}")
                print("Falling back to CPU")
                dml_device = torch.device("cpu")
        except ImportError:
            print("DirectML not available, using CPU")
            dml_available = False
            dml_device = torch.device("cpu")
        
        # Import environment creation function directly to avoid arg parsing
        from res.rml.python.train_robot_rl_positioning_revamped import (
            create_revamped_envs, 
            CustomDirectMLModel
        )
        
        # Create the environment
        print("\nCreating environment...")
        envs = create_revamped_envs(
            num_envs=1,
            viz_speed=viz_speed,
            parallel_viz=False,
            training_mode=False
        )
        env = envs[0]  # Get the first environment
        
        # Check if this is a DirectML model
        is_directml_model = "directml" in model_path.lower()
        
        # Load the model
        print(f"Loading model from {model_path}...")
        
        if is_directml_model:
            # Create and load DirectML model
            model = CustomDirectMLModel(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cpu"  # Start with CPU, we'll move to GPU later if available
            )
            
            # Load the model parameters
            success = model.load(model_path)
            
            if not success:
                print("ERROR: Failed to load DirectML model")
                return 1
                
            # Move to DirectML device if available
            if dml_available:
                model.to(dml_device)
                print("Model moved to DirectML device")
        else:
            # Load standard model
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print("Standard model loaded successfully")
        
        # Run evaluation manually
        print(f"\nRunning {episodes} evaluation episodes...")
        
        # Track metrics
        total_rewards = []
        episode_lengths = []
        success_count = 0
        all_distances = []
        
        # For visualization data
        episode_data = []
        
        # Run episodes
        for ep in range(episodes):
            print(f"\nEpisode {ep+1}/{episodes}")
            
            # Reset the environment
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            # Store episode data
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_infos = []
            
            # Run until episode completion
            while not done and step_count < 150:  # 150 steps max per episode
                # Store observation
                ep_obs.append(obs)
                
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Store action
                ep_actions.append(action)
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store reward and info
                ep_rewards.append(reward)
                ep_infos.append(info)
                
                # Update counters
                total_reward += reward
                step_count += 1
                
                # Slow down visualization if needed
                if viz_speed > 0 and not no_gui:
                    import time
                    time.sleep(viz_speed)
            
            # Track metrics
            total_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # Track success
            if 'success' in info and info['success']:
                success_count += 1
                
            # Track distance
            if 'distance' in info:
                all_distances.append(info['distance'])
                
            # Store episode data for visualization
            episode_data.append({
                'obs': ep_obs,
                'actions': ep_actions,
                'rewards': ep_rewards,
                'infos': ep_infos,
                'total_reward': total_reward,
                'success': info.get('success', False),
                'distance': info.get('distance', float('inf')),
                'length': step_count
            })
            
            # Print episode results
            print(f"  Result: {'Success' if info.get('success', False) else 'Failure'}")
            print(f"  Distance: {info.get('distance', float('inf')):.2f} cm")
            print(f"  Steps: {step_count}")
            print(f"  Reward: {total_reward:.2f}")
        
        # Calculate final metrics
        success_rate = success_count / episodes if episodes > 0 else 0
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        avg_steps = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
        avg_distance = sum(all_distances) / len(all_distances) if all_distances else float('inf')
        best_distance = min(all_distances) if all_distances else float('inf')
        
        # Print summary
        print("\n" + "="*50)
        print("Evaluation Results:")
        print(f"  Success rate: {success_rate:.1%} ({success_count}/{episodes})")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average distance: {avg_distance:.2f} cm")
        print(f"  Best distance: {best_distance:.2f} cm")
        print("="*50)
        
        # Generate visualizations if it's a DirectML model
        if is_directml_model and episode_data:
            try:
                # Try to generate visualizations
                print("\nGenerating visualizations...")
                
                # Import the visualization function
                from res.rml.python.train_robot_rl_positioning_revamped import generate_directml_visualizations
                
                # Generate visualizations
                viz_dir = generate_directml_visualizations(episode_data)
                
                if viz_dir:
                    print(f"Visualizations saved to: {viz_dir}")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
        # Close the environment
        env.close()
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return 1

def print_usage():
    """Print usage information"""
    print("Usage: python evaluate_directml_model.py <model_path> [episodes] [options]")
    print("")
    print("Arguments:")
    print("  model_path     Path to the model file (required)")
    print("  episodes       Number of episodes to evaluate (default: 5)")
    print("")
    print("Options:")
    print("  --no-gui       Disable visualization")
    print("  --verbose      Show detailed output")
    print("  --speed=X      Set visualization speed in seconds (default: 0.02)")
    print("")
    print("Example:")
    print("  python evaluate_directml_model.py ./models/ppo_directml_20250326_202801/final_model 3 --verbose")

if __name__ == "__main__":
    sys.exit(main()) 