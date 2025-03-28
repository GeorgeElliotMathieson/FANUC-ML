#!/usr/bin/env python3
# run_eval.py - Isolated script for evaluating DirectML models 
# This script avoids argument parsing conflicts by temporarily clearing sys.argv

import os
import sys
import time

def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"FANUC Robot - {title}")
    print("="*80 + "\n")

def main():
    # Save the original arguments
    original_args = sys.argv.copy()
    
    # Check for help request
    if len(original_args) < 2 or "--help" in original_args or "-h" in original_args:
        print_usage()
        return 0
    
    # Extract arguments before clearing sys.argv
    model_path = original_args[1]
    episodes = 5  # Default
    
    # Check if second arg is episodes number
    if len(original_args) > 2 and original_args[2].isdigit():
        episodes = int(original_args[2])
    
    # Parse options
    no_gui = "--no-gui" in original_args
    verbose = "--verbose" in original_args
    
    # Check for speed option
    viz_speed = 0.02  # Default
    for arg in original_args:
        if arg.startswith("--speed="):
            try:
                viz_speed = float(arg.split("=")[1])
            except:
                pass
    
    # Print banner
    print_banner("DirectML Model Evaluation (Isolated Mode)")
    
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
    
    # CRITICAL: Modify sys.argv to avoid argument parsing conflicts
    sys.argv = [original_args[0]]
    
    # Now we can safely import modules
    try:
        import torch
        import numpy as np
        
        # Import only the specific functions needed
        from res.rml.python.train_robot_rl_positioning_revamped import (
            create_revamped_envs, 
            CustomDirectMLModel, 
            generate_directml_visualizations
        )
        
        # Check for DirectML support
        try:
            import torch_directml
            print("DirectML support detected")
            
            # Create DirectML device
            dml_device = torch_directml.device()
            print(f"Using DirectML device: {dml_device}")
            
            # Verify DirectML works with a test tensor
            test_tensor = torch.ones((2, 3), device=dml_device)
            _ = test_tensor.cpu().numpy()
            print("✓ DirectML acceleration active")
        except ImportError:
            print("DirectML not available, using CPU")
            dml_device = torch.device("cpu")
        except Exception as e:
            print(f"Error initializing DirectML: {e}")
            print("Falling back to CPU")
            dml_device = torch.device("cpu")
        
        # Create evaluation environment
        print("\nCreating environment...")
        envs = create_revamped_envs(
            num_envs=1,
            viz_speed=viz_speed if not no_gui else 0.0,
            parallel_viz=False,
            training_mode=False
        )
        env = envs[0]
        
        # Determine if this is a DirectML model
        is_directml = "directml" in model_path.lower()
        
        # Load the model
        print(f"\nLoading model from {model_path}...")
        
        if is_directml:
            # Load DirectML model
            model = CustomDirectMLModel(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=dml_device
            )
            
            # Load the model
            model.load(model_path)
            print("DirectML model loaded successfully")
        else:
            # Load standard model
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print("Standard model loaded successfully")
        
        # Run evaluation episodes
        print(f"\nRunning {episodes} evaluation episodes...")
        
        # Track metrics
        total_rewards = []
        episode_lengths = []
        success_count = 0
        distances = []
        
        # For visualization data
        episode_data = []
        
        # Run all episodes
        for i in range(episodes):
            print(f"\nEpisode {i+1}/{episodes}:")
            
            # Reset environment
            obs, _ = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            
            # Track episode data
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_infos = []
            
            # Run episode
            while not done and step_count < 150:  # Max 150 steps per episode
                # Store observation
                ep_obs.append(obs)
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Store action
                ep_actions.append(action)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update counters
                total_reward += reward
                step_count += 1
                
                # Store step data
                ep_rewards.append(reward)
                ep_infos.append(info)
                
                # Add visualization delay if needed
                if viz_speed > 0 and not no_gui:
                    time.sleep(viz_speed)
            
            # Final observation
            ep_obs.append(obs)
            
            # Track episode results
            total_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # Check for success
            is_success = info.get('success', False)
            if is_success:
                success_count += 1
                
            # Track distance
            final_distance = info.get('distance', float('inf'))
            distances.append(final_distance)
            
            # Store episode data for visualization
            episode_data.append({
                'obs': ep_obs,
                'actions': ep_actions,
                'rewards': ep_rewards,
                'infos': ep_infos,
                'total_reward': total_reward,
                'success': is_success,
                'distance': final_distance,
                'length': step_count
            })
            
            # Print episode results
            print(f"  Result: {'Success' if is_success else 'Failure'}")
            print(f"  Distance: {final_distance:.2f} cm")
            print(f"  Steps: {step_count}")
            print(f"  Reward: {total_reward:.2f}")
        
        # Calculate overall statistics
        success_rate = success_count / episodes
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        avg_steps = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
        avg_distance = sum(distances) / len(distances) if distances else float('inf')
        best_distance = min(distances) if distances else float('inf')
        
        # Print summary
        print("\n" + "="*50)
        print("Evaluation Results:")
        print(f"  Success rate: {success_rate:.1%} ({success_count}/{episodes})")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average distance: {avg_distance:.2f} cm")
        print(f"  Best distance: {best_distance:.2f} cm")
        print("="*50)
        
        # Generate visualizations for DirectML models
        if is_directml and len(episode_data) > 0:
            try:
                print("\nGenerating visualizations...")
                viz_dir = generate_directml_visualizations(episode_data)
                if viz_dir:
                    print(f"Visualizations saved to: {viz_dir}")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
        # Close environment
        env.close()
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return 1
    
    finally:
        # Restore original arguments
        sys.argv = original_args

def print_usage():
    """Print usage information"""
    print("Usage: python run_eval.py <model_path> [episodes] [options]")
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
    print("  python run_eval.py ./models/ppo_directml_20250326_202801/final_model 3 --verbose")

if __name__ == "__main__":
    sys.exit(main()) 