#!/usr/bin/env python3
# demo_directml_model.py
# Demonstration script for DirectML-trained models

import os
import sys
import time
import argparse
import torch
import numpy as np

# Add predict method to DirectMLPPO for compatibility
def add_predict_method(model_class):
    """
    Add predict method to the model class to make it compatible with the demo script
    """
    def predict(self, observation, deterministic=True):
        """
        Predict action for a given observation
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
        Returns:
            action: Action to take
            state: None (unused, for compatibility)
        """
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        
        # Get action from the model
        with torch.no_grad():
            action, _, _ = self.policy_network.get_action(observation, deterministic=deterministic)
        
        # Convert to numpy and return
        action_np = action.cpu().numpy().flatten()
        return action_np, None
    
    # Add method to class
    setattr(model_class, 'predict', predict)

def main():
    """
    Main function to demonstrate a DirectML-trained model.
    """
    # Parse arguments directly at the beginning to avoid module import conflicts
    parser = argparse.ArgumentParser(description="Demonstrate DirectML-trained robot models")
    parser.add_argument("--model", type=str, help="Path to the trained model file")
    parser.add_argument("--viz-speed", type=float, default=0.02, help="Visualization speed (delay in seconds)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to demonstrate")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    
    # Parse arguments with handling for the single positional model path format
    args, unknown = parser.parse_known_args()
    
    # Check if there are unknown args and the first one might be the model path
    if unknown and not args.model:
        args.model = unknown[0]
        print(f"Using positional model path: {args.model}")
    
    # Ensure we have a model path
    if not args.model:
        print("ERROR: No model path specified. Use --model or provide it as a positional argument.")
        sys.exit(1)
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Demonstration")
    print("Showcasing trained model behavior")
    print("="*80 + "\n")
    
    # Add the project directory to the Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Check if model file exists and adjust path if needed
    model_path = args.model
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path += ".pt"
            print(f"Using model file with .pt extension: {model_path}")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            sys.exit(1)
    
    # Setup DirectML
    try:
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
        
        # Configure GUI
        if args.no_gui:
            print("Running in headless mode (no GUI)")
            os.environ["NO_GUI"] = "1"
            render = False
        else:
            print(f"Running with GUI visualization (speed: {args.viz_speed})")
            render = True
            
        # Now import required modules after setting environment variables
        from res.rml.python.train_robot_rl_positioning import (
            get_shared_pybullet_client,
            load_workspace_data,
            determine_reachable_workspace
        )
        from res.rml.python.train_robot_rl_positioning_revamped import (
            RobotPositioningRevampedEnv,
            CustomFeatureExtractor,
            JointLimitedBox
        )
        
        # Import DirectML-specific modules
        import train_robot_rl_ppo_directml
        from train_robot_rl_ppo_directml import DirectMLPPO
        
        # Add predict method to DirectMLPPO class for compatibility
        add_predict_method(DirectMLPPO)
        
        # Create environment
        print("\nCreating robot environment...")
        
        # Get a shared PyBullet client
        client_id = get_shared_pybullet_client(render=render)
        
        # Create the environment
        env = RobotPositioningRevampedEnv(
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
            print(f"\nRunning {args.episodes} demonstration episodes...")
            
            # Track metrics
            success_count = 0
            distances = []
            rewards = []
            episode_lengths = []
            
            for episode in range(args.episodes):
                print(f"\nEpisode {episode+1}/{args.episodes}")
                
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
            print(f"  Success rate: {success_count/args.episodes:.1%} ({success_count}/{args.episodes})")
            print(f"  Average distance: {np.mean(distances):.2f} cm")
            print(f"  Average steps: {np.mean(episode_lengths):.1f}")
            print(f"  Average reward: {np.mean(rewards):.2f}")
            print(f"  Best distance: {min(distances):.2f} cm")
            print("="*50)
            
        except Exception as e:
            import traceback
            print(f"Error loading or running model: {e}")
            print(traceback.format_exc())
            sys.exit(1)
            
    except ImportError as e:
        print(f"ERROR: DirectML package not found: {e}")
        print("AMD GPU acceleration will not be available.")
        print("To install DirectML support, run: pip install torch-directml")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR initializing DirectML: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 