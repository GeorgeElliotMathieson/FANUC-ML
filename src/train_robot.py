#!/usr/bin/env python3
# train_robot.py
# Unified training entry point for FANUC robot end-effector positioning

import os
import sys
import argparse
import torch

def is_directml_model(model_path):
    """
    Check if a model was trained with DirectML by looking at the file name pattern.
    """
    if model_path and 'directml' in model_path.lower():
        return True
    return False

def ensure_model_file_exists(model_path):
    """
    Ensure the model file exists and return the correct path.
    """
    if not model_path:
        return None
        
    # Check if the file exists as is
    if os.path.exists(model_path):
        return model_path
        
    # Try with .pt extension
    if os.path.exists(model_path + ".pt"):
        model_path += ".pt"
        return model_path
    
    # Try standard model directory
    model_dir = os.path.join("models", model_path)
    if os.path.exists(model_dir):
        return model_dir
    
    # Try with timestamp directories
    model_files = []
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pt") or file.endswith(".zip"):
                if model_path in os.path.join(root, file):
                    model_files.append(os.path.join(root, file))
    
    if model_files:
        # Return the most recent match
        print(f"Found {len(model_files)} possible model files matching {model_path}")
        print(f"Using: {model_files[0]}")
        return model_files[0]
    
    print(f"Warning: Could not find model file at {model_path}")
    return model_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FANUC Robot Training and Control Platform"
    )
    
    # Mode selection arguments
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument("--eval-only", action="store_true", help="Evaluate an existing model")
    mode_group.add_argument("--demo", action="store_true", help="Run a demo of an existing model")
    
    # Model loading/saving arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--load", type=str, help="Path to load an existing model from")
    model_group.add_argument("--save", type=str, default="./models/fanuc-ml", 
                         help="Path to save the trained model to")
    
    # Training parameters
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--steps", type=int, default=1000000, 
                          help="Number of training timesteps")
    train_group.add_argument("--eval-episodes", type=int, default=10,
                          help="Number of episodes to evaluate on")
    
    # Environment options
    env_group = parser.add_argument_group("Environment Options")
    env_group.add_argument("--no-gui", action="store_true", help="Disable GUI")
    env_group.add_argument("--viz-speed", type=float, default=0.02,
                        help="Visualization speed (smaller is faster)")
    
    # Miscellaneous options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--verbose", action="store_true", help="Enable verbose output")
    misc_group.add_argument("--seed", type=int, help="Random seed")
    misc_group.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    misc_group.add_argument("--use-directml", action="store_true", help="Use DirectML for AMD GPU acceleration")
    
    return parser.parse_args()

def main():
    """
    Main function to run using stable-baselines3 implementation.
    This will call the appropriate function based on command-line arguments.
    """
    # Parse command line arguments
    args = parse_args()
    
    if args.seed is not None:
        # Import and set seed
        from src.utils.seed import set_seed
        set_seed(args.seed)
        
    # Detect if we should use CUDA
    if args.use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device for training: {torch.cuda.get_device_name(0)}")
    else:
        print("Using standard CPU implementation")
        
        from src.core.train_robot_rl_positioning_revamped import main as train_main
        train_main()

def run_directml_demo(model_path, viz_speed=0.02, no_gui=False):
    """
    Run a DirectML-specific demo by calling the DirectML model directly.
    """
    try:
        print("\n" + "="*80)
        print("DirectML Model Demonstration")
        print("Running specialized demo for DirectML-trained models")
        print("="*80 + "\n")
        
        import torch
        import torch_directml
        import numpy as np
        import time
        
        # Check for available DirectML devices
        device_count = torch_directml.device_count()
        if device_count == 0:
            raise RuntimeError("No DirectML devices detected")
        
        # Create a DirectML device
        dml_device = torch_directml.device()
        print(f"DirectML devices available: {device_count}")
        print(f"Using DirectML device: {dml_device}")
        
        # Configure GUI
        if no_gui:
            print("Running in headless mode (no GUI)")
            os.environ["NO_GUI"] = "1"
            render = False
        else:
            print(f"Running with GUI visualization (speed: {viz_speed})")
            render = True
        
        # Import the DirectML implementation
        from src.directml.train_robot_rl_ppo_directml import DirectMLPPO
        
        # Add predict method to DirectML models if needed
        if not hasattr(DirectMLPPO, 'predict'):
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
            setattr(DirectMLPPO, 'predict', predict)
        
        # Import from standard modules
        from src.envs.robot_sim import FANUCRobotEnv
        from src.core.train_robot_rl_positioning_revamped import (
            RobotPositioningRevampedEnv,
            CustomFeatureExtractor,
            JointLimitedBox
        )
        
        # Create environment
        print("\nCreating robot environment...")
        
        # Create the environment
        env = RobotPositioningRevampedEnv(
            gui=render,
            gui_delay=0.0,
            workspace_size=0.7,
            clean_viz=True,
            viz_speed=viz_speed,
            verbose=True,
            parallel_viz=False,
            rank=0,
            offset_x=0.0,
            training_mode=False
        )
        
        # Create model
        print(f"\nLoading DirectML model from {model_path}...")
        
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
        episodes = 5
        print(f"\nRunning {episodes} demonstration episodes with DirectML...")
        
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
                if render and viz_speed > 0:
                    time.sleep(viz_speed)
            
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
        print("DirectML Demonstration Results:")
        print(f"  Success rate: {success_count/episodes:.1%} ({success_count}/{episodes})")
        print(f"  Average distance: {np.mean(distances):.2f} cm")
        print(f"  Average steps: {np.mean(episode_lengths):.1f}")
        print(f"  Average reward: {np.mean(rewards):.2f}")
        print(f"  Best distance: {min(distances):.2f} cm")
        print("="*50)
        
        return 0
    
    except Exception as e:
        import traceback
        print(f"Error in DirectML demo: {e}")
        print(traceback.format_exc())
        return 1

def train_robot_main(args):
    """
    Main function for training the robot with standard (non-DirectML) implementation.
    This is a wrapper to provide a consistent interface from main.py.
    
    Args:
        args: Command line arguments parsed by main.py
        
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Process model path if specified
        if hasattr(args, 'load') and args.load:
            args.load = ensure_model_file_exists(args.load)
        
        # Convert args to a format the existing functions can understand
        sys_argv = ["train_robot.py"]
        
        # Handle flags based on mode
        if args.train:
            # Training mode settings
            if hasattr(args, 'load') and args.load:
                sys_argv.extend(["--load", args.load])
            if hasattr(args, 'steps'):
                sys_argv.extend(["--steps", str(args.steps)])
        elif args.eval:
            # Evaluation mode settings
            sys_argv.append("--eval-only")
            if hasattr(args, 'load') and args.load:
                sys_argv.extend(["--load", args.load])
            if hasattr(args, 'eval_episodes'):
                sys_argv.extend(["--eval-episodes", str(args.eval_episodes)])
        elif args.demo:
            # Demo mode settings
            sys_argv.append("--demo")
            if hasattr(args, 'load') and args.load:
                sys_argv.extend(["--load", args.load])
        
        # Process common arguments
        if args.no_gui:
            sys_argv.append("--no-gui")
        if hasattr(args, 'viz_speed') and args.viz_speed > 0:
            sys_argv.extend(["--viz-speed", str(args.viz_speed)])
        if args.verbose:
            sys_argv.append("--verbose")
        if hasattr(args, 'seed') and args.seed is not None:
            sys_argv.extend(["--seed", str(args.seed)])
        
        # Check for DirectML-specific handling
        if hasattr(args, 'directml') and args.directml:
            sys_argv.append("--use-directml")
            
            # For demo mode with DirectML, use the specialized function
            if args.demo and hasattr(args, 'load') and args.load:
                return run_directml_demo(
                    args.load, 
                    args.viz_speed if hasattr(args, 'viz_speed') else 0.02,
                    args.no_gui
                )
        
        # Save the original argv and replace with our constructed one
        original_argv = sys.argv
        sys.argv = sys_argv
        
        # Call the existing main function
        try:
            main()
            return 0
        finally:
            # Restore the original argv
            sys.argv = original_argv
            
    except Exception as e:
        import traceback
        print(f"Error in train_robot_main: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main() 