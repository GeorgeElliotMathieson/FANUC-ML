#!/usr/bin/env python3
# train_robot.py
# Unified training entry point for FANUC robot end-effector positioning

import os
import sys

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
    if not model_path.endswith('.pt'):
        with_ext = model_path + '.pt'
        if os.path.exists(with_ext):
            return with_ext
    
    # Try without extension
    if model_path.endswith('.pt'):
        without_ext = model_path[:-3]
        if os.path.exists(without_ext):
            return without_ext
            
    # Original path will be tried anyway
    return model_path

def main():
    """
    Main function for the unified robot training approach.
    This is a wrapper that calls the actual training implementation.
    """
    try:
        # Print banner
        print("\n" + "="*80)
        print("FANUC Robot Training - Unified Training Platform")
        print("Using standardized training approach with improved architecture")
        print("="*80 + "\n")
        
        # Add the project directory to the Python path
        project_dir = os.path.dirname(os.path.abspath(__file__))
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        
        # Process arguments
        args = sys.argv[1:]
        
        # Find model path if specified
        model_path = None
        for i, arg in enumerate(args):
            if arg == "--load" and i + 1 < len(args):
                model_path = args[i + 1]
                model_path = ensure_model_file_exists(model_path)
                # Update args with the corrected path
                args[i + 1] = model_path
                break
        
        # Extract common parameters
        viz_speed = 0.0
        for i, arg in enumerate(args):
            if arg == "--viz-speed" and i + 1 < len(args):
                try:
                    viz_speed = float(args[i + 1])
                except ValueError:
                    pass
        
        # Check for AMD GPU acceleration request
        use_amd_gpu = False
        if "--use-directml" in args or "--use-gpu" in args or "--use-amd" in args or "--directml-demo" in args:
            use_amd_gpu = True
            # Remove these custom flags
            args = [arg for arg in args if arg not in ["--use-directml", "--use-gpu", "--use-amd"]]
        
        # If loading a model, check if it's a DirectML model
        if model_path and is_directml_model(model_path) and not use_amd_gpu:
            print("Detected DirectML-trained model. Enabling DirectML for compatibility.")
            use_amd_gpu = True
        
        # Check for DirectML demo mode
        directml_demo = False
        if "--directml-demo" in args:
            directml_demo = True
            # Remove this flag
            args = [arg for arg in args if arg != "--directml-demo"]
            
            # Make sure we have the model path and it's a DirectML model
            if not model_path:
                print("ERROR: --directml-demo requires a model path specified with --load")
                sys.exit(1)
            
            if not is_directml_model(model_path):
                print("WARNING: Model doesn't appear to be a DirectML model. This may not work.")
        
        # Add AMD GPU message to banner if using it
        if use_amd_gpu:
            print("✓ AMD GPU acceleration enabled through DirectML")
            print("✓ Optimized for RX 6700S and similar AMD GPUs")
            print("="*80 + "\n")
        
        # Process demo mode arguments
        demo_mode = False
        if "--demo" in args:
            # Flag that we're in demo mode
            demo_mode = True
            # Remove --demo from arguments
            args = [arg for arg in args if arg != "--demo"]
            
            # Add evaluation flags
            if "--eval-only" not in args:
                args.append("--eval-only")
            
            # Set visualization speed if not specified
            if not any(arg.startswith("--viz-speed") for arg in args):
                args.extend(["--viz-speed", "0.02"])
            
            # Set number of evaluation episodes
            args.extend(["--eval-episodes", "5"])
        
        # If DirectML demo mode is enabled, run a specialized demo
        if directml_demo:
            return run_directml_demo(model_path, viz_speed, "--no-gui" in args)
        
        # Update sys.argv for the imported script to use
        sys.argv = [sys.argv[0]] + args
        
        # Attempt to setup DirectML if AMD GPU acceleration was requested
        if use_amd_gpu:
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
                
                # Import the DirectML implementation without affecting global state
                print("Loading DirectML-optimized implementation for AMD GPU...")
                import train_robot_rl_ppo_directml
                
                # Create a directml-compatible args object
                class DirectMLArgs:
                    def __init__(self):
                        self.gpu = True  # This is what DirectML uses to check for GPU
                        self.steps = 1000000
                        self.save_freq = 50000
                        self.parallel = 1
                        self.seed = None
                        self.load = None
                        self.eval_only = False
                        self.viz_speed = 0.0
                        self.verbose = False
                        self.learning_rate = 3e-4
                        self.n_steps = 2048
                        self.batch_size = 256  # Optimized for AMD GPU
                        self.n_epochs = 4  # Reduced for AMD GPU optimization
                        self.gamma = 0.99
                        self.gae_lambda = 0.95
                        self.clip_range = 0.2
                        self.ent_coef = 0.0
                        self.vf_coef = 0.5
                        self.max_grad_norm = 0.5
                        # Add demo-related attributes
                        self.eval_episodes = 5
                        # Attributes for GUI
                        self.gui = True
                
                # Create and populate the args object
                directml_args = DirectMLArgs()
                
                # Parse command line arguments to override defaults
                for i in range(len(args)):
                    if i > 0 and args[i-1] in ["--steps", "--parallel", "--seed", "--load", 
                                               "--viz-speed", "--learning-rate", "--eval-episodes"]:
                        # Skip values that have already been processed with their flags
                        continue
                        
                    # Handle arguments with values
                    if args[i] == "--steps" and i + 1 < len(args):
                        directml_args.steps = int(args[i + 1])
                    elif args[i] == "--parallel" and i + 1 < len(args):
                        directml_args.parallel = int(args[i + 1])
                    elif args[i] == "--seed" and i + 1 < len(args):
                        directml_args.seed = int(args[i + 1])
                    elif args[i] == "--load" and i + 1 < len(args):
                        directml_args.load = args[i + 1]
                    elif args[i] == "--viz-speed" and i + 1 < len(args):
                        directml_args.viz_speed = float(args[i + 1])
                    elif args[i] == "--learning-rate" and i + 1 < len(args):
                        directml_args.learning_rate = float(args[i + 1])
                    elif args[i] == "--eval-episodes" and i + 1 < len(args):
                        directml_args.eval_episodes = int(args[i + 1])
                    # Handle flag arguments
                    elif args[i] == "--eval-only":
                        directml_args.eval_only = True
                    elif args[i] == "--verbose":
                        directml_args.verbose = True
                
                # For GUI, DirectML uses different system
                if "--no-gui" in args:
                    os.environ["NO_GUI"] = "1"
                    directml_args.gui = False
                
                print("Starting training with DirectML-optimized implementation...")
                
                # Call the training function from DirectML module with our args
                train_robot_rl_ppo_directml.train_robot_with_ppo_directml(directml_args)
                return
                
            except ImportError as e:
                print(f"ERROR: DirectML package not found: {e}")
                print("AMD GPU acceleration will not be available.")
                print("To install DirectML support, run: pip install torch-directml")
            except Exception as e:
                print(f"ERROR initializing DirectML: {e}")
                print(f"Exception details: {str(e)}")
                print("Falling back to CPU implementation.")
        
        # If DirectML failed or wasn't requested, use the standard implementation
        print("Using standard CPU implementation")
        
        from res.rml.python import train_robot_rl_positioning_revamped
        train_robot_rl_positioning_revamped.main()
        
    except Exception as e:
        import traceback
        print(f"Error in train_robot.py: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

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
        import train_robot_rl_ppo_directml
        from train_robot_rl_ppo_directml import DirectMLPPO
        
        # Add predict method to DirectMLPPO for compatibility
        def add_predict_method(model_class):
            """
            Add predict method to the model class to make it compatible with the demo
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
        
        # Add predict method to DirectMLPPO class
        add_predict_method(DirectMLPPO)
        
        # Import from standard modules
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

if __name__ == "__main__":
    main() 