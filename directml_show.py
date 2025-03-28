#!/usr/bin/env python3
# directml_show.py
# Specialized script to load and demonstrate a DirectML-trained model with the specific architecture

import os
import sys
import time
import argparse
import torch
import numpy as np

# Import PyBullet globally for helper functions
try:
    import pybullet
except ImportError:
    print("WARNING: PyBullet not found. Visualization enhancements will be limited.")
    pybullet = None

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstrate a DirectML-trained robot model")
    parser.add_argument("model", type=str, help="Path to the trained model file")
    parser.add_argument("--viz-speed", type=float, default=0.02, help="Visualization speed")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to run")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    return parser.parse_args()

def main():
    """
    Main function to demonstrate a specific DirectML-trained model.
    """
    # Parse arguments
    args = parse_args()
    model_path = args.model
    
    # Print banner
    print("\n" + "="*80)
    print("FANUC Robot - DirectML Model Demonstration (Specialized)")
    print("="*80 + "\n")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".pt"):
            model_path += ".pt"
            print(f"Using model file with .pt extension: {model_path}")
        else:
            print(f"ERROR: Model file not found at {model_path}")
            sys.exit(1)
    
    # Configure GUI
    render = not args.no_gui
    if args.no_gui:
        os.environ["NO_GUI"] = "1"
        print("Running in headless mode (no GUI)")
    else:
        print(f"Running with GUI visualization (speed: {args.viz_speed})")
    
    # Add the project directory to the Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Try to load DirectML
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
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"WARNING: Could not initialize DirectML: {e}")
        print("Falling back to CPU")
        dml_device = torch.device("cpu")
    
    # Set up environment variables
    os.environ["PYTORCH_DIRECTML_VERBOSE"] = "1" 
    os.environ["DIRECTML_ENABLE_OPTIMIZATION"] = "1"
    os.environ["USE_DIRECTML"] = "1"
    os.environ["USE_GPU"] = "1"
    
    # Import specific modules with modified sys.argv to avoid argument conflicts
    print("Importing required modules...")
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    try:
        # Import required modules
        from res.rml.python.train_robot_rl_positioning import get_shared_pybullet_client
        from res.rml.python.train_robot_rl_positioning_revamped import RobotPositioningRevampedEnv
        
        # Get shared pybullet client
        print("Initializing PyBullet...")
        p_client = get_shared_pybullet_client(render=render)
        
        # Improve visualization settings
        if render and pybullet is not None:
            print("Configuring enhanced visualization...")
            try:
                # Set better camera position for viewing
                pybullet.resetDebugVisualizerCamera(
                    cameraDistance=1.2,
                    cameraYaw=30,
                    cameraPitch=-20,
                    cameraTargetPosition=[0, 0, 0.5]
                )
                # Enable shadows for better depth perception
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
                # Set better rendering options
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            except Exception as e:
                print(f"Warning: Could not configure visualization: {e}")
        
        print("Creating environment...")
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
        
        # Define a custom model class for this specific architecture
        # Based on the analysis of the model file
        class CustomDirectMLModel:
            def __init__(self, observation_space, action_space, device):
                self.device = device
                self.observation_space = observation_space
                self.action_space = action_space
                
                # Initialize networks
                self._init_networks()
                
            def _init_networks(self):
                import torch.nn as nn
                
                # Feature extractor
                feature_dim = 256
                self.feature_extractor = nn.Sequential(
                    nn.Linear(self.observation_space.shape[0], 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, feature_dim)
                )
                
                # Shared trunk
                self.shared_trunk = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                
                # Action output 
                self.action_mean = nn.Linear(64, self.action_space.shape[0])
                self.action_log_std = nn.Linear(64, self.action_space.shape[0])
                
                # Value head
                self.value_head = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
                # Constants
                self.action_scale = torch.tensor(1.0).to(self.device)  # Will be replaced from state dict
                self.log_2pi = torch.log(torch.tensor(2.0 * np.pi)).to(self.device)
                self.sqrt_2 = torch.sqrt(torch.tensor(2.0)).to(self.device)
                self.eps = torch.tensor(1e-8).to(self.device)
                
                # Move to device
                self.to(self.device)
                
            def to(self, device):
                self.device = device
                self.feature_extractor = self.feature_extractor.to(device)
                self.shared_trunk = self.shared_trunk.to(device)
                self.action_mean = self.action_mean.to(device)
                self.action_log_std = self.action_log_std.to(device)
                self.value_head = self.value_head.to(device)
                return self
                
            def load(self, path):
                print(f"Loading model from {path}...")
                # Use 'cpu' string instead of device object to avoid comparison issues
                checkpoint = torch.load(path, map_location='cpu')
                
                # Load policy parameters
                if 'policy_state_dict' in checkpoint:
                    state_dict = checkpoint['policy_state_dict']
                    
                    # Try to load with some flexibility for different key names
                    try:
                        print("Found policy state dict with keys:", list(state_dict.keys()))
                        
                        # Direct mapping of parameters by name
                        # Feature extractor - Linear layers only
                        layer_map = {
                            # Feature extractor
                            'feature_extractor.0.weight': self.feature_extractor[0].weight,
                            'feature_extractor.0.bias': self.feature_extractor[0].bias,
                            'feature_extractor.2.weight': self.feature_extractor[2].weight,
                            'feature_extractor.2.bias': self.feature_extractor[2].bias,
                            'feature_extractor.4.weight': self.feature_extractor[4].weight,
                            'feature_extractor.4.bias': self.feature_extractor[4].bias,
                            
                            # Shared trunk
                            'shared_trunk.0.weight': self.shared_trunk[0].weight,
                            'shared_trunk.0.bias': self.shared_trunk[0].bias,
                            'shared_trunk.2.weight': self.shared_trunk[2].weight,
                            'shared_trunk.2.bias': self.shared_trunk[2].bias,
                            
                            # Action networks
                            'action_mean.weight': self.action_mean.weight,
                            'action_mean.bias': self.action_mean.bias,
                            'action_log_std.weight': self.action_log_std.weight,
                            'action_log_std.bias': self.action_log_std.bias,
                            
                            # Value head
                            'value_head.0.weight': self.value_head[0].weight,
                            'value_head.0.bias': self.value_head[0].bias,
                            'value_head.2.weight': self.value_head[2].weight,
                            'value_head.2.bias': self.value_head[2].bias,
                        }
                        
                        # Try loading each parameter
                        loaded_params = 0
                        for saved_name, target_param in layer_map.items():
                            # Try different naming variations (with or without feature_extractor prefix)
                            if saved_name in state_dict:
                                target_param.data.copy_(state_dict[saved_name])
                                loaded_params += 1
                            else:
                                # Try alternate naming schemes
                                alt_name = saved_name.replace('feature_extractor.', 'features.')
                                if alt_name in state_dict:
                                    target_param.data.copy_(state_dict[alt_name])
                                    loaded_params += 1
                        
                        print(f"Successfully loaded {loaded_params} parameters")
                        
                        # Move the model to the target device
                        self.to(self.device)
                        print("Model loaded successfully!")
                        
                    except Exception as e:
                        import traceback
                        print(f"Error loading model parameters: {e}")
                        print(traceback.format_exc())
                        # Continue with default parameters
                        print("Continuing with default parameters")
                else:
                    print("ERROR: No policy state dict found in checkpoint")
                    
            def predict(self, observation, deterministic=True):
                """Get the action for a given observation"""
                with torch.no_grad():
                    # Convert to tensor
                    if isinstance(observation, np.ndarray):
                        observation = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
                    
                    # Forward pass
                    features = self.feature_extractor(observation)
                    shared_features = self.shared_trunk(features)
                    
                    # Get action mean and std
                    action_mean = self.action_mean(shared_features)
                    
                    if deterministic:
                        action = action_mean
                    else:
                        action_log_std = self.action_log_std(shared_features)
                        action_std = torch.exp(action_log_std)
                        normal = torch.distributions.Normal(action_mean, action_std)
                        action = normal.sample()
                    
                    # Convert to numpy
                    action_np = action.cpu().numpy().flatten()
                    return action_np, None
        
        # Load the model
        print(f"\nLoading model from {model_path}...")
        model = CustomDirectMLModel(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=dml_device
        )
        model.load(model_path)
        
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
            state, info = env.reset()
            
            # Store target position for visualization
            target_position = info.get('target_position') if 'target_position' in info else None
            if target_position is None and hasattr(env, '_target_position'):
                target_position = env._target_position
                
            # Simple target visualization
            if render and pybullet is not None and target_position is not None:
                try:
                    # Simple target visualization using debug markers
                    pybullet.addUserDebugText(
                        "TARGET",
                        target_position,
                        textColorRGB=[1, 0, 0],
                        textSize=1.5
                    )
                    # Add cross to mark target
                    size = 0.05
                    pybullet.addUserDebugLine(
                        [target_position[0] - size, target_position[1], target_position[2]],
                        [target_position[0] + size, target_position[1], target_position[2]],
                        lineColorRGB=[1, 0, 0],
                        lineWidth=3
                    )
                    pybullet.addUserDebugLine(
                        [target_position[0], target_position[1] - size, target_position[2]],
                        [target_position[0], target_position[1] + size, target_position[2]],
                        lineColorRGB=[1, 0, 0],
                        lineWidth=3
                    )
                    pybullet.addUserDebugLine(
                        [target_position[0], target_position[1], target_position[2] - size],
                        [target_position[0], target_position[1], target_position[2] + size],
                        lineColorRGB=[1, 0, 0],
                        lineWidth=3
                    )
                except Exception as e:
                    print(f"Error visualizing target: {e}")
            
            done = False
            total_reward = 0
            steps = 0
            
            # Run episode
            while not done:
                # Get action from model
                action, _ = model.predict(state, deterministic=True)
                
                # Take step in environment
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Add delay for visualization
                if render and args.viz_speed > 0 and pybullet is not None:
                    # Display current distance
                    if steps % 25 == 0:
                        try:
                            # Add new distance display (or update it)
                            distance = info.get('distance', None)
                            if distance is not None:
                                pybullet.addUserDebugText(
                                    f"Distance: {distance:.2f} cm",
                                    [0, 0, 1.2],
                                    textColorRGB=[1, 1, 0],
                                    textSize=1.5,
                                    lifeTime=1.0
                                )
                        except Exception as e:
                            pass
                    
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
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
    finally:
        # Restore original arguments
        sys.argv = original_argv
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 