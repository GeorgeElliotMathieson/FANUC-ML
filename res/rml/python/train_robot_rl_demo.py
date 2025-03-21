#!/usr/bin/env python3
# train_robot_rl_demo.py
# Standalone wrapper script for running demonstrations with the revamped robot positioning

import os
import sys
import argparse
import time
import numpy as np
import pybullet as p
import torch
import gymnasium as gym  # Import gymnasium for the environment wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Check for GPU support (CUDA or ROCm/HIP)
def check_gpu_support():
    """Check for GPU support in PyTorch including both CUDA and AMD ROCm/HIP."""
    has_cuda = torch.cuda.is_available()
    
    # Check for AMD ROCm/HIP support (may not be directly detectable in all PyTorch versions)
    has_rocm = False
    try:
        has_rocm = hasattr(torch, 'hip') and torch.hip.is_available()
    except:
        # If the above explicit check fails, try alternative methods
        try:
            # Check for HIP version in torch.version
            has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        except:
            pass
    
    # Return available GPU backends and device info
    if has_cuda:
        return {
            'backend': 'cuda',
            'available': True,
            'devices': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0)
        }
    elif has_rocm:
        return {
            'backend': 'hip',
            'available': True,
            'devices': torch.hip.device_count() if hasattr(torch.hip, 'device_count') else 'Unknown',
            'device_name': 'AMD GPU (ROCm/HIP)'
        }
    else:
        return {
            'backend': 'cpu',
            'available': False
        }

# We need to prevent the imported module from parsing args when imported
# Save sys.argv and reset it temporarily
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # Keep only the script name

# Import functions from the revamped implementation
from train_robot_rl_positioning_revamped import (
    RobotPositioningRevampedEnv,
    load_workspace_data,
    evaluate_model as original_evaluate_model,
    run_evaluation_sequence as original_run_evaluation_sequence,
    train_revamped_robot
)

# Restore original argv
sys.argv = original_argv

# Utility function to strictly enforce joint limits from URDF
def ensure_joint_limits(robot, joint_positions):
    """
    Strictly enforce joint limits from the URDF model.
    
    Args:
        robot: The robot environment instance
        joint_positions: Array of joint positions to check and enforce
        
    Returns:
        Array of joint positions with limits enforced
    """
    limited_positions = joint_positions.copy()
    
    for i, pos in enumerate(joint_positions):
        if i in robot.joint_limits:
            limit_low, limit_high = robot.joint_limits[i]
            # Strictly enforce limits
            if pos < limit_low:
                limited_positions[i] = limit_low
                print(f"WARNING: Joint {i} below limit ({pos:.4f} < {limit_low:.4f}), enforcing limit")
            elif pos > limit_high:
                limited_positions[i] = limit_high
                print(f"WARNING: Joint {i} above limit ({pos:.4f} > {limit_high:.4f}), enforcing limit")
    
    return limited_positions

# Custom environment wrapper to enforce joint limits
class JointLimitEnforcingEnv(gym.Wrapper):
    """
    Environment wrapper that strictly enforces joint limits from the URDF model.
    """
    def __init__(self, env):
        super().__init__(env)
        self.robot = env.robot
        print("Joint limits will be strictly enforced according to URDF specifications")
    
    def step(self, action):
        # Extract the underlying robot action from the environment's step method
        if hasattr(self.env, 'robot') and hasattr(self.env.robot, 'step'):
            # Get current joint positions
            state = self.robot._get_state()
            current_joint_positions = state[:self.robot.dof*2:2]
            
            # For delta joint position control, calculate new positions
            new_joint_positions = []
            for i, delta in enumerate(action):
                # Current position plus delta
                new_pos = current_joint_positions[i] + delta
                new_joint_positions.append(new_pos)
            
            # Enforce joint limits
            limited_positions = ensure_joint_limits(self.robot, new_joint_positions)
            
            # Create zero velocities for the robot step
            zero_velocities = [0.0] * len(limited_positions)
            
            # Call the original step method with enforced limits
            next_state = self.robot.step((limited_positions, zero_velocities))
            
            # Pass to parent step with the original action (the environment will re-apply limits internally)
            return self.env.step(action)
        else:
            # Fallback if the environment doesn't match our expected structure
            return self.env.step(action)

# Create a wrapper function to handle API differences for evaluation
def evaluate_model_wrapper(model_path, num_episodes=20, visualize=True, verbose=True):
    """
    Wrapper for evaluate_model that handles API differences
    in the reset() method between different versions of Gymnasium.
    """
    print(f"\nEvaluating model: {model_path}")
    
    # Load workspace data if not already loaded
    if False: # _WORKSPACE_POSITIONS is None:
        load_workspace_data(verbose=verbose)
    
    # Create environment for evaluation
    env = RobotPositioningRevampedEnv(
        gui=visualize,
        viz_speed=0.01 if visualize else 0.0,
        verbose=verbose
    )
    
    # Wrap in VecEnv as required by Stable-Baselines3
    vec_env = DummyVecEnv([lambda: env])
    
    # Check if VecNormalize statistics are available
    vec_normalize_path = model_path.replace("final_model", "vec_normalize_stats")
    if os.path.exists(vec_normalize_path):
        # Load with normalization
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Don't update normalization statistics during evaluation
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        print("Loaded normalization statistics")
    else:
        print("No normalization statistics found, evaluating without normalization")
    
    # Load the model
    model = PPO.load(model_path, env=vec_env)
    print(f"Model loaded from {model_path}")
    
    # Initialize metrics
    metrics = {
        "success_rate": 0.0,
        "avg_distance": 0.0,
        "avg_steps": 0.0,
        "avg_reward": 0.0,
        "min_distance": float('inf'),
        "max_distance": 0.0,
        "success_episodes": [],
        "failure_episodes": [],
    }
    
    # Run evaluation episodes
    total_reward = 0.0
    total_steps = 0
    total_distance = 0.0
    successes = 0
    
    for ep in range(num_episodes):
        if verbose:
            print(f"\nEvaluation Episode {ep+1}/{num_episodes}")
        
        # Reset environment with API compatibility handling
        try:
            # Try new gymnasium API (returns obs, info)
            obs, info = vec_env.reset()
            initial_distance = info[0].get('initial_distance', 0.0)
        except ValueError:
            # Fall back to older gym API (returns only obs)
            obs = vec_env.reset()
            # Estimate initial distance as best we can
            state = env.robot._get_state()
            ee_position = state[12:15]
            initial_distance = np.linalg.norm(ee_position - env.target_position)
        
        if verbose:
            print(f"Initial distance to target: {initial_distance*100:.2f}cm")
        
        # Initialize episode variables
        done = False
        ep_reward = 0.0
        ep_steps = 0
        best_distance = initial_distance
        
        # Run episode
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Enforce joint limits before stepping (get current joint positions)
            state = env.robot._get_state()
            current_joint_positions = state[:env.robot.dof*2:2]
            
            # Calculate new joint positions from action deltas
            new_joint_positions = []
            for i, delta in enumerate(action[0]):  # action is wrapped in array due to vectorized env
                new_pos = current_joint_positions[i] + delta
                new_joint_positions.append(new_pos)
            
            # Enforce limits
            limited_positions = ensure_joint_limits(env.robot, new_joint_positions)
            
            # If action was limited, create a new action array with deltas that respect limits
            if not np.array_equal(new_joint_positions, limited_positions):
                limited_deltas = []
                for i, (pos, limited_pos) in enumerate(zip(new_joint_positions, limited_positions)):
                    if pos != limited_pos:
                        # Calculate the allowed delta that respects the limit
                        allowed_delta = limited_pos - current_joint_positions[i]
                        limited_deltas.append(allowed_delta)
                    else:
                        limited_deltas.append(action[0][i])
                
                # Replace action with limited version
                action = np.array([limited_deltas])
            
            # Step environment with API compatibility handling
            try:
                # Try new gymnasium API
                obs, reward, terminated, truncated, info = vec_env.step(action)
                done = terminated[0] or truncated[0]
            except ValueError:
                # Fall back to older gym API
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
            
            # Update metrics
            ep_reward += reward[0]
            ep_steps += 1
            
            try:
                current_distance = info[0].get('distance', 0.0)
            except (IndexError, TypeError):
                # Fallback if info is not structured as expected
                current_distance = best_distance
            
            # Update best distance
            best_distance = min(best_distance, current_distance)
            
            # Delay for visualization
            if visualize:
                time.sleep(0.01)
        
        # Calculate success
        try:
            success = info[0].get('target_reached', False)
        except (IndexError, TypeError):
            success = False
        
        # Update metrics
        total_reward += ep_reward
        total_steps += ep_steps
        total_distance += best_distance
        
        if success:
            successes += 1
            metrics["success_episodes"].append(ep)
        else:
            metrics["failure_episodes"].append(ep)
        
        # Update min/max distances
        metrics["min_distance"] = min(metrics["min_distance"], best_distance)
        metrics["max_distance"] = max(metrics["max_distance"], best_distance)
        
        # Print episode results
        if verbose:
            print(f"Episode {ep+1} - {'SUCCESS' if success else 'FAILURE'}")
            print(f"  Best distance: {best_distance*100:.2f}cm")
            print(f"  Steps: {ep_steps}")
            print(f"  Reward: {ep_reward:.2f}")
    
    # Calculate final metrics
    metrics["success_rate"] = successes / num_episodes
    metrics["avg_distance"] = total_distance / num_episodes
    metrics["avg_steps"] = total_steps / num_episodes
    metrics["avg_reward"] = total_reward / num_episodes
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average Distance: {metrics['avg_distance']*100:.2f}cm")
    print(f"Average Steps: {metrics['avg_steps']:.1f}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Best Distance: {metrics['min_distance']*100:.2f}cm")
    
    # Close environment
    vec_env.close()
    
    return metrics

# Create a wrapper function to handle API differences
def run_evaluation_sequence_wrapper(model_path, viz_speed=0.02, save_video=False):
    """
    Wrapper for run_evaluation_sequence that handles API differences
    in the reset() method between different versions of Gymnasium.
    """
    print(f"Running demonstration with model: {model_path}")
    
    # Load workspace data
    load_workspace_data(verbose=True)
    
    # Create environment
    env = RobotPositioningRevampedEnv(
        gui=True,
        viz_speed=viz_speed,
        verbose=True
    )
    
    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Check for normalization statistics
    vec_normalize_path = model_path.replace("final_model", "vec_normalize_stats")
    if os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=vec_env)
    
    # Setup video recording if requested
    if save_video:
        import imageio
        frames = []
        
        # Set higher resolution for video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=120,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.3]
        )
    
    # Run demo sequence (5 episodes)
    for ep in range(5):
        print(f"\nDemo Episode {ep+1}/5")
        
        # Reset environment - handle different API versions
        try:
            # Try new gymnasium API (returns obs, info)
            obs, info = vec_env.reset()
            initial_distance = info[0].get('initial_distance', 0.0)
        except ValueError:
            # Fall back to older gym API (returns only obs)
            obs = vec_env.reset()
            # Estimate initial distance as best we can
            state = env.robot._get_state()
            ee_position = state[12:15]
            initial_distance = np.linalg.norm(ee_position - env.target_position)
        
        print(f"Initial distance to target: {initial_distance*100:.2f}cm")
        
        done = False
        step = 0
        
        # Pre-episode pause for better visualization
        if save_video:
            for _ in range(30):  # Capture 30 frames of initial state
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
        else:
            time.sleep(1.0)
        
        # Run episode
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Enforce joint limits before stepping (get current joint positions)
            state = env.robot._get_state()
            current_joint_positions = state[:env.robot.dof*2:2]
            
            # Calculate new joint positions from action deltas
            new_joint_positions = []
            for i, delta in enumerate(action[0]):  # action is wrapped in array due to vectorized env
                new_pos = current_joint_positions[i] + delta
                new_joint_positions.append(new_pos)
            
            # Enforce limits
            limited_positions = ensure_joint_limits(env.robot, new_joint_positions)
            
            # If action was limited, create a new action array with deltas that respect limits
            if not np.array_equal(new_joint_positions, limited_positions):
                limited_deltas = []
                for i, (pos, limited_pos) in enumerate(zip(new_joint_positions, limited_positions)):
                    if pos != limited_pos:
                        # Calculate the allowed delta that respects the limit
                        allowed_delta = limited_pos - current_joint_positions[i]
                        limited_deltas.append(allowed_delta)
                    else:
                        limited_deltas.append(action[0][i])
                
                # Replace action with limited version
                action = np.array([limited_deltas])
            
            # Step environment
            try:
                # Try new gymnasium API
                obs, reward, terminated, truncated, info = vec_env.step(action)
                done = terminated[0] or truncated[0]
            except ValueError:
                # Fall back to older gym API
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
            
            # Capture frame for video or delay for visualization
            if save_video:
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
            else:
                time.sleep(viz_speed)
            
            step += 1
        
        # Post-episode pause
        if save_video:
            for _ in range(30):  # Capture 30 frames of final state
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
        else:
            time.sleep(1.0)
        
        # Print episode results
        try:
            success = info[0].get('target_reached', False)
            distance = info[0].get('distance', 0.0)
        except (IndexError, TypeError):
            success = False
            distance = 0.0
            
        print(f"Episode {ep+1} - {'SUCCESS' if success else 'FAILURE'}")
        print(f"  Final distance: {distance*100:.2f}cm")
        print(f"  Steps: {step}")
    
    # Save video if requested
    if save_video:
        from datetime import datetime
        video_path = f"./evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")
    
    # Close environment
    vec_env.close()

def parse_custom_args():
    """Parse command line arguments with our own parser."""
    parser = argparse.ArgumentParser(description='Train or demo robot arm positioning')
    
    # Basic options
    parser.add_argument('--steps', type=int, default=10000, help='Total number of training steps')
    parser.add_argument('--load', type=str, default=None, help='Load a pre-trained model')
    
    # Evaluation options
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    
    # Demo options
    parser.add_argument('--demo', action='store_true', help='Run demonstration of the model')
    parser.add_argument('--save-video', action='store_true', help='Save video of the demonstration')
    parser.add_argument('--viz-speed', type=float, default=0.02, help='Visualization speed (delay in seconds)')
    
    # Training options
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--gui', action='store_true', default=True, help='Enable GUI visualization')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI visualization')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
    parser.add_argument('--strict-limits', action='store_true', default=True, 
                      help='Strictly enforce joint limits from URDF (default: True)')
    
    # GPU options
    parser.add_argument('--use-gpu', action='store_true', help='Force use of GPU if available')
    parser.add_argument('--use-amd', action='store_true', help='Specifically use AMD GPU with ROCm/HIP')
    parser.add_argument('--use-cpu', action='store_true', help='Force use of CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Handle gui/no-gui conflict
    if args.no_gui:
        args.gui = False
    
    return args

def main():
    """Main function."""
    # Parse arguments with our custom parser
    args = parse_custom_args()
    
    # Check GPU support
    gpu_info = check_gpu_support()
    gpu_available = gpu_info['available']
    gpu_backend = gpu_info['backend']
    
    # Determine if we should use GPU based on arguments and availability
    use_gpu = False
    if args.use_cpu:
        use_gpu = False
        print("Forcing CPU usage as requested")
    elif args.use_amd and gpu_backend != 'hip':
        print("WARNING: AMD GPU (ROCm/HIP) specifically requested but not available")
        print("To use your AMD Radeon RX 6700S GPU, please install PyTorch with ROCm support")
        print("CPU will be used instead")
        use_gpu = False
    elif args.use_gpu or args.use_amd:
        if gpu_available:
            use_gpu = True
            print(f"Using GPU with backend: {gpu_backend}")
            if gpu_backend == 'cuda':
                print(f"CUDA Device: {gpu_info['device_name']}")
            elif gpu_backend == 'hip':
                print(f"AMD GPU with ROCm/HIP support detected")
            
            # Set environment variables to prefer the AMD GPU if requested
            if args.use_amd and 'hip' in gpu_backend:
                os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use first AMD GPU
        else:
            print("GPU usage requested but no compatible GPU found. Using CPU instead.")
            use_gpu = False
    else:
        if gpu_available:
            print(f"GPU available with backend: {gpu_backend}")
            print("Use --use-gpu to enable GPU acceleration")
        use_gpu = False
    
    # Print information about joint limit enforcement
    if args.strict_limits:
        print("Joint angle limits from URDF model will be strictly enforced")
    
    # Load workspace data
    load_workspace_data(verbose=True)
    
    # Demo mode
    if args.demo:
        if args.load is None:
            print("Error: Must provide a model path with --load for demonstration")
            return
        
        # Use our wrapper function instead of the original
        run_evaluation_sequence_wrapper(
            model_path=args.load,
            viz_speed=args.viz_speed,
            save_video=args.save_video
        )
        return
    
    # Evaluation mode
    if args.eval_only:
        if args.load is None:
            print("Error: Must provide a model path with --load for evaluation")
            return
        
        print(f"Evaluating model: {args.load}")
        # Use our wrapper function instead of the original
        evaluate_model_wrapper(
            model_path=args.load,
            num_episodes=args.eval_episodes,
            visualize=True,
            verbose=True
        )
        return
    
    # Training mode
    print(f"Training a new model for {args.steps} steps")
    
    # Create parameter object that matches what train_revamped_robot expects
    class Args:
        pass
    
    train_args = Args()
    train_args.steps = args.steps
    train_args.load = args.load
    train_args.parallel = args.parallel
    train_args.gui = args.gui
    train_args.no_gui = args.no_gui
    train_args.parallel_viz = False
    train_args.viz_speed = args.viz_speed
    train_args.learning_rate = args.learning_rate
    train_args.use_cuda = use_gpu  # Set based on our GPU detection and user preferences
    train_args.seed = args.seed
    train_args.verbose = args.verbose
    train_args.algorithm = 'ppo'
    train_args.eval_only = False
    train_args.eval_episodes = 0
    train_args.demo = False
    train_args.save_video = False
    train_args.strict_limits = args.strict_limits
    
    # Set additional environment variables for AMD GPU if requested
    if args.use_amd and gpu_backend == 'hip':
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'  # Optimize memory allocation
    
    train_revamped_robot(train_args)

if __name__ == "__main__":
    main() 