#!/usr/bin/env python3
# train_robot_rl_demo_directml.py
# Modified version of train_robot_rl_demo.py with DirectML support for AMD GPUs

import os
import sys
import argparse
import time
import numpy as np
import pybullet as p
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the original script, but don't let it parse arguments
original_argv = sys.argv
sys.argv = [sys.argv[0]]

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

# Check for GPU support including DirectML for AMD GPUs
def check_gpu_support():
    """Check for GPU support in PyTorch including CUDA, ROCm/HIP, and DirectML."""
    result = {
        'backend': 'cpu',
        'available': False,
        'devices': 0,
        'device_name': None,
        'directml_available': False
    }
    
    # Check for CUDA support
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        result['backend'] = 'cuda'
        result['available'] = True
        result['devices'] = torch.cuda.device_count()
        result['device_name'] = torch.cuda.get_device_name(0)
        return result
    
    # Check for ROCm/HIP support (AMD GPUs)
    has_rocm = False
    try:
        has_rocm = hasattr(torch, 'hip') and torch.hip.is_available()
    except:
        try:
            # Check for HIP version in torch.version
            has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        except:
            pass
    
    if has_rocm:
        result['backend'] = 'hip'
        result['available'] = True
        result['devices'] = torch.hip.device_count() if hasattr(torch.hip, 'device_count') else 1
        result['device_name'] = 'AMD GPU (ROCm/HIP)'
        return result
    
    # Check for DirectML support
    try:
        import torch_directml
        dml_device = torch_directml.device()
        result['backend'] = 'directml'
        result['available'] = True
        result['devices'] = 1  # DirectML typically sees one device
        result['device_name'] = 'AMD GPU (DirectML)'
        result['directml_available'] = True
        return result
    except (ImportError, AttributeError):
        pass
    
    # No GPU support found
    return result

# Import JointLimitEnforcingEnv class and other utility functions from the original script
from train_robot_rl_demo import (
    ensure_joint_limits,
    JointLimitEnforcingEnv,
    evaluate_model_wrapper,
    run_evaluation_sequence_wrapper
)

# Custom wrapper class for DirectML support
class DirectMLModel:
    """Wrapper for models to use DirectML device for inference."""
    
    def __init__(self, model):
        self.model = model
        
        # Import torch_directml
        import torch_directml
        self.dml_device = torch_directml.device()
        
        # Move model to DirectML device
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'to'):
            self.model.policy.to(self.dml_device)
        
        # Cache for observations to reduce repeated transfers
        self._obs_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100
        
        # Batch processing settings
        self._batch_size = 32  # Process observations in batches for better GPU utilization
        self._batch_buffer = []
        self._batch_results = []
    
    def _observation_to_key(self, observation):
        """Convert observation to a hashable key for caching."""
        if isinstance(observation, np.ndarray):
            return hash(observation.tobytes())
        elif isinstance(observation, torch.Tensor):
            return hash(observation.cpu().numpy().tobytes())
        return None
    
    def _maybe_process_batch(self, force=False):
        """Process the batched observations if the batch is full or if force is True."""
        if self._batch_buffer is None or len(self._batch_buffer) == 0:
            return
        
        if force or len(self._batch_buffer) >= self._batch_size:
            try:
                # Move to CPU for processing (sync point)
                cpu_observations = [obs.copy() for obs in self._batch_buffer]
                
                # Process batch with the model
                actions, states = self.model.predict(cpu_observations, deterministic=True)
                
                # Store results
                self._batch_results = list(zip(actions, states))
                self._batch_buffer = []
            except Exception as e:
                print(f"Error in DirectML batch processing: {e}")
                # Fallback to non-batched processing
                self._batch_results = []
                for i, obs in enumerate(self._batch_buffer):
                    action, state = self.model.predict([obs], deterministic=True)
                    self._batch_results.append((action[0], state))
                self._batch_buffer = []
    
    def predict(self, observation, deterministic=True, **kwargs):
        """Run prediction with DirectML support and caching."""
        # Special case for demo/eval with vectorized environment
        if isinstance(observation, np.ndarray) and observation.shape == (1, 67):
            # This is a VecEnv observation, handle it directly
            with torch.no_grad():
                return self.model.predict(observation, deterministic=deterministic, **kwargs)
        
        # Check cache first if observation is cacheable
        cache_key = self._observation_to_key(observation)
        if cache_key is not None and cache_key in self._obs_cache:
            self._cache_hits += 1
            return self._obs_cache[cache_key]
        else:
            self._cache_misses += 1
        
        # Process batched input if available
        if hasattr(observation, 'shape') and len(observation.shape) > 1 and observation.shape[0] > 1:
            # This is already a batch
            if isinstance(observation, np.ndarray):
                # Keep observation as numpy array for the model's predict function
                processed_observation = observation
            elif isinstance(observation, torch.Tensor):
                # Convert to numpy if it's a tensor
                processed_observation = observation.cpu().numpy() if observation.device.type != 'cpu' else observation.numpy()
            else:
                processed_observation = observation
                
            # Process the whole batch at once
            with torch.no_grad():
                actions, states = self.model.predict(processed_observation, deterministic=deterministic, **kwargs)
                
            # No need to cache batched results
            return actions, states
        
        # For single observations, process directly
        if isinstance(observation, np.ndarray):
            # Keep as numpy array
            processed_observation = observation
        elif isinstance(observation, torch.Tensor):
            # Convert to numpy
            processed_observation = observation.cpu().numpy() if observation.device.type != 'cpu' else observation.numpy()
        else:
            processed_observation = observation
        
        # Get prediction from model
        with torch.no_grad():
            actions, states = self.model.predict(processed_observation, deterministic=deterministic, **kwargs)
        
        # Cache the result if appropriate
        if cache_key is not None:
            # Maintain cache size limit
            if len(self._obs_cache) >= self._max_cache_size:
                # Remove a random entry
                self._obs_cache.pop(next(iter(self._obs_cache)))
            self._obs_cache[cache_key] = (actions, states)
            
        return actions, states
    
    def process_action(self, action):
        """Post-process action if needed (e.g., apply to DirectML tensors)."""
        return action
        
    def cleanup(self):
        """Clean up resources and report cache statistics."""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            print(f"DirectML cache hit rate: {hit_rate:.1f}% ({self._cache_hits}/{total})")
        self._obs_cache.clear()
        # Process any remaining batched observations
        self._maybe_process_batch(force=True)

def parse_custom_args():
    """Parse command line arguments with our own parser."""
    parser = argparse.ArgumentParser(description='Train or demo robot arm positioning with DirectML support for AMD GPUs')
    
    # Basic options
    parser.add_argument('--steps', type=int, default=300000, help='Total number of training steps')
    parser.add_argument('--load', type=str, default=None, help='Load a pre-trained model')
    
    # Evaluation options
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--eval-episodes', type=int, default=20, help='Number of episodes for evaluation')
    
    # Demo options
    parser.add_argument('--demo', action='store_true', help='Run demonstration of the model')
    parser.add_argument('--save-video', action='store_true', help='Save video of the demonstration')
    parser.add_argument('--viz-speed', type=float, default=0.02, help='Visualization speed (delay in seconds)')
    
    # Training options
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--parallel-viz', action='store_true', help='Enable visualization for parallel environments')
    parser.add_argument('--gui', action='store_true', default=True, help='Enable GUI visualization')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI visualization')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
    parser.add_argument('--strict-limits', action='store_true', help='Strictly enforce joint limits from URDF')
    parser.add_argument('--algorithm', choices=['ppo', 'sac', 'td3'], default='ppo', help='RL algorithm to use')
    
    # GPU options
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    # DirectML-specific arguments
    parser.add_argument('--disable-directml', action='store_true',
                        help='Disable DirectML acceleration (run on CPU instead)')
    parser.add_argument('--sync-freq', type=int, default=10,
                        help='How often to synchronize with DirectML device (in steps)')
    parser.add_argument('--memory-limit', type=int, default=128,
                        help='Memory limit for DirectML operations in MB')
    parser.add_argument('--optimized-directml', action='store_true',
                        help='Use the optimized DirectML PPO implementation for maximum GPU performance')
    
    args = parser.parse_args()
    
    # Handle gui/no-gui conflict
    if args.no_gui:
        args.gui = False
    
    return args

def main():
    """Main entry point for training."""
    # Parse arguments
    args = parse_custom_args()
    
    # Check if DirectML is available
    dml_device = check_directml_availability()
    
    # Enable parallel visualization for DirectML training
    if dml_device is not None and getattr(args, 'optimized_directml', False):
        print("Enabling visualization for", args.parallel, "parallel environments...")
        args.parallel_viz = True
        args.gui = True
        args.no_gui = False
        # Train with DirectML
        train_with_directml(args)
    else:
        # Default training without DirectML
        args.func(args)

# Wrap the original evaluation functions with DirectML support
def evaluate_model_wrapper(model_path, num_episodes=20, visualize=True, verbose=True, use_directml=False):
    """Evaluate a model with optional DirectML support."""
    print(f"Evaluating model at {model_path}")
    
    # If DirectML is requested, wrap the model
    if use_directml:
        try:
            import torch_directml
            
            # Custom implementation for DirectML
            print("Using DirectML-powered evaluation")
            
            # Create environment
            env = RobotPositioningRevampedEnv(
                gui=visualize,
                viz_speed=0.01,
                verbose=verbose
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
            
            # Wrap with DirectML
            dml_model = DirectMLModel(model)
            
            # Run evaluation
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
                print(f"\nEvaluation Episode {ep+1}/{num_episodes}")
                
                # Reset environment
                try:
                    obs, info = vec_env.reset()
                    initial_distance = info[0].get('initial_distance', 0.0)
                except ValueError:
                    obs = vec_env.reset()
                    state = env.robot._get_state()
                    ee_position = state[12:15]
                    initial_distance = np.linalg.norm(ee_position - env.target_position)
                
                print(f"Initial distance to target: {initial_distance*100:.2f}cm")
                
                # Initialize episode variables
                done = False
                ep_reward = 0.0
                ep_steps = 0
                best_distance = initial_distance
                
                # Run episode
                while not done:
                    # Get action from DirectML model
                    action, _ = dml_model.predict(obs, deterministic=True)
                    
                    # Step environment
                    try:
                        obs, reward, terminated, truncated, info = vec_env.step(action)
                        done = terminated[0] or truncated[0]
                    except ValueError:
                        obs, reward, done, info = vec_env.step(action)
                        done = done[0]
                    
                    # Update metrics
                    ep_reward += reward[0]
                    ep_steps += 1
                    
                    try:
                        current_distance = info[0].get('distance', 0.0)
                    except (IndexError, TypeError):
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
            
        except ImportError:
            print("WARNING: DirectML requested but not available. Falling back to CPU.")
            return original_evaluate_model(
                model_path=model_path,
                num_episodes=num_episodes,
                visualize=visualize,
                verbose=verbose
            )
    else:
        # Standard evaluation
        return original_evaluate_model(
            model_path=model_path,
            num_episodes=num_episodes,
            visualize=visualize,
            verbose=verbose
        )

def run_evaluation_sequence_wrapper(model_path, viz_speed=0.02, save_video=False, use_directml=False):
    """Run an evaluation sequence with optional DirectML support."""
    print(f"Running evaluation sequence for model at {model_path}")
    
    # If DirectML is requested, wrap the model
    if use_directml:
        try:
            import torch_directml
            
            # Custom implementation for DirectML
            print("Using DirectML-powered demonstration")
            
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
            
            # Wrap model with DirectML support
            dml_model = DirectMLModel(model)
            
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
                
                # Reset environment
                try:
                    obs, info = vec_env.reset()
                    initial_distance = info[0].get('initial_distance', 0.0)
                except ValueError:
                    obs = vec_env.reset()
                    state = env.robot._get_state()
                    ee_position = state[12:15]
                    initial_distance = np.linalg.norm(ee_position - env.target_position)
                
                print(f"Initial distance to target: {initial_distance*100:.2f}cm")
                
                done = False
                step = 0
                
                # Pre-episode pause for better visualization
                if save_video:
                    for _ in range(30):
                        img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                        frames.append(img[2])
                else:
                    time.sleep(1.0)
                
                # Run episode
                while not done:
                    # Get action from DirectML model
                    action, _ = dml_model.predict(obs, deterministic=True)
                    
                    # Step environment
                    try:
                        obs, reward, terminated, truncated, info = vec_env.step(action)
                        done = terminated[0] or truncated[0]
                    except ValueError:
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
                    for _ in range(30):
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
            
        except ImportError:
            print("WARNING: DirectML requested but not available. Falling back to CPU.")
            return original_run_evaluation_sequence(
                model_path=model_path,
                viz_speed=viz_speed,
                save_video=save_video
            )
    else:
        # Standard evaluation
        return original_run_evaluation_sequence(
            model_path=model_path,
            viz_speed=viz_speed,
            save_video=save_video
        )

# Define DirectML-specific functions
def patch_torch_for_directml(dml_device):
    """Apply patches to PyTorch to redirect operations to DirectML."""
    print("Patching PyTorch functions to use DirectML...")
    
    # Store original PyTorch functions for later restoration
    original_tensor_to = torch.Tensor.to
    original_torch_device = torch.device
    
    # Set DirectML-specific environment variables for better performance
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['DML_PREFETCH_BUFFERS'] = '1'  # Enable prefetching
    
    # Try to set optimal DirectML thread settings
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        # Use half of available threads, but at least 4
        dml_thread_count = max(4, cpu_count // 2)
        os.environ['DML_THREAD_COUNT'] = str(dml_thread_count)
        print(f"Setting DirectML to use {dml_thread_count} threads")
    except:
        pass
    
    # Override torch.device to intercept 'cuda' calls and redirect to DirectML
    def directml_device_override(device_str='cpu'):
        if isinstance(device_str, str) and (device_str == 'cuda' or device_str == 'auto'):
            # Only print once for reduce noise
            if not hasattr(directml_device_override, 'shown_redirect_msg'):
                print(f"Redirecting device '{device_str}' to DirectML")
                directml_device_override.shown_redirect_msg = True
            return dml_device
        return original_torch_device(device_str)
    directml_device_override.shown_redirect_msg = False
    
    # Override tensor.to() to intercept CUDA device requests
    def directml_tensor_to_override(self, *args, **kwargs):
        # Check if the destination is 'cuda' and redirect to DirectML
        if args and isinstance(args[0], str) and args[0] == 'cuda':
            # Only print once for reduce noise
            if not hasattr(directml_tensor_to_override, 'shown_redirect_msg'):
                print("Redirecting tensor.to('cuda') to DirectML")
                directml_tensor_to_override.shown_redirect_msg = True
            return original_tensor_to(self, dml_device, *args[1:], **kwargs)
        elif args and hasattr(args[0], 'type') and args[0].type == 'cuda':
            # Only print once for reduce noise
            if not hasattr(directml_tensor_to_override, 'shown_redirect_cuda_msg'):
                print("Redirecting tensor.to(cuda_device) to DirectML")
                directml_tensor_to_override.shown_redirect_cuda_msg = True
            return original_tensor_to(self, dml_device, *args[1:], **kwargs)
        return original_tensor_to(self, *args, **kwargs)
    directml_tensor_to_override.shown_redirect_msg = False
    directml_tensor_to_override.shown_redirect_cuda_msg = False
    
    # Apply the patches
    torch.device = directml_device_override
    torch.Tensor.to = directml_tensor_to_override
    
    # Also need to patch torch.cuda namespace for auto-detection
    if not hasattr(torch, 'directml_original_cuda_is_available'):
        torch.directml_original_cuda_is_available = torch.cuda.is_available
        
    # Make torch.cuda.is_available() return True so frameworks think GPU is available
    def directml_is_available():
        return True
        
    # Patch CUDA availability check
    torch.cuda.is_available = directml_is_available
    
    # Add device count that returns 1 to simulate a single GPU
    if not hasattr(torch, 'directml_original_cuda_device_count'):
        torch.directml_original_cuda_device_count = torch.cuda.device_count
        
    # Make device_count return 1 to simulate a single GPU
    def directml_device_count():
        return 1
        
    torch.cuda.device_count = directml_device_count
    
    # Add a get_device_name function that returns "AMD DirectML"
    if not hasattr(torch, 'directml_original_cuda_get_device_name'):
        if hasattr(torch.cuda, 'get_device_name'):
            torch.directml_original_cuda_get_device_name = torch.cuda.get_device_name
        else:
            torch.directml_original_cuda_get_device_name = lambda x: "Unknown"
    
    def directml_get_device_name(device=None):
        return "AMD Radeon RX 6700S (DirectML)"
        
    torch.cuda.get_device_name = directml_get_device_name
    
    # Set current device to 0
    torch.cuda.current_device = lambda: 0
    
    # Add synchronization functions for DirectML
    if not hasattr(torch.cuda, 'directml_original_synchronize'):
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.directml_original_synchronize = torch.cuda.synchronize
        else:
            torch.cuda.directml_original_synchronize = lambda: None
    
    # Create a synchronize function that forces CPU sync using the more efficient sync_dml
    def directml_synchronize():
        sync_dml(dml_device, force=True)
    
    # Patch synchronize
    torch.cuda.synchronize = directml_synchronize
    
    # Create optimized versions of common tensor operations
    original_matmul = torch.matmul
    def optimized_matmul(input, other, *, out=None):
        # Check if inputs are on the DirectML device
        input_on_dml = isinstance(input, torch.Tensor) and input.device.type == 'privateuseone'
        other_on_dml = isinstance(other, torch.Tensor) and other.device.type == 'privateuseone'
        
        # If both inputs are already on DirectML, use them directly
        if input_on_dml and other_on_dml:
            result = original_matmul(input, other, out=out)
            # No need to sync here, that will happen when needed
            return result
            
        # Move inputs to DirectML if needed
        if not input_on_dml and isinstance(input, torch.Tensor):
            input = input.to(dml_device)
        if not other_on_dml and isinstance(other, torch.Tensor):
            other = other.to(dml_device)
            
        # Perform computation on DirectML
        result = original_matmul(input, other, out=out)
        
        # Leave result on device for now
        return result
    
    # Patch common operations for better performance
    torch.matmul = optimized_matmul
    
    # Also need to patch autograd.Function for custom ops
    if hasattr(torch, 'autograd') and hasattr(torch.autograd, 'Function'):
        class DirectMLFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args, **kwargs):
                # Move any tensor args to DirectML
                dml_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        dml_args.append(arg.to(dml_device))
                    else:
                        dml_args.append(arg)
                
                # Call the original forward
                result = ctx.forward_original(*dml_args, **kwargs)
                
                # Ensure the result is on DirectML
                if isinstance(result, torch.Tensor):
                    result = result.to(dml_device)
                elif isinstance(result, tuple):
                    result = tuple(r.to(dml_device) if isinstance(r, torch.Tensor) else r for r in result)
                
                return result
            
            @staticmethod
            def backward(ctx, *grad_outputs):
                # Implement backward pass with DirectML
                # This is a simplified version - in reality it would depend on the function
                return grad_outputs
    
    # We don't actually replace Function, as that would break too much
    # But we can add this for custom ops that might be defined later
    torch.autograd.DirectMLFunction = DirectMLFunction
    
    print("PyTorch functions patched to use DirectML")
    
    return {
        'original_tensor_to': original_tensor_to,
        'original_torch_device': original_torch_device,
        'original_matmul': original_matmul
    }

# Function to restore original PyTorch behavior
def restore_torch_original(originals):
    """Restore original PyTorch functions."""
    print("Restoring original PyTorch functions...")
    
    # Restore original functions
    if originals and isinstance(originals, dict):
        if 'original_torch_device' in originals:
            torch.device = originals['original_torch_device']
        
        if 'original_tensor_to' in originals:
            torch.Tensor.to = originals['original_tensor_to']
        
        # Restore original matmul if we patched it
        if 'original_matmul' in originals:
            torch.matmul = originals['original_matmul']
    
    # Restore CUDA functions if we patched them
    if hasattr(torch, 'directml_original_cuda_is_available'):
        torch.cuda.is_available = torch.directml_original_cuda_is_available
        
    if hasattr(torch, 'directml_original_cuda_device_count'):
        torch.cuda.device_count = torch.directml_original_cuda_device_count
        
    if hasattr(torch, 'directml_original_cuda_get_device_name'):
        torch.cuda.get_device_name = torch.directml_original_cuda_get_device_name
    
    # Restore synchronize
    if hasattr(torch.cuda, 'directml_original_synchronize'):
        torch.cuda.synchronize = torch.cuda.directml_original_synchronize
    
    # Remove our custom DirectMLFunction if we added it
    if hasattr(torch.autograd, 'DirectMLFunction'):
        delattr(torch.autograd, 'DirectMLFunction')

# DirectML helper functions
def sync_dml(dml_device, force=False):
    """
    Force a synchronization with the DirectML device.
    
    Args:
        dml_device: The DirectML device to synchronize with
        force: If True, always synchronize. If False, only synchronize if enough time has passed
               since the last synchronization (reduces unnecessary sync calls)
    """
    # Use a static variable to track last sync time
    if not hasattr(sync_dml, "last_sync_time"):
        sync_dml.last_sync_time = 0
    
    # Only sync if forced or enough time has passed (50ms)
    current_time = time.time()
    if force or (current_time - sync_dml.last_sync_time > 0.05):
        # Create a small tensor on DirectML, do an operation, and copy back to CPU
        x = torch.ones(1, device=dml_device)
        y = x + 1
        # Moving to CPU forces sync
        _ = y.to('cpu').item()
        sync_dml.last_sync_time = current_time

def to_dml(tensor, dml_device):
    """Move a tensor to the DirectML device with proper handling."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dml_device)
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(to_dml(t, dml_device) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: to_dml(v, dml_device) for k, v in tensor.items()}
    return tensor

def to_cpu(tensor):
    """Move a tensor to CPU with proper handling."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu')
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(to_cpu(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: to_cpu(v) for k, v in tensor.items()}
    return tensor

def train_with_directml(args):
    """Train a robot using the DirectML backend."""
    # Handle the environment variables
    memory_limit = getattr(args, 'memory_limit', 128)
    
    # Set environment variables for DirectML
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = f"max_split_size_mb:{memory_limit}"
    
    # Get a DirectML device
    device_info = check_directml_availability()
    if not device_info:
        print("DirectML is not available. Please install it with: pip install torch-directml")
        return
    
    dml_device = device_info
    
    # Check if we should use the optimized PPO implementation
    optimized_directml = getattr(args, 'optimized_directml', False)
    
    # Print information
    print_system_info(True)
    
    # Estimate VRAM size based on parallel environments
    estimated_vram_mb = args.parallel * 500  # Rough estimate, 500MB per environment
    print(f"Estimated VRAM: {estimated_vram_mb/1024:.1f} GB, Memory limit: {memory_limit} MB")
    
    # Print DirectML device information
    print(f"Using DirectML device: {dml_device}")
    
    # Setup DirectML backend for PyTorch
    print("Patching PyTorch functions to use DirectML...")
    originals = patch_torch_for_directml(dml_device)
    
    # Set thread count for DirectML (experimental)
    thread_count = os.cpu_count() // 2 if os.cpu_count() else 4  # Use half of available cores
    print(f"Setting DirectML to use {thread_count} threads")
    
    try:
        # If we're using the optimized PPO implementation, call that directly
        if optimized_directml:
            # Enable visualization of multiple robots in the same window
            args.parallel_viz = True
            args.gui = True
            args.no_gui = False
            
            print("Using fully optimized DirectML PPO implementation...")
            train_ppo_with_directml(args, dml_device)
        else:
            # Otherwise proceed with normal training but on DirectML device
            args.func(args)
    finally:
        # Restore original PyTorch functions
        restore_torch_original(originals)

def train_ppo_with_directml(args, dml_device):
    """
    Implement PPO training with DirectML optimizations for AMD GPUs.
    This function replaces the standard PPO implementation with a version
    that runs key operations on the DirectML device.
    
    Args:
        args: Arguments for training
        dml_device: DirectML device to use for computation
    """
    import importlib
    from datetime import datetime
    
    # Import our DirectML-optimized PPO implementation
    try:
        # First try to import the module - it will be in the same directory
        directml_ppo = importlib.import_module('directml_ppo')
    except ImportError:
        print("Could not import directml_ppo module. Make sure it's in the same directory.")
        print("Falling back to standard implementation.")
        args.func(args)
        return
    
    print(f"Setting up DirectML-optimized PPO training...")
    
    # Import necessary libraries
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import numpy as np
    import torch
    
    # Setup environment based on the original function's logic
    if args.algorithm.lower() != 'ppo':
        print("DirectML optimization only available for PPO algorithm.")
        print("Falling back to standard implementation.")
        args.func(args)
        return
    
    # Create environments using the same mechanism as the original train function
    print("Creating and wrapping environments...")
    from train_robot_rl_positioning_revamped import create_revamped_envs, SaveModelCallback, TrainingMonitorCallback, CustomPPO
    
    # Set up environment - handle attribute name differences
    use_parallel_viz = False
    if hasattr(args, 'parallel_viz'):
        use_parallel_viz = args.parallel_viz
    elif hasattr(args, 'no_parallel_viz'):
        use_parallel_viz = not args.no_parallel_viz
    
    # Create the environment
    env_list = create_revamped_envs(args.parallel, args.viz_speed, use_parallel_viz)
    
    # The function might return a list of environments or a VecEnv directly
    if isinstance(env_list, list):
        env = DummyVecEnv([lambda env=individual_env: env for individual_env in env_list])
    else:
        env = env_list
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create model arguments
    model_args = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'verbose': args.verbose,
        'dml_device': dml_device,  # Pass the DirectML device
    }
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("./models", f"revamped_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create SaveModelCallback for DirectML PPO
    class DirectMLSaveModelCallback(BaseCallback):
        def __init__(self, save_freq, save_path, verbose=0):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path
            
        def _on_step(self):
            if self.n_calls % self.save_freq == 0:
                model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
                if self.verbose > 0:
                    print(f"Saving model to {model_path}")
                self.model.save(model_path)
                # Also save the environment
                env_path = os.path.join(self.save_path, f"env_{self.n_calls}")
                self.training_env.save(env_path)
            return True
    
    # Create performance tracking callback for DirectML PPO
    class DirectMLPerformanceCallback(BaseCallback):
        def __init__(self, log_interval=100, verbose=0):
            super().__init__(verbose)
            self.log_interval = log_interval
            self.start_time = time.time()
            self.episode_rewards = []
            self.episode_lengths = []
            self.last_log_time = time.time()
            
        def _on_step(self):
            if self.n_calls % self.log_interval == 0:
                # Calculate FPS
                elapsed = time.time() - self.last_log_time
                fps = int(self.log_interval / elapsed)
                
                # Log performance
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}, FPS: {fps}")
                    # Try to get memory usage
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        memory_gb = memory_info.rss / (1024**3)
                        print(f"Memory usage: {memory_gb:.2f} GB")
                    except:
                        pass
                
                self.last_log_time = time.time()
                
            return True
    
    # List of callbacks
    callbacks = []
    
    # Add save model callback
    save_freq = min(10000, args.steps // 10)
    callbacks.append(DirectMLSaveModelCallback(save_freq=save_freq, save_path=save_path, verbose=args.verbose))
    
    # Add performance callback
    if args.verbose:
        callbacks.append(DirectMLPerformanceCallback(log_interval=100, verbose=args.verbose))
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Initialize the DirectML PPO model
    print("Creating DirectML-optimized PPO model...")
    
    if args.load:
        print(f"Loading pre-trained model from {args.load}...")
        # Use our DirectML PPO's load method
        model = directml_ppo.DirectMLPPO.load(args.load, env=env, dml_device=dml_device)
        # Update parameters that can change between training sessions
        model.learning_rate = args.learning_rate
        model.verbose = args.verbose
    else:
        print("Creating new DirectML-optimized PPO model...")
        model = directml_ppo.DirectMLPPO(**model_args)
    
    # Attach callbacks to the model for proper initialization
    for cb in callback.callbacks:
        cb.model = model
        if hasattr(cb, 'training_env'):
            cb.training_env = env
    
    # Start training
    print(f"Starting DirectML-optimized PPO training for {args.steps} steps...")
    
    # Train the model with our DirectML-optimized PPO implementation
    model.learn(total_timesteps=args.steps, callback=callback, log_interval=1)
    
    # Save the final model
    print(f"Saving final model to {save_path}/final_model")
    model.save(os.path.join(save_path, "final_model"))
    
    # Save also the normalized environment
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    # Close environment
    env.close()
    
    print(f"DirectML-optimized PPO training completed: {args.steps} steps")

def check_directml_availability():
    """Check if DirectML is available and return a DirectML device if available."""
    try:
        import torch_directml
        # Return the actual DirectML device object instead of just True
        return torch_directml.device()
    except ImportError:
        return None

def print_system_info(has_directml):
    """Print system information including GPU and memory status."""
    import platform
    import psutil
    
    # Get memory information
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024**3)  # Convert to GB
    available_memory = mem.available / (1024**3)  # Convert to GB
    
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=True)
    
    # Get PyTorch version
    torch_version = torch.__version__
    
    # Print system information
    print(f"System: {platform.system()} ({sys.platform})")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch_version}")
    print(f"CPU Threads: {cpu_count}")
    print(f"Memory: {total_memory:.1f} GB total, {available_memory:.1f} GB available")
    
    # Print GPU information
    if has_directml is not None:
        try:
            import torch_directml
            dml_version = getattr(torch_directml, "__version__", "Unknown")
            print(f"DirectML: Available (v{dml_version})")
            print(f"GPU: AMD GPU detected (DirectML backend)")
        except ImportError:
            print("DirectML: Not available (module not found)")
    else:
        print("DirectML: Not available")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"CUDA: Available ({device_count} devices)")
            print(f"GPU: {device_name}")
        else:
            print("CUDA: Not available")
            print("GPU: Not detected or not supported")
            
    print(f"Training Mode: {'GPU (DirectML)' if has_directml is not None else 'CPU'}")
    print("=========================================")

if __name__ == "__main__":
    # Parse arguments
    args = parse_custom_args()
    
    # Check if DirectML is available
    dml_device = check_directml_availability()
    if not dml_device:
        print("\n" + "="*70)
        print("WARNING: DirectML is not available but required for AMD GPU acceleration")
        print("To install DirectML, run the following command:")
        print("    pip install torch-directml")
        print("After installing, restart this script")
        print("="*70 + "\n")
        
        # Ask user if they want to continue with CPU-only
        if sys.stdin.isatty():  # Only ask if running in interactive terminal
            response = input("Do you want to continue with CPU-only mode? (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Please install DirectML and try again.")
                sys.exit(1)
    
    print("\n" + "="*50)
    print("Robot Training with DirectML Support")
    print_system_info(dml_device)
    print("="*50 + "\n")
    
    # Handle evaluation mode
    if args.eval_only:
        if not args.load:
            print("ERROR: Must specify a model to load with --load when using --eval-only")
            sys.exit(1)
            
        # Evaluate the model
        evaluate_model_wrapper(
            model_path=args.load,
            num_episodes=args.eval_episodes,
            visualize=True,
            verbose=args.verbose,
            use_directml=dml_device is not None
        )
        sys.exit(0)
        
    # Handle demo mode
    if args.demo:
        if not args.load:
            print("ERROR: Must specify a model to load with --load when using --demo")
            sys.exit(1)
            
        # Run the demo
        run_evaluation_sequence_wrapper(
            model_path=args.load,
            viz_speed=args.viz_speed or 0.01,
            save_video=args.save_video,
            use_directml=dml_device is not None
        )
        sys.exit(0)
        
    # Main training
    if dml_device is not None:
        # Check if using optimized DirectML implementation
        if args.optimized_directml and args.algorithm.lower() == 'ppo':
            print("\nUsing fully optimized DirectML PPO implementation for maximum performance\n")
        
        # Train with DirectML
        main()
    else:
        print("\nNOTE: Training will use CPU only. Install DirectML for GPU acceleration.\n")
        # Train with CPU
        train_revamped_robot(args) 