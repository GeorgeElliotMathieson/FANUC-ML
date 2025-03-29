"""
Evaluation functionality for the FANUC Robot ML Platform.

This module provides functions for evaluating trained models,
including visualization of evaluation results.
"""

import os
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def print_eval_usage():
    """Print usage information for evaluation mode."""
    print("Evaluation Mode Usage:")
    print("  python fanuc_platform.py eval <model-path> [num_episodes] [options]")
    print("")
    print("Arguments:")
    print("  model-path    - Path to the trained model file")
    print("  num_episodes  - Number of episodes to evaluate (default: 10)")
    print("")
    print("Options:")
    print("  --no-gui       - Disable visualization")
    print("  --verbose      - Enable verbose output")
    print("  --speed        - Set visualization speed (default: 0.02)")
    print("  --max-steps    - Maximum steps per episode before timeout (default: 1000)")
    print("")

def create_eval_environment(visualize=True, viz_speed=0.02, verbose=False):
    """
    Create an environment for model evaluation.
    
    Args:
        visualize: Whether to use visualization (can be overridden by FANUC_VISUALIZE env var)
        viz_speed: Visualization speed (seconds per step)
        verbose: Whether to print verbose output (can be overridden by FANUC_VERBOSE env var)
        
    Returns:
        Evaluation environment
    """
    # Dynamic import to avoid circular imports
    try:
        from src.core.env import JointLimitEnforcingEnv
        from src.utils.pybullet_utils import get_visualization_settings_from_env
    except ImportError as e:
        raise ImportError(f"Failed to import required modules for evaluation: {e}")
    
    # Check environment variables to potentially override parameters
    try:
        env_visualize, env_verbose = get_visualization_settings_from_env()
        # Use environment values if variables are explicitly set
        if 'FANUC_VISUALIZE' in os.environ:
            visualize = env_visualize
            if verbose:
                print(f"Using FANUC_VISUALIZE environment variable: {visualize}")
        if 'FANUC_VERBOSE' in os.environ:
            verbose = env_verbose
            print(f"Using FANUC_VERBOSE environment variable: {verbose}")
    except Exception as e:
        if verbose:
            print(f"Warning: Error getting environment variables: {e}")
            print("Using provided function parameters instead")
    
    # Handle import of RobotPositioningRevampedEnv with try-except
    try:
        from src.envs.robot_sim import RobotPositioningRevampedEnv
    except ImportError as e:
        if verbose:
            print(f"Warning: Could not import RobotPositioningRevampedEnv: {e}")
            print("Falling back to RobotPositioningEnv")
        try:
            from src.envs.robot_sim import RobotPositioningEnv as RobotPositioningRevampedEnv
        except ImportError as e2:
            raise ImportError(f"Could not import robot environment classes from src.envs.robot_sim: {e2}")
    
    # Create the environment - passing visualize to gui parameter for consistency
    try:
        env = RobotPositioningRevampedEnv(
            gui=visualize,  # Using visualize parameter for gui
            viz_speed=viz_speed,
            verbose=verbose,
            training_mode=False
        )
        
        # Apply joint limit enforcement wrapper
        env = JointLimitEnforcingEnv(env)
        
        # Vectorize for compatibility with SB3
        vec_env = DummyVecEnv([lambda: env])
        
        return vec_env
    except Exception as e:
        raise RuntimeError(f"Failed to create evaluation environment: {e}")

def evaluate_model_wrapper(model_path, num_episodes=10, visualize=True, verbose=False, max_steps=1000, viz_speed=0.02):
    """
    Wrapper for model evaluation that loads the model and runs evaluation.
    
    Args:
        model_path: Path to the model to evaluate
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        max_steps: Maximum number of steps per episode before timing out
        viz_speed: Visualization speed (seconds per step)
        
    Returns:
        Dictionary of evaluation results
    """
    from src.core.utils import ensure_model_file_exists
    from src.core.models import CustomPPO
    
    # Ensure model file exists and has correct extension
    model_path = ensure_model_file_exists(model_path)
    
    # Create environment for evaluation
    env = create_eval_environment(
        visualize=visualize,
        viz_speed=viz_speed if visualize else 0.0,
        verbose=verbose
    )
    
    # Check for normalization stats
    stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize_stats")
    if os.path.exists(stats_path):
        if verbose:
            print(f"Loading normalization stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        # Don't update stats during evaluation
        env.training = False
        env.norm_reward = False
    
    # Load the model
    try:
        print(f"Attempting to load model from {model_path} using CustomPPO...")
        model = CustomPPO.load(model_path, env=env)
        print(f"Successfully loaded model from {model_path} using CustomPPO")
    except Exception as e:
        print(f"Error loading model with PPO: {e}")
        try:
            from src.core.models import CustomDirectMLModel
            print("Attempting to load model using DirectML...")
            model = CustomDirectMLModel(env.observation_space, env.action_space)
            success = model.load(model_path)
            if not success:
                print(f"Failed to load model as DirectML model")
                return None
            print(f"Successfully loaded model from {model_path} using CustomDirectMLModel")
        except Exception as e2:
            print(f"Error loading model as DirectML: {e2}")
            print(f"Could not load model from {model_path} using any available model class")
            return None
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        env=env,
        num_episodes=num_episodes,
        visualize=visualize,
        verbose=verbose,
        max_steps=max_steps,
        viz_speed=viz_speed
    )
    
    return results

def evaluate_model(model, env, num_episodes=10, visualize=True, verbose=False, max_steps=1000, viz_speed=0.02):
    """
    Evaluate a model on the environment.
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        max_steps: Maximum number of steps per episode before timing out
        viz_speed: Visualization speed (seconds per step) - passed to env during creation
        
    Returns:
        Dictionary of evaluation results
    """
    # Lists to store metrics
    rewards = []
    episode_lengths = []
    success = []
    distances = []
    times = []
    
    # Reset the environment
    obs = env.reset()
    
    # Evaluation loop
    for i in range(num_episodes):
        # Print episode start
        if verbose:
            print(f"\nEvaluating episode {i+1}/{num_episodes}...")
        
        # Reset metrics for this episode
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_distance = float('inf')
        
        # Record start time
        start_time = time.time()
        
        # Reset environment
        obs = env.reset()
        done = False
        
        # Episode loop
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward[0]
            episode_length += 1
            
            # Check for success and distance - handle different info formats
            current_info = None
            if isinstance(info, list) and len(info) > 0:
                current_info = info[0]  # Get info from first environment
            elif isinstance(info, dict):
                current_info = info
                
            if current_info is not None:
                if 'success' in current_info:
                    episode_success = episode_success or current_info['success'] # Once successful, stays successful
                if 'distance' in current_info:
                    episode_distance = min(episode_distance, current_info['distance'])
                    
                # Print step information if verbose
                if verbose and episode_length % 10 == 0:
                    print(f"  Step {episode_length}: reward={reward[0]:.3f}, "
                          f"distance={current_info.get('distance', float('inf')):.4f}m")
            
            # Break early if episode times out
            if episode_length >= max_steps:  # Timeout
                if verbose:
                    print("  Episode timed out.")
                break
                
        # Calculate episode time
        episode_time = time.time() - start_time
        
        # Store metrics
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success.append(episode_success)
        distances.append(episode_distance)
        times.append(episode_time)
        
        # Print episode results
        if verbose:
            print(f"  Episode {i+1} finished:")
            print(f"    Reward: {episode_reward:.2f}")
            print(f"    Length: {episode_length}")
            print(f"    Success: {episode_success}")
            print(f"    Best distance: {episode_distance:.4f}m")
            print(f"    Time: {episode_time:.2f}s")
    
    # Calculate aggregate metrics
    success_rate = np.mean(success) * 100
    avg_reward = np.mean(rewards)
    avg_length = np.mean(episode_lengths)
    avg_distance = np.mean(distances)
    avg_time = np.mean(times)
    
    # Print summary
    if verbose or num_episodes > 1:
        print("\nEvaluation Summary:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Episode Length: {avg_length:.1f}")
        print(f"  Average Distance: {avg_distance:.4f}m")
        print(f"  Average Episode Time: {avg_time:.2f}s")
    
    # Return results as dictionary
    results = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_length,
        'avg_distance': avg_distance,
        'avg_time': avg_time,
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'success': success,
        'distances': distances,
        'times': times
    }
    
    return results

def evaluate_model_directml(model_path, num_episodes=10, visualize=True, verbose=False, max_steps=1000, viz_speed=0.02):
    """
    Evaluate a model using DirectML.
    
    This is a compatibility function that redirects to the DirectML-specific implementation.
    
    Args:
        model_path: Path to the model to evaluate
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        max_steps: Maximum number of steps per episode before timing out
        viz_speed: Visualization speed (seconds per step)
        
    Returns:
        Dictionary of evaluation results
    """
    # Import and use the DirectML-specific implementation
    from src.dml import evaluate_model_directml as dml_evaluate_model
    return dml_evaluate_model(
        model_path=model_path,
        num_episodes=num_episodes,
        visualize=visualize,
        verbose=verbose,
        max_steps=max_steps,
        viz_speed=viz_speed
    )

def run_evaluation_sequence(model_path, viz_speed=0.02, max_steps=200):
    """
    Run an evaluation sequence with visualization for the model.
    
    This function loads a model and runs it against a set of predefined target
    positions to evaluate its performance in a more structured way.
    
    Args:
        model_path: Path to the model to evaluate
        viz_speed: Speed of visualization (seconds per step)
        max_steps: Maximum number of steps per position before timing out
        
    Returns:
        Dictionary of evaluation results
    """
    from src.core.utils import print_banner, ensure_model_file_exists
    import numpy as np
    
    # Ensure model file exists
    model_path = ensure_model_file_exists(model_path)
    
    # Print banner
    print_banner(f"Evaluation Sequence: {model_path}")
    
    # Create environment with visualization
    env = create_eval_environment(
        visualize=True,
        viz_speed=viz_speed,
        verbose=True
    )
    
    # Load the model
    try:
        from src.core.models import CustomPPO
        print(f"Attempting to load model from {model_path} using CustomPPO...")
        model = CustomPPO.load(model_path, env=env)
        print(f"Successfully loaded model from {model_path} using CustomPPO")
    except Exception as e:
        print(f"Error loading model with PPO: {e}")
        try:
            from src.core.models import CustomDirectMLModel
            print("Attempting to load model using DirectML...")
            model = CustomDirectMLModel(env.observation_space, env.action_space)
            success = model.load(model_path)
            if not success:
                print(f"Failed to load model as DirectML model")
                return None
            print(f"Successfully loaded model from {model_path} using CustomDirectMLModel")
        except Exception as e2:
            print(f"Error loading model as DirectML: {e2}")
            print(f"Could not load model from {model_path} using any available model class")
            return None
    
    # Define a set of test positions
    test_positions = [
        [0.5, 0.0, 0.5],  # Center
        [0.6, 0.2, 0.4],  # Front right
        [0.6, -0.2, 0.4], # Front left
        [0.4, 0.2, 0.6],  # Back right high
        [0.4, -0.2, 0.6]  # Back left high
    ]
    
    # Run evaluation on each position
    results = []
    
    for i, target_pos in enumerate(test_positions):
        print(f"\nTest position {i+1}: {target_pos}")
        
        # Reset environment with this target
        # Need to access the unwrapped env to set target
        try:
            # For VecEnv, get the underlying env
            raw_env = env.envs[0]
            # Set target manually
            raw_env.target_position = np.array(target_pos)
            raw_env._sample_target = lambda: np.array(target_pos)
        except Exception as e:
            print(f"Warning: Could not set target position: {e}")
            
        # Evaluate on this position
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        min_distance = float('inf')
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0]
            steps += 1
            
            # Track minimum distance
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
                if 'distance' in info:
                    min_distance = min(min_distance, info['distance'])
                    
                # Check for success
                if 'success' in info and info['success']:
                    print(f"  Success! Distance: {min_distance:.4f}m in {steps} steps")
                    break
        
        # Determine success status more robustly
        success = False
        if isinstance(info, list) and len(info) > 0:
            success = info[0].get('success', False)
        elif isinstance(info, dict):
            success = info.get('success', False)
            
        if not success:
            print(f"  Failed. Best distance: {min_distance:.4f}m in {steps} steps")
            
        # Store results
        result = {
            'target_pos': target_pos,
            'min_distance': min_distance,
            'steps': steps,
            'reward': total_reward,
            'success': success  # Use the success value we determined above
        }
        
        results.append(result)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\nEvaluation Sequence Summary:")
    print(f"  Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  Average Distance: {np.mean([r['min_distance'] for r in results]):.4f}m")
    print(f"  Average Steps: {np.mean([r['steps'] for r in results]):.1f}")
    
    return results 