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
    print("  python fanuc_platform.py eval <model-path> [episodes] [options]")
    print("")
    print("Arguments:")
    print("  model-path  - Path to the trained model file")
    print("  episodes    - Number of episodes to evaluate (default: 10)")
    print("")
    print("Options:")
    print("  --no-gui     - Disable visualization")
    print("  --verbose    - Enable verbose output")
    print("  --speed      - Set visualization speed (default: 0.02)")
    print("")

def create_eval_environment(gui=True, viz_speed=0.02, verbose=False):
    """
    Create an environment for model evaluation.
    
    Args:
        gui: Whether to use visualization
        viz_speed: Visualization speed (seconds per step)
        verbose: Whether to print verbose output
        
    Returns:
        Evaluation environment
    """
    # Dynamic import to avoid circular imports
    from src.core.env import JointLimitEnforcingEnv
    
    # Handle import of RobotPositioningRevampedEnv with try-except
    try:
        from src.envs.robot_sim import RobotPositioningRevampedEnv
    except ImportError:
        if verbose:
            print("Warning: Could not import RobotPositioningRevampedEnv, falling back to RobotPositioningEnv")
        try:
            from src.envs.robot_sim import RobotPositioningEnv as RobotPositioningRevampedEnv
        except ImportError:
            raise ImportError("Could not import robot environment classes from src.envs.robot_sim")
    
    # Create the environment
    env = RobotPositioningRevampedEnv(
        gui=gui,
        viz_speed=viz_speed,
        verbose=verbose,
        training_mode=False
    )
    
    # Apply joint limit enforcement wrapper
    env = JointLimitEnforcingEnv(env)
    
    # Vectorize for compatibility with SB3
    vec_env = DummyVecEnv([lambda: env])
    
    return vec_env

def evaluate_model_wrapper(model_path, num_episodes=10, visualize=True, verbose=False):
    """
    Wrapper for model evaluation that loads the model and runs evaluation.
    
    Args:
        model_path: Path to the model to evaluate
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary of evaluation results
    """
    from src.core.utils import ensure_model_file_exists
    from src.core.models import CustomPPO
    
    # Ensure model file exists and has correct extension
    model_path = ensure_model_file_exists(model_path)
    
    # Create environment for evaluation
    env = create_eval_environment(
        gui=visualize,
        viz_speed=0.02 if visualize else 0.0,
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
        if verbose:
            print(f"Loading model from {model_path}")
        model = CustomPPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try DirectML model as fallback
        try:
            from src.core.models import CustomDirectMLModel
            if verbose:
                print("Trying DirectML model...")
            model = CustomDirectMLModel(env.observation_space, env.action_space)
            model.load(model_path)
        except Exception as e2:
            print(f"Error loading model as DirectML model: {e2}")
            return None
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        env=env,
        num_episodes=num_episodes,
        visualize=visualize,
        verbose=verbose
    )
    
    return results

def evaluate_model(model, env, num_episodes=10, visualize=True, verbose=False):
    """
    Evaluate a model on the environment.
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        
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
            
            # Check for success and distance
            if isinstance(info, list) and len(info) > 0:
                info = info[0]  # Get info from first environment
                if 'success' in info:
                    episode_success = info['success']
                if 'distance' in info:
                    episode_distance = min(episode_distance, info['distance'])
                    
                # Print step information if verbose
                if verbose and episode_length % 10 == 0:
                    print(f"  Step {episode_length}: reward={reward[0]:.3f}, "
                          f"distance={info.get('distance', None):.4f}m")
            
            # Break early if episode times out
            if episode_length >= 1000:  # Timeout
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

def evaluate_model_directml(model_path, num_episodes=10, visualize=True, verbose=False):
    """
    Evaluate a model using DirectML.
    
    This is a compatibility function for the DirectML-specific implementation.
    
    Args:
        model_path: Path to the model to evaluate
        num_episodes: Number of evaluation episodes
        visualize: Whether to use visualization
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary of evaluation results
    """
    # This is a wrapper that uses the main evaluation function
    return evaluate_model_wrapper(
        model_path=model_path,
        num_episodes=num_episodes,
        visualize=visualize,
        verbose=verbose
    )

def run_evaluation_sequence(model_path, viz_speed=0.02):
    """
    Run an evaluation sequence with visualization for the model.
    
    This function loads a model and runs it against a set of predefined target
    positions to evaluate its performance in a more structured way.
    
    Args:
        model_path: Path to the model to evaluate
        viz_speed: Speed of visualization (seconds per step)
        
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
    env = create_eval_environment(gui=True, viz_speed=viz_speed, verbose=True)
    
    # Load the model
    try:
        from src.core.models import CustomPPO
        model = CustomPPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model with PPO: {e}")
        try:
            from src.core.models import CustomDirectMLModel
            print("Trying DirectML model...")
            model = CustomDirectMLModel(env.observation_space, env.action_space)
            model.load(model_path)
        except Exception as e2:
            print(f"Error loading model as DirectML: {e2}")
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
            raw_env.target_pos = np.array(target_pos)
            raw_env._sample_target = lambda: np.array(target_pos)
        except Exception as e:
            print(f"Warning: Could not set target position: {e}")
            
        # Evaluate on this position
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        min_distance = float('inf')
        
        while not done and steps < 200:
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
        
        if not done or (isinstance(info, list) and len(info) > 0 and not info[0].get('success', False)):
            print(f"  Failed. Best distance: {min_distance:.4f}m in {steps} steps")
            
        # Store results
        result = {
            'target_pos': target_pos,
            'min_distance': min_distance,
            'steps': steps,
            'reward': total_reward,
            'success': done and (isinstance(info, list) and len(info) > 0 and info[0].get('success', False))
        }
        results.append(result)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\nEvaluation Sequence Summary:")
    print(f"  Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  Average Distance: {np.mean([r['min_distance'] for r in results]):.4f}m")
    print(f"  Average Steps: {np.mean([r['steps'] for r in results]):.1f}")
    
    return results 