# curriculum_training.py
import os
import numpy as np
import torch
import platform
import torch.nn as nn
import argparse
import time
import multiprocessing
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a robot arm with curriculum learning')
parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
parser.add_argument('--verbose', action='store_true', help='Show detailed accuracy metrics at each step')
parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer steps')
parser.add_argument('--parallel', type=int, default=0, help='Number of parallel environments (0=auto)')
args = parser.parse_args()

# Determine optimal number of parallel environments
if args.parallel <= 0:
    # Use all available CPU cores minus 1 (to keep system responsive)
    optimal_envs = max(1, multiprocessing.cpu_count() - 1)
    print(f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {optimal_envs} parallel environments")
    n_parallel_envs = optimal_envs
else:
    n_parallel_envs = args.parallel
    print(f"Using {n_parallel_envs} parallel environments as specified")

# Device selection
if args.cpu:
    print("Forcing CPU usage as requested")
    device = torch.device("cpu")
else:
    # Try to import DirectML for AMD GPU support on Windows
    try:
        import torch_directml
        directml_available = torch_directml.is_available()
        if directml_available:
            print("DirectML is available! Using AMD GPU acceleration.")
            dml_device = torch_directml.device()
            device = dml_device
        else:
            print("DirectML is not available. Falling back to CPU.")
            device = torch.device("cpu")
    except ImportError:
        print("torch_directml not found. Falling back to CUDA or CPU.")
        # Check if CUDA is available (for NVIDIA GPUs or ROCm on Linux)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rl_env import DomainRandomizedRLEnv

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Print device information
print(f"Using device: {device}")
if device == torch.device("cuda") and torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif 'directml' in str(device).lower():
    print("Using AMD GPU via DirectML")
    print("Note: If training is slow, try running with --cpu flag for potentially better performance")
elif device == torch.device("cpu"):
    print("Using CPU for training")

# Custom MLP policy configuration for better performance
policy_kwargs = dict(
    net_arch=[256, 256, 256],  # Three hidden layers of 256 units each
    activation_fn=nn.ReLU      # ReLU activation function
)

# Custom callback to display training progress
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_steps = 0
        self.last_print_time = time.time()
        
    def _on_step(self):
        current_time = time.time()
        
        # Print every 50 steps or at least every 5 seconds, whichever comes first
        time_since_last_print = current_time - self.last_print_time
        if self.n_calls % 50 == 0 or time_since_last_print >= 5.0:
            # Calculate overall speed
            elapsed_time = current_time - self.start_time
            overall_steps_per_second = self.n_calls / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate recent speed
            recent_elapsed = current_time - self.last_time
            steps_since_last = self.n_calls - self.last_steps
            
            if steps_since_last > 0 and recent_elapsed > 0:
                recent_steps_per_second = steps_since_last / recent_elapsed
                
                # Update last values
                self.last_time = current_time
                self.last_steps = self.n_calls
                self.last_print_time = current_time
                
                # Print progress with immediate flush to ensure output is visible
                progress_msg = f"Step {self.n_calls} | Recent: {recent_steps_per_second:.1f} steps/s | Avg: {overall_steps_per_second:.1f} steps/s"
                print(progress_msg, flush=True)
                
                # If we're in an evaluation, print a message to indicate training is still happening
                if self.n_calls % 1000 == 0 and self.n_calls > 0:
                    print("(If evaluation is running, please wait...)", flush=True)
        
        return True

# Custom callback to display accuracy metrics at each step
class AccuracyMonitorCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=500, verbose=1):
        super(AccuracyMonitorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_time = time.time()
        self.best_accuracy = float('inf')
        self.last_eval_step = 0
        
    def _on_step(self):
        # Skip if we've already evaluated at this step (prevents duplicate evaluations)
        if self.n_calls <= self.last_eval_step:
            return True
            
        if self.n_calls % self.eval_freq == 0:
            self.last_eval_step = self.n_calls
            
            # Print message before evaluation starts
            print(f"\nStarting quick evaluation at step {self.n_calls}...", flush=True)
            
            # Run quick evaluation with fewer episodes for speed
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=2,  # Use fewer episodes for faster evaluation
                return_episode_rewards=True,
            )
            
            # Calculate accuracy metrics
            distances = -np.array(episode_rewards)  # Convert rewards to distances
            mean_distance_m = np.mean(distances)
            mean_distance_cm = mean_distance_m * 100  # Convert to cm
            std_distance_cm = np.std(distances) * 100  # Standard deviation in cm
            min_distance_cm = np.min(distances) * 100  # Best accuracy in cm
            max_distance_cm = np.max(distances) * 100  # Worst accuracy in cm
            
            # Update best accuracy
            if mean_distance_m < self.best_accuracy:
                self.best_accuracy = mean_distance_m
                improvement = "✓ NEW BEST!"
            else:
                improvement = ""
            
            # Calculate evaluation time
            current_time = time.time()
            eval_time = current_time - self.last_eval_time
            self.last_eval_time = current_time
            
            # Print accuracy metrics with more prominence
            print(f"\n{'*'*60}")
            print(f"ACCURACY AT STEP {self.n_calls}:")
            print(f"CURRENT: {mean_distance_cm:.2f} cm {improvement}")
            print(f"BEST OVERALL: {self.best_accuracy * 100:.2f} cm")
            print(f"RANGE: {min_distance_cm:.2f} - {max_distance_cm:.2f} cm (±{std_distance_cm:.2f} cm)")
            print(f"TARGET: 6.00 cm ({min(100, 100 * 6.0 / mean_distance_cm):.1f}% complete)")
            print(f"{'*'*60}\n")
            print("Continuing training...", flush=True)
            
        return True

# Custom callback to stop training when target accuracy is reached
class TargetAccuracyCallback(BaseCallback):
    def __init__(self, eval_env, target_accuracy=0.06, eval_freq=5000, verbose=1):
        super(TargetAccuracyCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.target_accuracy = target_accuracy
        self.eval_freq = eval_freq
        self.best_mean_accuracy = float('inf')
        self.last_eval_step = 0
        
    def _on_step(self):
        # Skip if we've already evaluated at this step (prevents duplicate evaluations)
        if self.n_calls <= self.last_eval_step:
            return True
            
        if self.n_calls % self.eval_freq == 0:
            self.last_eval_step = self.n_calls
            
            # Print message before evaluation starts
            print(f"\nStarting formal evaluation at step {self.n_calls}...", flush=True)
            
            # Run evaluation with fewer episodes for speed
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=3,  # Reduced from 5 for faster evaluation
                return_episode_rewards=True,
            )
            
            # Calculate mean accuracy (distance to target)
            # We're using negative rewards, so we need to negate them to get the distance
            mean_accuracy = -np.mean(episode_rewards)
            mean_accuracy_cm = mean_accuracy * 100  # Convert to cm
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"FORMAL EVALUATION AT STEP {self.n_calls}")
                print(f"ACCURACY: {mean_accuracy_cm:.2f} cm")
                print(f"TARGET: {self.target_accuracy * 100:.2f} cm")
                print(f"{'='*60}\n")
            
            # Update best accuracy
            if mean_accuracy < self.best_mean_accuracy:
                self.best_mean_accuracy = mean_accuracy
                if self.verbose > 0:
                    print(f"NEW BEST ACCURACY: {self.best_mean_accuracy * 100:.2f} cm")
                
            # Check if target accuracy is reached
            if mean_accuracy <= self.target_accuracy:
                if self.verbose > 0:
                    print(f"\n{'!'*60}")
                    print(f"TARGET ACCURACY OF {self.target_accuracy * 100:.2f} cm REACHED!")
                    print(f"STOPPING TRAINING AT STEP {self.n_calls}")
                    print(f"{'!'*60}\n")
                return False  # Stop training
                
            print("Continuing training...", flush=True)
                
        return True  # Continue training

# Environment setup helpers
def make_env(rank, seed=0, randomize=True, task_difficulty=0):
    """
    Helper function to create an environment with proper settings
    
    Args:
        rank: process rank for multi-processing
        seed: random seed
        randomize: whether to use domain randomization
        task_difficulty: difficulty level (0-1) for curriculum learning
    """
    def _init():
        env = DomainRandomizedRLEnv(render=False, randomize=randomize)
        # Adjust workspace size based on difficulty
        env.workspace_size = 0.2 + 0.5 * task_difficulty
        # Set random seed
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Curriculum learning
def train_with_curriculum():
    # Determine training parameters based on quick mode
    if args.quick:
        print("Running in quick test mode with reduced steps")
        n_curriculum_stages = 2
        timesteps_per_stage = 10000
    else:
        n_curriculum_stages = 5
        timesteps_per_stage = 200000
    
    # Start with easy tasks
    task_difficulty = 0.0
    
    # Initialize the model with the first difficulty level
    env = SubprocVecEnv([make_env(i, task_difficulty=task_difficulty) for i in range(n_parallel_envs)])
    
    # Create evaluation environment (no randomization for consistent evaluation)
    eval_env = DummyVecEnv([make_env(0, randomize=False, task_difficulty=1.0)])
    
    # Initialize the model with GPU support and optimized MLP policy
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # Reduce SB3's built-in verbosity to avoid cluttering output
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=1024 if args.quick else 2048,  # Smaller batch for quick mode
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device=device,
        policy_kwargs=policy_kwargs
    )
    
    # Iterate through curriculum stages
    for stage in range(n_curriculum_stages):
        # Update difficulty
        task_difficulty = stage / (n_curriculum_stages - 1)
        print(f"\n{'='*60}")
        print(f"TRAINING ON DIFFICULTY LEVEL: {task_difficulty:.2f}")
        print(f"{'='*60}")
        
        # Create new environments with the updated difficulty
        env.close()
        env = SubprocVecEnv([make_env(i, task_difficulty=task_difficulty) for i in range(n_parallel_envs)])
        
        # Update the environment reference in the model
        model.set_env(env)
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000 if args.quick else 10000, 
            save_path=f"./models/stage_{stage}/",
            name_prefix="fanuc_model",
            verbose=0  # Reduce verbosity
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/stage_{stage}/best/",
            log_path=f"./logs/stage_{stage}/",
            eval_freq=5000 if args.quick else 10000,
            deterministic=True,
            render=False,
            verbose=0  # Reduce verbosity
        )
        
        # Add target accuracy callback
        target_accuracy_callback = TargetAccuracyCallback(
            eval_env=eval_env,
            target_accuracy=0.06,  # 6cm accuracy
            eval_freq=5000 if args.quick else 10000,
            verbose=1
        )
        
        # Add callbacks based on verbosity
        callbacks = [checkpoint_callback, eval_callback, target_accuracy_callback]
        
        # Always add progress callback to show training is happening
        progress_callback = ProgressCallback(verbose=1)
        callbacks.append(progress_callback)
        
        # Add accuracy monitor callback if verbose mode is enabled
        if args.verbose:
            accuracy_monitor = AccuracyMonitorCallback(
                eval_env=eval_env,
                eval_freq=500 if args.quick else 1000,  # Check accuracy more frequently
                verbose=1
            )
            callbacks.append(accuracy_monitor)
        
        # Train for this stage
        try:
            continue_training = model.learn(
                total_timesteps=timesteps_per_stage,
                callback=callbacks,
                tb_log_name=f"PPO_stage_{stage}",
                reset_num_timesteps=False  # Don't reset step counter between stages
            )
            
            # Save the model for this curriculum stage
            model.save(f"./models/fanuc_model_stage_{stage}")
            
            # Check if target accuracy was reached
            if not continue_training:
                print(f"\nTarget accuracy reached at stage {stage}. Stopping curriculum training.")
                break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model...")
            model.save(f"./models/fanuc_model_stage_{stage}_interrupted")
            print(f"Model saved to ./models/fanuc_model_stage_{stage}_interrupted")
            sys.exit(0)
    
    # Final evaluation
    print("\nFinal evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Calculate final accuracy (distance to target)
    mean_accuracy = -mean_reward  # Negative reward is the distance
    mean_accuracy_cm = mean_accuracy * 100  # Convert to cm
    std_accuracy_cm = std_reward * 100  # Convert to cm
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {mean_accuracy_cm:.2f} cm ± {std_accuracy_cm:.2f} cm")
    print(f"TARGET ACCURACY: 6.00 cm")
    print(f"{'='*60}\n")
    
    # Save the final model
    model.save("./models/fanuc_final_model")
    
    # Clean up
    env.close()
    eval_env.close()

if __name__ == "__main__":
    try:
        train_with_curriculum()
        
        # Print final message with instructions
        print("\nTraining completed!")
        print("To run again with CPU (which might be faster for MLP policies), use:")
        print("python curriculum_training.py --cpu")
        print("\nTo run with detailed accuracy metrics, use:")
        print("python curriculum_training.py --verbose")
        print("\nTo run a quick test to verify everything works:")
        print("python curriculum_training.py --quick --verbose")
        print("\nTo specify number of parallel environments:")
        print("python curriculum_training.py --parallel 8")
        print("\nTo use multiple options:")
        print("python curriculum_training.py --cpu --verbose --parallel 8")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)