# curriculum_training_simple.py
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import multiprocessing
import sys
import traceback

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a robot arm with curriculum learning (simplified version)')
parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
parser.add_argument('--parallel', type=int, default=0, help='Number of parallel environments (0=auto)')
parser.add_argument('--steps', type=int, default=200000, help='Number of steps per curriculum stage')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
args = parser.parse_args()

# Enable debug mode
debug_mode = args.debug

# Determine optimal number of parallel environments
if args.parallel <= 0:
    # Use half of available CPU cores to avoid overloading
    optimal_envs = max(1, multiprocessing.cpu_count() // 2)
    print(f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {optimal_envs} parallel environments")
    n_parallel_envs = optimal_envs
else:
    n_parallel_envs = args.parallel
    print(f"Using {n_parallel_envs} parallel environments as specified")

# Device selection - simplified to just use CPU for reliability
device = torch.device("cpu")
print("Using CPU for training (most reliable option)")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

# Import environment
from rl_env import DomainRandomizedRLEnv

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Custom MLP policy configuration for better performance
policy_kwargs = dict(
    net_arch=[256, 256, 256],  # Three hidden layers of 256 units each
    activation_fn=nn.ReLU      # ReLU activation function
)

# Simple progress callback that shows training progress and current accuracy
class SimpleProgressCallback(BaseCallback):
    def __init__(self, eval_env=None, verbose=0):
        super(SimpleProgressCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_steps = 0
        self.best_accuracy = float('inf')
        self.last_eval_step = 0
        
    def _on_step(self):
        # Print progress every 100 steps
        if self.n_calls % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            steps_per_second = self.n_calls / elapsed_time if elapsed_time > 0 else 0
            
            # Print progress
            print(f"Step {self.n_calls} | Speed: {steps_per_second:.1f} steps/s", flush=True)
        
        # Quick accuracy check every 1000 steps
        if self.n_calls % 1000 == 0 and self.eval_env is not None and self.n_calls > 0:
            try:
                print(f"\nStarting accuracy check at step {self.n_calls}...", flush=True)
                
                # Use the current policy to run a single episode with timeout protection
                start_time = time.time()
                max_eval_time = 30  # Maximum 30 seconds for evaluation
                
                obs = self.eval_env.reset()
                if debug_mode:
                    print(f"Reset environment, initial observation shape: {obs.shape}", flush=True)
                
                done = False
                total_reward = 0
                step_count = 0
                max_steps = 1000  # Maximum steps to prevent infinite loops
                
                while not done and step_count < max_steps:
                    # Check for timeout
                    if time.time() - start_time > max_eval_time:
                        print(f"Evaluation timed out after {max_eval_time} seconds", flush=True)
                        break
                        
                    # Get action from policy
                    if debug_mode:
                        print(f"Getting action for step {step_count}", flush=True)
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Take step in environment
                    if debug_mode:
                        print(f"Taking step {step_count} with action: {action}", flush=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    
                    total_reward += reward
                    step_count += 1
                    
                    if debug_mode and step_count % 10 == 0:
                        print(f"Completed {step_count} evaluation steps", flush=True)
                
                # Calculate accuracy (distance to target)
                accuracy = -total_reward  # Negative reward is the distance
                accuracy_cm = accuracy * 100  # Convert to cm
                
                # Update best accuracy
                if accuracy < self.best_accuracy:
                    self.best_accuracy = accuracy
                    improvement = "✓ NEW BEST!"
                else:
                    improvement = ""
                
                # Print accuracy
                print(f"\nACCURACY CHECK at step {self.n_calls}:")
                print(f"Current: {accuracy_cm:.2f} cm {improvement}")
                print(f"Best: {self.best_accuracy * 100:.2f} cm")
                print(f"Target: 6.00 cm ({min(100, 100 * 6.0 / accuracy_cm):.1f}% complete)")
                print(f"Evaluation steps: {step_count}")
                print("Continuing training...\n", flush=True)
                
                # Check if target accuracy is reached (6cm)
                if accuracy <= 0.06:
                    print(f"\nTARGET ACCURACY OF 6.00 cm REACHED at step {self.n_calls}!")
                    return False  # Stop training
                    
            except Exception as e:
                print(f"\nError during evaluation: {str(e)}")
                if debug_mode:
                    traceback.print_exc()
                print("Continuing training despite evaluation error...\n", flush=True)
                
        return True

# Environment setup helpers
def make_env(rank, seed=0, randomize=True, task_difficulty=0):
    """
    Helper function to create an environment with proper settings
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

# Simplified curriculum learning
def train_with_curriculum():
    # Training parameters
    n_curriculum_stages = 5
    timesteps_per_stage = args.steps
    
    # Start with easy tasks
    task_difficulty = 0.0
    
    # Create evaluation environment (no randomization for consistent evaluation)
    eval_env = DummyVecEnv([make_env(0, randomize=False, task_difficulty=1.0)])
    
    # Initialize the model with the first difficulty level
    env = SubprocVecEnv([make_env(i, task_difficulty=task_difficulty) for i in range(n_parallel_envs)])
    
    # Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
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
        
        # Create simple progress callback
        progress_callback = SimpleProgressCallback(eval_env=eval_env, verbose=1)
        
        # Train for this stage
        try:
            continue_training = model.learn(
                total_timesteps=timesteps_per_stage,
                callback=progress_callback,
                tb_log_name=f"PPO_stage_{stage}",
                reset_num_timesteps=False
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
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            if debug_mode:
                traceback.print_exc()
            print("Saving model before exiting...")
            model.save(f"./models/fanuc_model_stage_{stage}_error")
            print(f"Model saved to ./models/fanuc_model_stage_{stage}_error")
            sys.exit(1)
    
    # Final quick evaluation
    print("\nFinal evaluation...")
    try:
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 1000
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            step_count += 1
        
        # Calculate final accuracy
        final_accuracy = -total_reward
        final_accuracy_cm = final_accuracy * 100
        
        print(f"\n{'='*60}")
        print(f"FINAL ACCURACY: {final_accuracy_cm:.2f} cm")
        print(f"TARGET ACCURACY: 6.00 cm")
        print(f"EVALUATION STEPS: {step_count}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\nError during final evaluation: {str(e)}")
        if debug_mode:
            traceback.print_exc()
    
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
        print("To run again with different parameters:")
        print("python curriculum_training_simple.py --parallel 4 --steps 100000")
        print("To enable debug mode:")
        print("python curriculum_training_simple.py --parallel 1 --steps 10000 --debug")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        if debug_mode:
            traceback.print_exc()
        sys.exit(1) 