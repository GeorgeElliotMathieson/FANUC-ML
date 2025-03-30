import os
import multiprocessing
import time
import torch
import numpy as np
import json
import glob
import re
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import the custom environment
from fanuc_env import FanucEnv

# --- Parameters to potentially tune (copy from ppo_tune.py for analysis) ---
# PARAMS_TO_TUNE = { ... } # Removed - no longer needed here

# --- Initial Default Hyperparameters ---
INITIAL_DEFAULT_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

# --- Central Best Parameters File ---
BEST_PARAMS_FILE = "best_params.json"

# --- Function to load parameters ---
def load_best_or_default_params():
    if os.path.exists(BEST_PARAMS_FILE):
        print(f"Loading best parameters from {BEST_PARAMS_FILE}")
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                loaded_params = json.load(f)
            # Optional: Validate loaded params against expected keys?
            return loaded_params
        except Exception as e:
            print(f"Warning: Failed to load or parse {BEST_PARAMS_FILE}. Using initial defaults. Error: {e}")
            return INITIAL_DEFAULT_PARAMS.copy()
    else:
        print(f"No {BEST_PARAMS_FILE} found. Using initial defaults.")
        return INITIAL_DEFAULT_PARAMS.copy()

# --- Function to determine best parameters (Removed - logic moved to ppo_tune.py) ---
# def determine_best_params_from_results(results, initial_baseline):
#    ...

# --- Time Limit Callback ---
class TimeLimitCallback(BaseCallback):
    """
    A custom callback that stops training after a specific time duration.

    :param max_duration_seconds: Maximum training duration in seconds.
    :param verbose: Verbosity level.
    """
    def __init__(self, max_duration_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_duration_seconds = max_duration_seconds
        self.start_time = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"Starting training timer. Max duration: {self.max_duration_seconds} seconds.")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) False to stop training.
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration_seconds:
            if self.verbose > 0:
                print(f"Stopping training due to time limit ({elapsed_time:.2f}s > {self.max_duration_seconds}s)")
            return False
        return True

# Define a function to create the environment. This is needed for parallelisation.
def make_env(rank, seed=0):
    """Utility function for multiprocessed env creation.

    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Use a different seed for each environment
        env = FanucEnv(render_mode=None) # No rendering during parallel training
        # Optional: Seed the environment for reproducibility
        # env.reset(seed=seed + rank)
        return env
    # set_global_seeds(seed + rank) # Deprecated in SB3
    return _init

if __name__ == "__main__":
    # --- Argument Parser --- 
    parser = argparse.ArgumentParser(description="Train PPO model for FANUC environment.")
    parser.add_argument("-d", "--duration", type=int, default=180,
                        help="Maximum training duration in seconds (default: 180).")
    # Add test_only flag
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run the visual test on the latest model.")
    args = parser.parse_args()

    # --- Define paths and device early --- 
    log_dir = "./ppo_fanuc_logs/" # Main log directory
    save_path = os.path.join(log_dir, "ppo_fanuc_model") # Base path for model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Training Block (skipped if --test_only is True) ---
    if not args.test_only:
        # --- Load Best or Default Params ---
        current_params = load_best_or_default_params()

        print("Using the following parameters for training:")
        for key, val in current_params.items():
            print(f"  {key}: {val}")
        print("-" * 30)

        # --- Set Training Constants from loaded/default params ---
        # Use duration from command line argument
        training_duration_seconds = args.duration
        # training_duration_seconds = 180 # e.g., 3 minutes. Adjust as needed. - Replaced by argparse

        # log_dir = "./ppo_fanuc_logs/" # Defined earlier
        # save_path = os.path.join(log_dir, "ppo_fanuc_model") # Defined earlier
        os.makedirs(log_dir, exist_ok=True)
        # Checkpoint saving frequency (can still be separate from tuning params)
        save_freq = 50_000

        # --- Hyperparameters ---
        # Determine the number of CPUs available
        cpu_count = multiprocessing.cpu_count()
        # Use all available CPUs instead of leaving one free
        num_cpu = cpu_count # Number of parallel environments
        # num_cpu = max(1, cpu_count - 1) # Previous setting: leave one free
        # num_cpu = 4 # Or set manually

        # --- Environment Setup ---
        print(f"Using {num_cpu} parallel environments.")

        # Create the vectorized environment using SubprocVecEnv for true parallelism
        # If SubprocVecEnv causes issues (e.g., on Windows without proper freezing),
        # fallback to DummyVecEnv (runs environments sequentially in one process).
        vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
        # Use make_vec_env for simplicity when render_mode is the same for all
        train_env = make_vec_env(lambda: FanucEnv(render_mode=None), n_envs=num_cpu, vec_env_cls=vec_env_cls)
        print("Vectorized environment created.")

        # --- Agent and Training ---
        # Check if CUDA is available, otherwise use CPU
        # device = "cuda" if torch.cuda.is_available() else "cpu" # Defined earlier
        # print(f"Using device: {device}") # Printed earlier

        # Define the PPO model using current_params
        model = PPO(
            current_params.get("policy", "MlpPolicy"), # Use get with fallback for safety
            train_env,
            learning_rate=current_params["learning_rate"],
            n_steps=current_params["n_steps"],
            batch_size=current_params["batch_size"],
            n_epochs=current_params["n_epochs"],
            gamma=current_params["gamma"],
            gae_lambda=current_params["gae_lambda"],
            clip_range=current_params["clip_range"],
            ent_coef=current_params["ent_coef"],
            vf_coef=current_params["vf_coef"],
            max_grad_norm=current_params["max_grad_norm"],
            verbose=1, # Print training progress
            tensorboard_log=log_dir, # Log to the main training log dir
            device=device
            # policy_kwargs can be added if it's in current_params
            # policy_kwargs=current_params.get("policy_kwargs", None)
        )

        # --- Callbacks ---
        # Save a checkpoint periodically in the main log dir
        checkpoint_callback = CheckpointCallback(
            save_freq=max(save_freq // num_cpu, 1), # Adjust freq based on num envs
            save_path=log_dir,
            name_prefix="rl_model"
        )

        # Add the time limit callback
        time_limit_callback = TimeLimitCallback(max_duration_seconds=training_duration_seconds, verbose=1)

        # --- Train the agent ---
        # Update print statement to reflect duration
        print(f"Starting PPO training for {training_duration_seconds} seconds...")
        start_time = time.time() # Keep this for final duration calculation
        try:
            # Use float('inf') for timesteps and pass list of callbacks
            model.learn(
                total_timesteps=float('inf'), # Run indefinitely until callback stops it
                callback=[checkpoint_callback, time_limit_callback], # Pass both callbacks
                log_interval=1 # Log every update
            )
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            # --- Save the final model ---
            print("Training finished. Saving final model...")
            model.save(save_path)
            train_env.close() # Important to close the parallel environments
            end_time = time.time()
            print(f"Model saved to {save_path}.zip")
            print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
    # --- End of Training Block ---
    else: # If args.test_only is True
        print("\n--- Skipping Training: Running Visual Test Only ---")

    # --- Load and test the trained model (This section runs always) ---
    model_file = save_path + ".zip"
    print(f"\nLoading model from {model_file} for testing...")
    # Check if model file exists RIGHT BEFORE loading
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at {model_file}")
        print("Please ensure a model has been trained and saved (e.g., run without --test_only first).")
        exit()

    # Load the model
    loaded_model = PPO.load(save_path, device=device)

    # Create a single test environment with rendering
    print("Creating test environment with rendering...")
    test_env = FanucEnv(render_mode='human')

    print("Testing model for 3 episodes...")
    num_test_episodes = 3
    for episode in range(num_test_episodes):
        # Force stage 0 for testing
        obs, _ = test_env.reset(options={'force_curriculum_stage': 0})
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        print(f"--- Test Episode {episode + 1} ---")
        while not (terminated or truncated):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            # Rendering is handled by PyBullet GUI connection, step triggers update
            # test_env.render() # No need to call explicitly if using GUI
            total_reward += reward
            step += 1
            # Add a small delay to make visualisation smoother
            time.sleep(1./60.)

            if terminated or truncated:
                print(f"Episode finished after {step} steps.")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Final Distance: {info['distance']:.4f}")
                # Pause briefly between episodes
                time.sleep(1)
                break # Exit while loop

    print("Testing finished.")
    test_env.close()
    print("Script finished.") 