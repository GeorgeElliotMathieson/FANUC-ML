import os
import multiprocessing
import time
import torch
import numpy as np
import json
import glob
import re
import argparse
import sys
import logging # Import logging
from typing import Type, Union

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Add collections for deque
import collections

# Import the custom environment using relative import
from .fanuc_env import FanucEnv

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

# Define paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
SAVE_PATH = os.path.join(LOG_DIR, "ppo_fanuc_model")

# --- Function to load parameters ---
def load_best_or_default_params():
    # Use updated BEST_PARAMS_FILE path
    if os.path.exists(BEST_PARAMS_FILE):
        logging.info(f"Loading best parameters from {BEST_PARAMS_FILE}")
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                loaded_params = json.load(f)
            return loaded_params
        except Exception as e:
            logging.warning(f"Failed to load {BEST_PARAMS_FILE}. Using defaults. Error: {e}")
            return INITIAL_DEFAULT_PARAMS.copy()
    else:
        logging.info(f"No {BEST_PARAMS_FILE} found. Using initial defaults.")
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
        self.start_time: float = 0.0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_time = time.time()
        if self.verbose > 0:
            # Keep print here as it's within a callback's verbose logic
            print(f"Starting training timer. Max duration: {self.max_duration_seconds} seconds.")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) False to stop training.
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration_seconds:
            if self.verbose > 0:
                 # Keep print here as it's within a callback's verbose logic
                 print(f"Stopping training due to time limit ({elapsed_time:.2f}s > {self.max_duration_seconds}s)")
            return False
        return True

# Define a function to create the environment (NOT NEEDED for make_vec_env)
# def make_env(rank, seed=0):
#    ...

# --- Custom Callback for Logging Environment Info ---
class TrainingMonitorCallback(BaseCallback):
    """
    A callback to log custom environment information during training.
    Monitors success rate, collision rates, and curriculum radius.
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        # It's better to rely on the model's ep_info_buffer than maintain separate deques here

    def _on_step(self) -> bool:
        # Log custom metrics periodically or when an episode finishes
        # Relying on SB3's built-in logging interval for ep_info_buffer is cleaner

        # Log current max target radius (if available in the first env's info)
        if self.n_calls % self.check_freq == 0: # Check periodically
            if self.training_env is not None and isinstance(self.training_env, VecEnv):
                # Attempt to get info from the buffer if available
                infos = getattr(self.training_env, 'buf_infos', None)
                if infos is not None and len(infos) > 0:
                    first_env_info = infos[0]
                    if "current_max_target_radius" in first_env_info:
                        # Use self.logger provided by BaseCallback
                        self.logger.record("custom/current_max_target_radius", first_env_info["current_max_target_radius"])

        # Log success/collision rate using the episode info buffer (populated by SB3 wrapper)
        # This buffer usually contains info from the last 100 episodes
        # Add check for None before accessing the buffer
        if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
            # Use 'True' and 'False' counts for rates
            # Add default value [] if key is missing in an episode's info
            successes = [ep_info.get("is_success", False) for ep_info in self.model.ep_info_buffer]
            collisions = [ep_info.get("collision", False) for ep_info in self.model.ep_info_buffer]
            obstacle_collisions = [ep_info.get("obstacle_collision", False) for ep_info in self.model.ep_info_buffer]

            # No need to check lists again, they are guaranteed to be populated if buffer is not empty
            # if successes: # Avoid division by zero
            success_rate = sum(1 for s in successes if s) / len(successes)
            self.logger.record("rollout/success_rate", success_rate)
            # if collisions:
            collision_rate = sum(1 for c in collisions if c) / len(collisions)
            self.logger.record("rollout/collision_rate", collision_rate)
            # if obstacle_collisions:
            obstacle_collision_rate = sum(1 for oc in obstacle_collisions if oc) / len(obstacle_collisions)
            self.logger.record("rollout/obstacle_collision_rate", obstacle_collision_rate)

        return True # Continue training

if __name__ == "__main__":
    # --- Configure Logging FIRST --- 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # --- Argument Parser --- 
    parser = argparse.ArgumentParser(description="Train PPO model for FANUC environment.")
    parser.add_argument("-d", "--duration", type=int, default=3, # Default in minutes
                        help="Maximum training duration in minutes (default: 3).")
    args = parser.parse_args()

    # --- Define paths and device early --- 
    # Paths are defined globally. Device setup needed here.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Load Best or Default Params --- 
    # Uses global BEST_PARAMS_FILE path
    current_params = load_best_or_default_params()

    # Use logger to print parameters
    logger.info("Using the following parameters for training:")
    param_string = json.dumps(current_params, indent=4)
    for line in param_string.split('\n'):
        logger.info(f"  {line}")
    # logger.info("-" * 30) # Divider less necessary with logger timestamps

    # --- Set Training Constants --- 
    training_duration_seconds = args.duration * 60
    logger.info(f"Requested training duration: {args.duration} mins ({training_duration_seconds} s).")
    os.makedirs(LOG_DIR, exist_ok=True)
    save_freq = 50_000

    # --- Determine Number of CPUs --- 
    # This needs to be inside __main__ to correctly use multiprocessing
    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count # Use all available cores
    logger.info(f"Using {num_cpu} parallel environments.")

    # --- Environment Setup --- 
    loaded_angle_bonus = current_params.get("angle_bonus_factor", 5.0)
    logger.info(f"Using angle_bonus_factor: {loaded_angle_bonus}")
    vec_env_cls: Union[Type[SubprocVecEnv], Type[DummyVecEnv]] = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
    train_env = make_vec_env(
        lambda: FanucEnv(render_mode=None, angle_bonus_factor=loaded_angle_bonus),
        n_envs=num_cpu,
        vec_env_cls=vec_env_cls
    )
    logger.info("Vectorized environment created.")

    # --- Agent --- 
    policy_kwargs = current_params.get("policy_kwargs", None)
    model = PPO(
        current_params.get("policy", "MlpPolicy"),
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
        tensorboard_log=LOG_DIR, # Use updated LOG_DIR path
        device=device,
        policy_kwargs=policy_kwargs # Pass loaded policy_kwargs
    )

    # --- Callbacks --- 
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // num_cpu, 1),
        save_path=LOG_DIR,
        name_prefix="rl_model"
    )
    time_limit_callback = TimeLimitCallback(max_duration_seconds=training_duration_seconds, verbose=1)
    monitor_callback = TrainingMonitorCallback(check_freq=num_cpu * 100)

    # --- Train the agent --- 
    # Update print statement to reflect duration
    logger.info(f"Starting PPO training for {args.duration} minutes ({training_duration_seconds} seconds)...")
    start_time = time.time() # Keep this for final duration calculation
    try:
        # Use sys.maxsize for timesteps and pass list of callbacks
        model.learn(
            total_timesteps=sys.maxsize, # Run indefinitely until callback stops it
            callback=[checkpoint_callback, time_limit_callback, monitor_callback], # Pass all callbacks
            log_interval=1 # Log every update
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.") # Use logger
    finally:
        # --- Save the final model ---
        logger.info("Training finished. Saving final model...") # Use logger
        model.save(SAVE_PATH) # Use updated SAVE_PATH
        train_env.close() # Important to close the parallel environments
        end_time = time.time()
        logger.info(f"Model saved to {SAVE_PATH}.zip") # Use logger
        logger.info(f"Total training time: {(end_time - start_time)/60:.2f} minutes") # Use logger
    # --- End of Training Block ---
    
    logger.info("Training script finished.") 