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
import logging
from typing import Type, Union
import datetime
import shutil
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import collections

# Custom environment
from .fanuc_env import FanucEnv

# Initial default parameters
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

# Project paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")
BASE_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
ARCHIVE_LOG_DIR = os.path.join(PROJECT_ROOT, "archive", "archived_ppo_logs")

def load_best_or_default_params():
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

# Time limit callback
class TimeLimitCallback(BaseCallback):
    def __init__(self, max_duration_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_duration_seconds = max_duration_seconds
        self.start_time: float = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"Starting training timer. Max duration: {self.max_duration_seconds} seconds.")

    def _on_step(self) -> bool:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration_seconds:
            if self.verbose > 0:
                print(f"Stopping training due to time limit ({elapsed_time:.2f}s > {self.max_duration_seconds}s)")
            return False
        return True

# Training monitor callback
class TrainingMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.training_env is not None and isinstance(self.training_env, VecEnv):
                infos = getattr(self.training_env, 'buf_infos', None)
                if infos is not None and len(infos) > 0:
                    first_env_info = infos[0]
                    if "current_min_target_radius" in first_env_info:
                        self.logger.record("custom/current_min_target_radius", first_env_info["current_min_target_radius"])

        if self.model.ep_info_buffer is not None and len(self.model.ep_info_buffer) > 0:
            successes = [ep_info.get("is_success", False) for ep_info in self.model.ep_info_buffer]
            collisions = [ep_info.get("collision", False) for ep_info in self.model.ep_info_buffer]
            obstacle_collisions = [ep_info.get("obstacle_collision", False) for ep_info in self.model.ep_info_buffer]

            success_rate = sum(1 for s in successes if s) / len(successes)
            self.logger.record("rollout/success_rate", success_rate)
            collision_rate = sum(1 for c in collisions if c) / len(collisions)
            self.logger.record("rollout/collision_rate", collision_rate)
            obstacle_collision_rate = sum(1 for oc in obstacle_collisions if oc) / len(obstacle_collisions)
            self.logger.record("rollout/obstacle_collision_rate", obstacle_collision_rate)

        return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Argument parser
    parser = argparse.ArgumentParser(description="Train PPO model for FANUC environment.")
    parser.add_argument("-d", "--duration", type=int, default=3,
                        help="Maximum training duration in minutes (default: 3).")
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load parameters
    current_params = load_best_or_default_params()

    # Run-specific directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    current_log_dir = os.path.join(BASE_LOG_DIR, run_name)
    os.makedirs(current_log_dir, exist_ok=True)
    logger.info(f"Saving logs and models for this run to: {current_log_dir}")

    # Final model save path
    final_model_save_path = os.path.join(current_log_dir, "final_model.zip")

    # Log parameters
    logger.info("Using the following parameters for training:")
    param_string = json.dumps(current_params, indent=4)
    for line in param_string.split('\n'):
        logger.info(f"  {line}")

    # Training duration
    training_duration_seconds = args.duration * 60
    logger.info(f"Requested training duration: {args.duration} mins ({training_duration_seconds} s).")
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    save_freq = 50_000

    # CPU count for parallel environments
    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count
    logger.info(f"Using {num_cpu} parallel environments.")

    # Environment setup
    loaded_angle_bonus = current_params.get("angle_bonus_factor", 5.0)
    logger.info(f"Using angle_bonus_factor: {loaded_angle_bonus}")
    vec_env_cls: Union[Type[SubprocVecEnv], Type[DummyVecEnv]] = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
    train_env = make_vec_env(
        lambda: FanucEnv(render_mode=None, angle_bonus_factor=loaded_angle_bonus),
        n_envs=num_cpu,
        vec_env_cls=vec_env_cls
    )
    logger.info("Vectorized environment created.")

    # PPO model setup
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
        verbose=1,
        tensorboard_log=current_log_dir,
        device=device,
        policy_kwargs=policy_kwargs
    )

    # Training callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // num_cpu, 1),
        save_path=current_log_dir,
        name_prefix="rl_model"
    )
    time_limit_callback = TimeLimitCallback(max_duration_seconds=training_duration_seconds, verbose=1)
    monitor_callback = TrainingMonitorCallback(check_freq=num_cpu * 100)

    # Start training
    logger.info(f"Starting PPO training for {args.duration} minutes ({training_duration_seconds} seconds)...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=sys.maxsize,
            callback=[checkpoint_callback, time_limit_callback, monitor_callback],
            log_interval=1
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        logger.info("Training finished. Saving final model...")
        model.save(final_model_save_path)
        train_env.close()
        end_time = time.time()
        logger.info(f"Model saved to {final_model_save_path}")
        logger.info(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

    # Archive old runs
    try:
        logger.info("Checking for old runs to archive...")
        keep_latest = 5
        os.makedirs(ARCHIVE_LOG_DIR, exist_ok=True)

        all_runs = [os.path.join(BASE_LOG_DIR, d) for d in os.listdir(BASE_LOG_DIR)
                    if os.path.isdir(os.path.join(BASE_LOG_DIR, d)) and d.startswith("run_")]

        if len(all_runs) > keep_latest:
            all_runs.sort(key=os.path.getctime)
            runs_to_archive = all_runs[:-keep_latest]
            logger.info(f"Found {len(all_runs)} runs. Archiving {len(runs_to_archive)} oldest runs to {ARCHIVE_LOG_DIR}")

            for run_path in runs_to_archive:
                run_dir_name = os.path.basename(run_path)
                destination_path = os.path.join(ARCHIVE_LOG_DIR, run_dir_name)
                
                if os.path.abspath(run_path) == os.path.abspath(ARCHIVE_LOG_DIR):
                    continue
                if os.path.abspath(run_path) == os.path.abspath(destination_path):
                    logger.warning(f"Skipping move: Source and destination are the same ({run_path}).")
                    continue

                try:
                    logger.info(f"  Archiving {run_dir_name}...")
                    shutil.move(run_path, destination_path)
                    logger.info(f"  Successfully moved {run_dir_name} to archive.")
                except Exception as e:
                    logger.error(f"  Failed to archive {run_dir_name}: {e}")
        else:
            logger.info(f"Found {len(all_runs)} runs. No archiving needed (keeping latest {keep_latest}).")

    except Exception as e:
        logger.error(f"An error occurred during the archiving process: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Training script finished.") 