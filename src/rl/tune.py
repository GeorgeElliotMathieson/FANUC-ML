import os
import multiprocessing
import time
import torch
import numpy as np
import json
import argparse
import math
import logging
import traceback
from typing import Dict, Any
import random
import collections
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import optuna
from optuna.exceptions import TrialPruned

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from .fanuc_env import FanucEnv

# Tuning configuration
TUNE_TIMESTEPS = 2_000_000
NUM_EVAL_EPISODES = 30
EVAL_FREQ = 125_000

# Project paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OPTUNA_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "optuna_study")
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")

# Default policy
DEFAULT_POLICY = "MlpPolicy"

# Default storage
DEFAULT_OPTUNA_DB_NAME = "default_fanuc_study.db"
DEFAULT_STORAGE_URL = f"sqlite:///{os.path.join(OPTUNA_LOG_DIR, DEFAULT_OPTUNA_DB_NAME)}"

# Ensure storage directory exists
try:
    os.makedirs(OPTUNA_LOG_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create Optuna study directory {OPTUNA_LOG_DIR}: {e}")

def run_experiment(trial: optuna.Trial, params: Dict[str, Any], trial_number: int, seed: int | None = None) -> float:
    """Run a single training experiment.

    Args:
        trial: Optuna trial object
        params: Hyperparameters dictionary
        trial_number: Trial identifier
        seed: Random seed

    Returns:
        Mean reward
    """
    run_name = f"trial_{trial_number}"
    logging.info(f"\nStarting Optuna {run_name}")
    logging.info(f"Parameters: {params}")

    train_env = None
    eval_env = None
    mean_reward = -float('inf')
    std_reward = float('inf')
    training_time = 0.0

    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count

    try:
        logging.info(f"Using {num_cpu} parallel environments for training.")
        vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
        angle_bonus = params.get("angle_bonus_factor", 5.0)
        train_env = make_vec_env(
            lambda: FanucEnv(render_mode=None, angle_bonus_factor=angle_bonus, start_with_obstacles=False),
            n_envs=num_cpu,
            vec_env_cls=vec_env_cls
        )
        eval_env = FanucEnv(render_mode=None, angle_bonus_factor=angle_bonus, start_with_obstacles=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trial_log_dir = os.path.join(OPTUNA_LOG_DIR, run_name)
        os.makedirs(trial_log_dir, exist_ok=True)

        policy = params.get("policy", DEFAULT_POLICY)
        policy_kwargs = params.get("policy_kwargs", None)

        model = PPO(
            policy=policy,
            env=train_env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            verbose=0,
            tensorboard_log=trial_log_dir,
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed
        )

        class TrialCallback(BaseCallback):
            def __init__(self, eval_env: FanucEnv, optuna_trial: optuna.Trial, eval_freq: int, n_eval_episodes: int, verbose: int = 0):
                super().__init__(verbose)
                self.eval_env = eval_env
                self.trial = optuna_trial
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes
                self.eval_count = 0

            def _on_step(self) -> bool:
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    self.eval_count += 1
                    step_num = self.num_timesteps
                    logging.info(f"  Trial {self.trial.number}, Step {step_num}: Running intermediate evaluation...")
                    _mean_reward, _std_reward = evaluate_policy(self.model, self.eval_env,
                                                              n_eval_episodes=self.n_eval_episodes,
                                                              deterministic=True)
                    mean_reward = float(_mean_reward)
                    std_reward = float(_std_reward)

                    logging.info(f"  Intermediate Eval Results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
                    self.trial.report(mean_reward, step=step_num)

                    if self.trial.should_prune():
                        logging.info(f"  Pruning trial {self.trial.number} at step {step_num} based on intermediate result.")
                        raise TrialPruned(f"Pruned at step {step_num}")
                return True

        trial_callback = TrialCallback(eval_env, trial, EVAL_FREQ, NUM_EVAL_EPISODES)

        logging.info(f"Training for {TUNE_TIMESTEPS} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=TUNE_TIMESTEPS,
            log_interval=1000,
            callback=trial_callback
        )
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds.")

        logging.info(f"Evaluating final model over {NUM_EVAL_EPISODES} episodes...")
        _mean_reward, _std_reward = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVAL_EPISODES, deterministic=True)
        mean_reward = float(_mean_reward)
        std_reward = float(_std_reward)

        logging.info(f"Final Evaluation results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("duration_seconds", training_time)

    except TrialPruned as e:
         logging.info(f"Trial {trial_number} pruned: {e}")
         raise
    except Exception as e:
        logging.error(f"Error during Optuna trial {trial_number} ({run_name}): {e}")
        logging.error(traceback.format_exc())
        raise TrialPruned(f"Trial failed due to error: {e}")
    finally:
        if train_env is not None:
            try:
                train_env.close()
            except Exception as e:
                logging.warning(f"Error closing train_env: {e}")
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception as e:
                 logging.warning(f"Error closing eval_env: {e}")
        logging.info(f"Finished Optuna {run_name}")

    return mean_reward

def objective(trial: optuna.Trial, seed: int | None = None) -> float:
    """Define objective for Optuna optimisation.

    Args:
        trial: Optuna trial object
        seed: Random seed

    Returns:
        Mean reward
    """
    params = {
        "policy": trial.suggest_categorical("policy", ["MlpPolicy"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.99, 0.999, log=False),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "angle_bonus_factor": trial.suggest_float("angle_bonus_factor", 1.0, 15.0, log=False),
    }

    if params["batch_size"] > params["n_steps"]:
         logging.info(f"Pruning trial {trial.number}: batch_size ({params['batch_size']}) > n_steps ({params['n_steps']})")
         raise TrialPruned("batch_size > n_steps")

    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large"])
    if net_arch_size == "small":
        params["policy_kwargs"] = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    elif net_arch_size == "medium":
        params["policy_kwargs"] = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else:
        params["policy_kwargs"] = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    mean_reward = run_experiment(trial, params, trial.number, seed=seed)
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO hyperparameter tuning using Optuna for Fanuc Env.")
    parser.add_argument("-d", "--duration", type=int, default=12,
                        help="Maximum tuning duration (minutes)")
    parser.add_argument("--study_name", type=str, default="ppo_fanuc_default_study", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=DEFAULT_STORAGE_URL,
                        help=f"Optuna storage URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trials")
    args = parser.parse_args()

    logger.info(f"Starting Optuna study '{args.study_name}' for a maximum duration of {args.duration} minutes.")
    logger.info(f"Using storage: {args.storage}")
    logger.info(f"Each trial trains for {TUNE_TIMESTEPS} timesteps and evaluates over {NUM_EVAL_EPISODES} episodes.")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0, interval_steps=1
        )
    )

    if os.path.exists(BEST_PARAMS_FILE):
        logger.info(f"Attempting to warm start study with parameters from {BEST_PARAMS_FILE}")
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                warm_start_params = json.load(f)
            
            required_keys = ["learning_rate", "n_steps", "batch_size", "n_epochs", 
                            "gamma", "gae_lambda", "clip_range", "ent_coef", 
                            "vf_coef", "max_grad_norm", "angle_bonus_factor", 
                            "net_arch_size"]
            
            valid_params = True
            params_for_enqueue = {}
            for key in required_keys:
                if key not in warm_start_params:
                    logger.warning(f"Warm start file missing key '{key}'. Skipping warm start.")
                    valid_params = False
                    break
                params_for_enqueue[key] = warm_start_params[key]

            if "policy" not in params_for_enqueue:
                 params_for_enqueue["policy"] = DEFAULT_POLICY 

            if valid_params:
                if "gamma" in params_for_enqueue and params_for_enqueue["gamma"] < 0.99:
                    logger.warning(f"Warm start gamma ({params_for_enqueue['gamma']}) is below the new minimum (0.99). Adjusting to 0.99 for enqueue.")
                    params_for_enqueue["gamma"] = 0.99

                study.enqueue_trial(params_for_enqueue)
                logger.info("Successfully enqueued warm start trial.")
            
        except Exception as e:
            logger.error(f"Failed to load or enqueue warm start parameters from {BEST_PARAMS_FILE}: {e}")
    else:
        logger.info(f"{BEST_PARAMS_FILE} not found. Starting study without warm start.")

    tuning_duration_seconds = args.duration * 60
    logger.info(f"Starting Optuna tuning for {args.duration} minutes ({tuning_duration_seconds} seconds)...")
    try:
        study.optimize(
            lambda trial: objective(trial, seed=args.seed),
            n_trials=None,
            timeout=tuning_duration_seconds
        )
    except KeyboardInterrupt:
        logger.warning("Optuna study interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}")
        logger.error(traceback.format_exc())

    logger.info("=============== Optuna Study Summary ===============")
    logger.info(f"Study Name: {study.study_name}")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        logger.info(f"Best trial finished with value (mean reward): {best_trial.value:.4f}")
        logger.info("Best hyperparameters found:")
        best_params_dict = {}
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
            best_params_dict[key] = value

        logger.info(f"Saving best parameters found to {BEST_PARAMS_FILE}...")
        try:
            if "policy" not in best_params_dict:
                 best_params_dict["policy"] = DEFAULT_POLICY
            if "policy_kwargs" in best_params_dict and "net_arch_size" not in best_params_dict:
                 arch_pi = best_params_dict["policy_kwargs"].get("net_arch", {}).get("pi", [])
                 if arch_pi == [64, 64]:
                     best_params_dict["net_arch_size"] = "small"
                 elif arch_pi == [128, 128]:
                     best_params_dict["net_arch_size"] = "medium"
                 elif arch_pi == [256, 256]:
                     best_params_dict["net_arch_size"] = "large"

            with open(BEST_PARAMS_FILE, 'w') as f:
                json.dump(best_params_dict, f, indent=4)
            logger.info(f"Successfully saved best parameters to {BEST_PARAMS_FILE}")
        except IOError as e:
            logger.error(f"Error saving best parameters to {BEST_PARAMS_FILE}: {e}")

    except ValueError:
        logger.warning("No trials completed successfully in the study. Cannot determine best parameters.")

    logger.info("Optuna tuning script finished.") 