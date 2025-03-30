import os
import multiprocessing
import time
import torch
import numpy as np
import json
import argparse
import math # Keep for potential calculations
import logging # Import logging
import traceback # Import traceback for logging errors
from typing import Dict, Any # Updated typing imports
import random # <-- Import random for seeding
import collections # Import collections for deque
import sys # Import sys for adjusting sys.path

# --- Configure Logging (Global Scope) --- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger for this module

# Import optuna
import optuna # type: ignore
# Import TrialPruned for handling trial errors
from optuna.exceptions import TrialPruned # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import the custom environment using relative import
from .fanuc_env import FanucEnv

# --- Optuna Tuning Configuration ---
# Keep timestep/eval config, remove iteration/convergence config
TUNE_TIMESTEPS = 2_000_000  # Timesteps per Optuna trial (Increased significantly)
NUM_EVAL_EPISODES = 20  # Number of episodes for evaluation after each trial
# N_OPTUNA_TRIALS = 50 # Defined via command-line argument now

# Define paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
OPTUNA_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "optuna_study") # Adjusted path
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json") # Adjusted path

# Default policy if not specified in Optuna trial
DEFAULT_POLICY = "MlpPolicy"

# --- Function to Run a Single Experiment (modified for Optuna) ---
def run_experiment(params: Dict[str, Any], trial_number: int, seed: int | None = None) -> float:
    """Runs a single training and evaluation experiment for an Optuna trial.

    Args:
        params: Dictionary of hyperparameters suggested by Optuna.
        trial_number: The Optuna trial number for logging.
        seed: Optional random seed for reproducibility.

    Returns:
        The mean reward achieved during evaluation.
        Raises TrialPruned if the trial fails or should be stopped early.
    """
    run_name = f"trial_{trial_number}"
    # Use logger for trial info
    logging.info(f"\n--- Starting Optuna {run_name} ---")
    logging.info(f"Parameters: {params}")

    train_env = None
    eval_env = None
    mean_reward = -float('inf') # Default reward if run fails

    # Determine number of CPUs (can be moved outside if constant)
    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count # Use all available cores

    # --- Set Seeds handled by PPO constructor --- 

    try:
        # --- Environment Setup ---
        logging.info(f"Using {num_cpu} parallel environments for training.")
        vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
        angle_bonus = params.get("angle_bonus_factor", 5.0)
        # Use simple lambda, FanucEnv is imported relatively now
        train_env = make_vec_env(
            lambda: FanucEnv(render_mode=None, angle_bonus_factor=angle_bonus),
            n_envs=num_cpu,
            vec_env_cls=vec_env_cls # type: ignore
        )
        eval_env = FanucEnv(render_mode=None, angle_bonus_factor=angle_bonus)

        # --- Agent and Training ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {device}") # Less verbose during Optuna runs

        # Create log directory for this specific trial (path updated)
        trial_log_dir = os.path.join(OPTUNA_LOG_DIR, run_name)
        os.makedirs(trial_log_dir, exist_ok=True)

        # Ensure policy is correctly passed
        policy = params.get("policy", DEFAULT_POLICY)
        policy_kwargs = params.get("policy_kwargs", None)

        # Create PPO model with suggested parameters and seed
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
            verbose=0, # Keep verbosity low for Optuna runs
            tensorboard_log=trial_log_dir, # Use trial-specific log dir
            device=device,
            policy_kwargs=policy_kwargs,
            seed=seed
        )

        # Use logger for info
        logging.info(f"Training for {TUNE_TIMESTEPS} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=TUNE_TIMESTEPS,
            log_interval=1000 # Log even less frequently
        )
        training_time = time.time() - start_time
        # Use logger for info
        logging.info(f"Training completed in {training_time:.2f} seconds.")

        # --- Evaluation ---
        # Use logger for info
        logging.info(f"Evaluating model over {NUM_EVAL_EPISODES} episodes...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVAL_EPISODES, deterministic=True) # type: ignore
        # Use logger for info
        logging.info(f"Evaluation results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

        # Optional: Save the model from this run
        # model_save_path = os.path.join(trial_log_dir, "tuned_model.zip")
        # model.save(model_save_path)

    except Exception as e:
        # Use logger for error and traceback
        logging.error(f"Error during Optuna trial {trial_number} ({run_name}): {e}")
        # Log the full traceback for debugging
        logging.error(traceback.format_exc())
        # Prune the trial if it fails catastrophically
        raise TrialPruned(f"Trial failed due to error: {e}")
    finally:
        # --- Cleanup ---
        # print(f"Cleaning up environments for run {run_name}...") # Less verbose
        if train_env is not None:
            try:
                train_env.close()
            except Exception as e:
                # Use logger for warning
                logging.warning(f"Error closing train_env: {e}")
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception as e:
                # Use logger for warning
                 logging.warning(f"Error closing eval_env: {e}")
        # Use logger for info
        logging.info(f"--- Finished Optuna {run_name} ---")

    # Return the metric Optuna should optimize (higher is better)
    return mean_reward

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, seed: int | None = None) -> float:
    """Defines the objective function for Optuna optimisation.

    Args:
        trial: The Optuna trial object.
        seed: Optional random seed to pass to the experiment.

    Returns:
        The evaluation score (mean reward).
    """

    # Define hyperparameters to search using trial.suggest_*
    params = {
        "policy": trial.suggest_categorical("policy", ["MlpPolicy"]), # Keep policy fixed for now
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        # Suggest n_steps as power of 2 for efficiency? Let's try categorical for powers of 2
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048]),
        # Batch size often related to n_steps, also suggest powers of 2
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999, log=False), # Linear scale might be ok here
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        # Add angle_bonus_factor to tuning parameters
        "angle_bonus_factor": trial.suggest_float("angle_bonus_factor", 1.0, 15.0, log=False), # Linear scale
        # "policy_kwargs": None # Placeholder - will be set below
    }

    # --- Constraint: batch_size <= n_steps ---
    # Ensure batch_size is not larger than n_steps, which is invalid for PPO.
    # If it is, we can prune the trial early or simply adjust batch_size.
    # Pruning is cleaner as it avoids running an invalid configuration.
    if params["batch_size"] > params["n_steps"]: # type: ignore
         # Use logger for info
         logging.info(f"Pruning trial {trial.number}: batch_size ({params['batch_size']}) > n_steps ({params['n_steps']})")
         raise TrialPruned("batch_size > n_steps")

    # --- Add Policy Kwargs suggestion (Network size) ---
    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large"])
    if net_arch_size == "small":
        params["policy_kwargs"] = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    elif net_arch_size == "medium":
        params["policy_kwargs"] = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    else: # "large"
        params["policy_kwargs"] = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # Run the experiment with the suggested parameters and seed
    mean_reward = run_experiment(params, trial.number, seed=seed)

    # Optional: Report intermediate results for pruning (if using a pruner)
    # trial.report(mean_reward, step=TUNE_TIMESTEPS)
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    return mean_reward


if __name__ == "__main__":
    # --- Adjust sys.path if run directly (less ideal now) ---
    if '' not in sys.path:
         sys.path.insert(0, os.path.dirname(__file__))
    # Re-define paths needed for direct execution context
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
    OPTUNA_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "optuna_study")
    BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")
    logger.info(f"Executing {__file__} directly. Paths adjusted.")

    parser = argparse.ArgumentParser(description="Run PPO hyperparameter tuning using Optuna for Fanuc Env.")
    parser.add_argument("-d", "--duration", type=int, default=12, # Default in minutes
                        help="Maximum tuning duration in minutes (default: 12).")
    parser.add_argument("--study_name", type=str, default="ppo_fanuc_tuning", help="Name for the Optuna study.")
    # Add argument for storage URL (e.g., SQLite database)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., 'sqlite:///optuna_study.db'). If None, uses in-memory storage.")
    # Add argument for seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible tuning trials (default: 42).")
    args = parser.parse_args()

    # Create Optuna log directory if it doesn't exist (path updated)
    os.makedirs(OPTUNA_LOG_DIR, exist_ok=True)

    # Use logger for info
    logger.info(f"Starting Optuna study '{args.study_name}' for a maximum duration of {args.duration} minutes.")
    logger.info(f"Using storage: {'In-memory' if args.storage is None else args.storage}")
    logger.info(f"Each trial trains for {TUNE_TIMESTEPS} timesteps and evaluates over {NUM_EVAL_EPISODES} episodes.")

    # --- Create or Load Optuna Study ---
    # Using storage allows resuming studies
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize", # We want to maximize the mean reward
        load_if_exists=True, # Load study if it already exists in the storage
        # Optional: Add a pruner for early stopping of unpromising trials
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0, interval_steps=1
        )
    )

    # --- Run Optuna Optimisation ---
    try:
        # Convert duration from minutes to seconds for timeout
        timeout_seconds = args.duration * 60
        logger.info(f"Setting Optuna timeout to {timeout_seconds} seconds ({args.duration} minutes).")
        study.optimize(
            # Pass the seed from args to the objective function using lambda
            lambda trial: objective(trial, seed=args.seed),
            n_trials=None, # Run indefinitely until timeout
            timeout=timeout_seconds # Use timeout in seconds
            # n_jobs=1 # Parallel trials can be tricky with PyBullet/multiprocessing envs
        )
    except KeyboardInterrupt:
        # Use logger for warning
        logger.warning("Optuna study interrupted by user.")
    except Exception as e:
        # Use logger for error and traceback
        logger.error(f"An error occurred during the Optuna study: {e}")
        logger.error(traceback.format_exc())

    # --- Post-Study Summary ---
    # Use logger for info
    logger.info("=============== Optuna Study Summary ===============")
    logger.info(f"Study Name: {study.study_name}")
    # Number of trials can be different if study was loaded/resumed
    logger.info(f"Number of finished trials: {len(study.trials)}")

    # Find the best trial
    try:
        best_trial = study.best_trial
        # Use logger for info
        logger.info(f"Best trial finished with value (mean reward): {best_trial.value:.4f}")
        logger.info("Best hyperparameters found:")
        # Format best parameters nicely
        best_params_dict = {}
        for key, value in best_trial.params.items():
            # Use logger for info
            logger.info(f"  {key}: {value}")
            best_params_dict[key] = value

        # --- Save Best Parameters to Central File (path updated) ---
        logger.info(f"Saving best parameters found to {BEST_PARAMS_FILE}...")
        try:
            # Ensure policy is included if it was tuned (add it if fixed)
            if "policy" not in best_params_dict:
                 best_params_dict["policy"] = DEFAULT_POLICY # Add default if not explicitly tuned
            # Similarly add policy_kwargs if tuned
            # Add net_arch_size to saved params for info
            if "policy_kwargs" in best_params_dict and "net_arch_size" not in best_params_dict:
                 # Infer net_arch_size based on policy_kwargs for saving
                 arch_pi = best_params_dict["policy_kwargs"].get("net_arch", {}).get("pi", [])
                 if arch_pi == [64, 64]:
                     best_params_dict["net_arch_size"] = "small"
                 elif arch_pi == [128, 128]:
                     best_params_dict["net_arch_size"] = "medium"
                 elif arch_pi == [256, 256]:
                     best_params_dict["net_arch_size"] = "large"
                 # else: leave it out if not matching known patterns

            with open(BEST_PARAMS_FILE, 'w') as f:
                json.dump(best_params_dict, f, indent=4)
            logger.info(f"Successfully saved best parameters to {BEST_PARAMS_FILE}")
        except IOError as e:
            logger.error(f"Error saving best parameters to {BEST_PARAMS_FILE}: {e}")

        # --- Optional: Display Trials DataFrame ---
        # try:
        #     df = study.trials_dataframe()
        #     print("--- Trials DataFrame ---")
        #     print(df.sort_values(by="value", ascending=False))
        #     # Save DataFrame to CSV
        #     # df.to_csv(os.path.join(OPTUNA_LOG_DIR, f"{args.study_name}_trials.csv"), index=False)
        # except Exception as e:
        #     print(f"Could not display or save trials dataframe: {e}")

    except ValueError:
        # Use logger for warning
        logger.warning("No trials completed successfully in the study. Cannot determine best parameters.")

    # Use logger for info
    logger.info("Optuna tuning script finished.") 