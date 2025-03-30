import os
import time
import torch
import argparse
import glob
import logging
import traceback
import json
import sys

from stable_baselines3 import PPO
# Import the custom environment using relative import
from .fanuc_env import FanucEnv

# Define paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..') # Adjusted for src/rl/
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
DEFAULT_MODEL_NAME = "final_model.zip" # Default model name saved by train.py
# Point to the new config directory
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")

def find_latest_run_dir(base_log_dir: str) -> str | None:
    """Finds the latest timestamped run directory (e.g., 'run_YYYYMMDD_HHMMSS')."""
    try:
        run_dirs = [d for d in os.listdir(base_log_dir)
                    if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("run_")]
        if not run_dirs:
            return None
        # Sort directories, assuming timestamp format ensures chronological order
        latest_run_dir = max(run_dirs)
        return os.path.join(base_log_dir, latest_run_dir)
    except FileNotFoundError:
        logger.warning(f"Base log directory not found: {base_log_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest run directory in {base_log_dir}: {e}")
        return None

def find_model_in_run_dir(run_dir: str) -> str | None:
    """Finds the final model or the latest checkpoint in a specific run directory."""
    try:
        # Look for the final model first
        final_model_path = os.path.join(run_dir, DEFAULT_MODEL_NAME)
        if os.path.exists(final_model_path):
            logger.info(f"Found final model: {final_model_path}")
            return final_model_path

        # Fallback to finding the latest checkpoint model
        list_of_checkpoints = glob.glob(os.path.join(run_dir, 'rl_model_*_steps.zip'))
        if not list_of_checkpoints:
            logger.warning(f"No final model or checkpoints found in {run_dir}")
            return None
        latest_checkpoint = max(list_of_checkpoints, key=os.path.getctime)
        logger.info(f"Found latest checkpoint model: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        logger.error(f"Error finding model in {run_dir}: {e}")
        return None

if __name__ == "__main__":
    # --- Configure Logging FIRST --- 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # --- Argument Parser --- 
    parser = argparse.ArgumentParser(description="Run visual test for trained PPO FANUC model.")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help=f"Path to model .zip. If None, searches {DEFAULT_LOG_DIR}")
    parser.add_argument("-e", "--episodes", type=int, default=5,
                        help="Number of test episodes to run (default: 5).")
    args = parser.parse_args()

    # --- Determine Model Path --- 
    model_file = args.model_path
    if model_file is None:
        logger.info(f"Model path not specified, searching {DEFAULT_LOG_DIR} for latest run...")
        latest_run_dir = find_latest_run_dir(DEFAULT_LOG_DIR)
        if latest_run_dir:
            logger.info(f"Found latest run directory: {latest_run_dir}")
            model_file = find_model_in_run_dir(latest_run_dir)
        else:
            logger.error(f"No run directories found in {DEFAULT_LOG_DIR}. Cannot find model.")
            sys.exit(1)

    # Check if a model was found or specified
    if not model_file or not os.path.exists(model_file):
        # Update error message
        logger.error(f"Error: Model file not found. Tried searching latest run in {DEFAULT_LOG_DIR} or using specified path: {args.model_path or '(searched)'}")
        logger.error("Please ensure a model has been trained or specify a valid path using --model_path.")
        sys.exit(1) # Exit if no model found

    logger.info(f"\nLoading model from {model_file} for visual testing...")

    # --- Determine Device --- 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Load the model --- 
    try:
        loaded_model = PPO.load(model_file, device=device)
    except Exception as e:
        logger.error(f"Error loading model from {model_file}: {e}")
        sys.exit(1) # Exit if model loading fails

    # --- Load Best Params for Environment Config --- #
    loaded_angle_bonus = 5.0 # Default value (matches FanucEnv default)
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
            loaded_angle_bonus = best_params.get("angle_bonus_factor", 5.0)
            logger.info(f"Loaded angle_bonus_factor ({loaded_angle_bonus}) from {BEST_PARAMS_FILE} for test env.")
        except Exception as e:
            logger.warning(f"Failed to load {BEST_PARAMS_FILE}. Using default angle_bonus ({loaded_angle_bonus}). Error: {e}")
    else:
        logger.warning(f"{BEST_PARAMS_FILE} not found. Using default angle_bonus ({loaded_angle_bonus}).")

    # --- Create Test Environment --- 
    logger.info("Creating test environment with rendering...")
    test_env = None
    try:
        # Use relative import FanucEnv
        test_env = FanucEnv(
            render_mode='human',
            angle_bonus_factor=loaded_angle_bonus,
            force_outer_radius=True # Keep forcing outer radius for testing
        )

        logger.info(f"Testing model for {args.episodes} episodes...")
        for episode in range(args.episodes):
            logger.info(f"--- Test Episode {episode + 1}/{args.episodes} ---")

            obs, info = test_env.reset() # Reset without stage options

            terminated = False
            truncated = False
            step = 0
            total_reward = 0

            while not (terminated or truncated):
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                # Rendering is handled by PyBullet GUI connection, step triggers update
                total_reward += reward
                step += 1
                # Add a small delay to make visualisation smoother
                time.sleep(1./60.) # ~60 Hz

                if terminated or truncated:
                    # Use logger for multi-line info
                    episode_summary = (
                        f"Episode finished after {step} steps.\n"
                        f"  Terminated: {terminated}, Truncated: {truncated}\n"
                        f"  Total Reward: {total_reward:.2f}\n"
                        f"  Final Distance: {info.get('distance', 'N/A'):.4f}\n"
                        f"  Success: {info.get('success', 'N/A')}\n"
                        f"  Collision: {info.get('collision', 'N/A')}"
                    )
                    logger.info(episode_summary)
                    # Pause briefly between episodes for better viewing
                    logger.info("Pausing before next episode...")
                    time.sleep(2)
                    break # Exit while loop

        logger.info("Visual testing finished.")

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        logger.error(traceback.format_exc())
    finally:
        if test_env is not None:
             logger.info("Closing test environment.")
             test_env.close()

    logger.info("Test script finished.") 