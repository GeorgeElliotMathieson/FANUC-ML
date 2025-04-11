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

# Project root path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
# Default log directory
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
# Default model filename
DEFAULT_MODEL_NAME = "final_model.zip"
# Config directory
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")

# Latest run directory finder
def find_latest_run_dir(base_log_dir: str) -> str | None:
    """Find latest run dir."""
    try:
        # List run directories
        run_dirs = [d for d in os.listdir(base_log_dir)
                    if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("run_")]
        if not run_dirs:
            return None
        latest_run_dir = max(run_dirs)
        return os.path.join(base_log_dir, latest_run_dir)
    except FileNotFoundError:
        logger.warning(f"Base log directory not found: {base_log_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest run directory in {base_log_dir}: {e}")
        return None

# Model file finder in run directory
def find_model_in_run_dir(run_dir: str) -> str | None:
    """Find model in run dir."""
    try:
        # Check final model path
        final_model_path = os.path.join(run_dir, DEFAULT_MODEL_NAME)
        if os.path.exists(final_model_path):
            logger.info(f"Found final model: {final_model_path}")
            return final_model_path

        # Check for checkpoints
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
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Command line arguments
    parser = argparse.ArgumentParser(description="Run visual test for trained PPO FANUC model.")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help=f"Path to model .zip. If None, searches {DEFAULT_LOG_DIR}")
    parser.add_argument("-e", "--episodes", type=int, default=5,
                        help="Number of test episodes to run (default: 5).")
    args = parser.parse_args()

    # Set model file path
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

    if not model_file or not os.path.exists(model_file):
        logger.error(f"Error: Model file not found. Tried searching latest run in {DEFAULT_LOG_DIR} or using specified path: {args.model_path or '(searched)'}")
        logger.error("Please ensure a model has been trained or specify a valid path using --model_path.")
        sys.exit(1)

    logger.info(f"Loading model from {model_file} for visual testing...")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    try:
        loaded_model = PPO.load(model_file, device=device)
    except Exception as e:
        logger.error(f"Error loading model from {model_file}: {e}")
        sys.exit(1)

    # Default angle bonus
    loaded_angle_bonus = 5.0
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

    logger.info("Creating test environment with rendering...")
    test_env = None
    try:
        # Initialise test environment
        test_env = FanucEnv(
            render_mode='human',
            angle_bonus_factor=loaded_angle_bonus,
            force_outer_radius=True
        )

        # Start testing episodes
        logger.info(f"Testing model for {args.episodes} episodes...")
        for episode in range(args.episodes):
            logger.info(f"Test Episode {episode + 1}/{args.episodes}")

            # Reset environment
            obs, info = test_env.reset()

            # Episode variables
            terminated = False
            truncated = False
            step = 0
            total_reward = 0

            # Run episode loop
            while not (terminated or truncated):
                # Predict action
                action, _states = loaded_model.predict(obs, deterministic=True)
                # Step environment
                obs, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                step += 1
                # Delay for visualisation
                time.sleep(1./60.)

                if terminated or truncated:
                    # Log episode summary
                    episode_summary = (
                        f"Episode finished after {step} steps.\n"
                        f"  Terminated: {terminated}, Truncated: {truncated}\n"
                        f"  Total Reward: {total_reward:.2f}\n"
                        f"  Final Distance: {info.get('distance', 'N/A'):.4f}\n"
                        f"  Success: {info.get('success', 'N/A')}\n"
                        f"  Collision: {info.get('collision', 'N/A')}"
                    )
                    logger.info(episode_summary)
                    logger.info("Pausing before next episode...")
                    time.sleep(2)
                    break

        # Testing complete
        logger.info("Visual testing finished.")

    except Exception as e:
        # Log testing errors
        logger.error(f"An error occurred during testing: {e}")
        logger.error(traceback.format_exc())
    finally:
        if test_env is not None:
             # Close environment
             logger.info("Closing test environment.")
             test_env.close()

    # Script completion
    logger.info("Test script finished.") 