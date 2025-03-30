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
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
DEFAULT_MODEL_NAME = "ppo_fanuc_model.zip"
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")

def find_latest_model(log_dir: str) -> str | None:
    """Finds the latest saved model .zip file in the log directory."""
    try:
        # Look for the specific model name first
        specific_model_path = os.path.join(log_dir, DEFAULT_MODEL_NAME)
        if os.path.exists(specific_model_path):
             return specific_model_path

        # Fallback to finding any rl_model or ppo_fanuc_model zip
        list_of_files = glob.glob(os.path.join(log_dir, 'ppo_fanuc_model*.zip')) + \
                          glob.glob(os.path.join(log_dir, 'rl_model_*.zip'))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        logging.error(f"Error finding latest model in {log_dir}: {e}")
        return None

if __name__ == "__main__":
    # --- Configure Logging FIRST --- 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # --- Adjust sys.path if run directly (less ideal now) ---
    if '' not in sys.path:
         sys.path.insert(0, os.path.dirname(__file__))
    # Re-define paths needed for direct execution context
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
    DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
    BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")
    logger.info(f"Executing {__file__} directly. Paths adjusted.")

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
        logger.info(f"Model path not specified, searching {DEFAULT_LOG_DIR}...")
        model_file = find_latest_model(DEFAULT_LOG_DIR)

    # Check if a model was found or specified
    if not model_file or not os.path.exists(model_file):
        fallback_path = os.path.join(DEFAULT_LOG_DIR, DEFAULT_MODEL_NAME)
        logger.error(f"Error: Model file not found or specified. Tried searching {DEFAULT_LOG_DIR} and looking for {model_file or fallback_path}")
        logger.error("Please ensure a model has been trained or specify path using --model_path.")
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