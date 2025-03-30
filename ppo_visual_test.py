import os
import time
import torch
import argparse
import glob
import logging
import traceback
import json

from stable_baselines3 import PPO
from fanuc_env import FanucEnv # Import the custom environment

def find_latest_model(log_dir="./ppo_fanuc_logs/") -> str | None:
    """Finds the latest saved model .zip file in the log directory."""
    try:
        list_of_files = glob.glob(os.path.join(log_dir, 'ppo_fanuc_model*.zip'))
        if not list_of_files:
            # Fallback check for older naming convention if primary not found
            list_of_files = glob.glob(os.path.join(log_dir, 'rl_model_*.zip'))
            if not list_of_files:
                return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        logging.error(f"Error finding latest model: {e}")
        return None

if __name__ == "__main__":
    # --- Configure Logging --- 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run visual test for trained PPO FANUC model.")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Path to the trained model .zip file. If not specified, tries to find the latest model in ./ppo_fanuc_logs/")
    parser.add_argument("-e", "--episodes", type=int, default=5,
                        help="Number of test episodes to run (default: 5).")

    args = parser.parse_args()

    # --- Determine Model Path ---
    model_file = args.model_path
    if model_file is None:
        logger.info("Model path not specified, searching for latest model in ./ppo_fanuc_logs/...")
        model_file = find_latest_model()
        if model_file:
            logger.info(f"Found latest model: {model_file}")
        else:
             # Default path if search fails
             model_file = "./ppo_fanuc_logs/ppo_fanuc_model.zip"
             logger.warning(f"Could not find latest model. Trying default path: {model_file}")


    # --- Load and test the trained model ---
    logger.info(f"\nLoading model from {model_file} for visual testing...")

    # Check if model file exists RIGHT BEFORE loading
    if not os.path.exists(model_file):
        logger.error(f"Error: Model file not found at {model_file}")
        logger.error("Please ensure a model has been trained and saved, or specify the correct path using --model_path.")
        exit()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the model
    try:
        loaded_model = PPO.load(model_file, device=device)
    except Exception as e:
        logger.error(f"Error loading model from {model_file}: {e}")
        exit()

    # --- Load Best Params for Environment Config --- #
    BEST_PARAMS_FILE = "best_params.json"
    loaded_angle_bonus = 5.0 # Default value (matches FanucEnv default)
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
            loaded_angle_bonus = best_params.get("angle_bonus_factor", 5.0)
            logger.info(f"Loaded angle_bonus_factor ({loaded_angle_bonus}) from {BEST_PARAMS_FILE} for test environment.")
        except Exception as e:
            logger.warning(f"Warning: Failed to load or parse {BEST_PARAMS_FILE}. Using default angle_bonus_factor ({loaded_angle_bonus}). Error: {e}")
    else:
        logger.warning(f"Warning: {BEST_PARAMS_FILE} not found. Using default angle_bonus_factor ({loaded_angle_bonus}).")

    # Create a single test environment with rendering
    logger.info("Creating test environment with rendering...")
    test_env = None
    try:
        test_env = FanucEnv(
            render_mode='human',
            angle_bonus_factor=loaded_angle_bonus,
            force_outer_radius=True # Force targets to outer edge for visual test
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

    logger.info("Script finished.") 