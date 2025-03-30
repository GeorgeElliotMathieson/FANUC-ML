import subprocess
import argparse
import logging
import os
import sys
import shlex

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Project root is one level up from the scripts directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")
# Define the path to the source directory relative to project root
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
OPTUNA_STUDY_DIR = os.path.join(OUTPUT_DIR, 'optuna_study')

def run_command(command_list, step_name):
    """Runs a command as a subprocess and checks the return code."""
    logger.info(f"--- Starting Step: {step_name} ---")
    # Prepend python executable and use '-m' to run modules from src
    # Example: python -m src.tune --duration ...
    module_path = command_list[0] # e.g., "src.tune"
    args = command_list[1:]
    full_command = [sys.executable, "-m", module_path] + args

    logger.info(f"Executing command: {shlex.join(full_command)}")
    try:
        process = subprocess.run(
            full_command,
            check=True,
            text=True,
            # Set cwd ensures scripts run from project root, helps with relative paths
            cwd=PROJECT_ROOT
        )
        logger.info(f"--- Step Completed: {step_name} (Exit Code: {process.returncode}) ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"--- Step Failed: {step_name} (Exit Code: {e.returncode}) ---")
        logger.error(f"Command: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        # This error is less likely with '-m', but kept for safety
        logger.error(f"--- Step Failed: {step_name} --- Error finding module '{module_path}'")
        return False
    except Exception as e:
        logger.error(f"--- Step Failed: {step_name} ---")
        logger.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Remove sys.path manipulation - not needed when running scripts/run_pipeline.py
    # if SRC_DIR not in sys.path:
    #     sys.path.insert(0, SRC_DIR)

    parser = argparse.ArgumentParser(description="Run the full Fanuc RL pipeline (tune, train, test).")
    parser.add_argument("--tune_duration", type=int, default=16,
                        help="Duration for hyperparameter tuning in minutes (default: 16).")
    parser.add_argument("--train_duration", type=int, default=29,
                        help="Duration for training in minutes (default: 29).")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of episodes for visual testing (default: 5).")
    parser.add_argument("--study_name", type=str, default="ppo_fanuc_pipeline_study",
                        help="Name for the Optuna study (default: ppo_fanuc_pipeline_study).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible tuning trials (default: 42).")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="Skip the tuning step and use existing best_params.json (if available).")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the training step (requires existing model).")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Skip the visual testing step.")
    parser.add_argument("--delete_old_params", action="store_true",
                        help=f"Delete the existing {BEST_PARAMS_FILE} before tuning.")

    args = parser.parse_args()

    logger.info("Starting Fanuc RL Pipeline...")
    logger.info(f"Arguments: {vars(args)}")

    pipeline_successful = True

    # --- Step 1: Tuning ---
    if not args.skip_tuning:
        if args.delete_old_params:
            if os.path.exists(BEST_PARAMS_FILE):
                try:
                    os.remove(BEST_PARAMS_FILE)
                    logger.info(f"Deleted existing {BEST_PARAMS_FILE}.")
                except OSError as e:
                    logger.warning(f"Could not delete {BEST_PARAMS_FILE}: {e}")

        # Command no longer needs --storage argument
        tune_command = [
            "src.tune",
            "--duration", str(args.tune_duration),
            "--study_name", args.study_name,
            "--seed", str(args.seed)
        ]

        if not run_command(tune_command, "Hyperparameter Tuning"):
            logger.error("Tuning failed. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("--- Skipping Step: Hyperparameter Tuning ---")
        if not os.path.exists(BEST_PARAMS_FILE):
             logger.warning(f"Tuning skipped, but {BEST_PARAMS_FILE} not found. Training will use default parameters.")

    # --- Step 2: Training ---
    if not args.skip_training:
        # Update command to use module path 'src.train'
        train_command = [
            "src.train", # Module path
            "--duration", str(args.train_duration)
        ]
        if not run_command(train_command, "Training"):
            logger.error("Training failed. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("--- Skipping Step: Training ---")

    # --- Step 3: Visual Testing ---
    if not args.skip_testing:
        # Update command to use module path 'src.test'
        test_command = [
            "src.test", # Module path
            "--episodes", str(args.test_episodes)
        ]
        # We generally don't need to abort the whole pipeline if only the visual test fails
        if not run_command(test_command, "Visual Testing"):
            logger.warning("Visual testing step failed or encountered an error.")
            # pipeline_successful = False # Optionally mark pipeline as failed
    else:
        logger.info("--- Skipping Step: Visual Testing ---")

    logger.info("--- Fanuc RL Pipeline Finished ---")
    if not pipeline_successful:
         logger.warning("Pipeline completed, but one or more non-critical steps failed.") 