import subprocess
import argparse
import logging
import os
import sys
import shlex

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Best params file path
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")
# Source directory path
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
# Output directory path
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
# Optuna study directory path
OPTUNA_STUDY_DIR = os.path.join(OUTPUT_DIR, 'optuna_study')

def run_command(command_list, step_name):
    """Runs a command."""
    logger.info(f"Starting Step: {step_name}")
    # Run module via python -m
    module_path = command_list[0]
    args = command_list[1:]
    full_command = [sys.executable, "-m", module_path] + args

    logger.info(f"Executing command: {shlex.join(full_command)}")
    try:
        process = subprocess.run(
            full_command,
            check=True,
            text=True,
            # Run from project root
            cwd=PROJECT_ROOT
        )
        logger.info(f"Step Completed: {step_name} (Exit Code: {process.returncode})")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Step Failed: {step_name} (Exit Code: {e.returncode})")
        logger.error(f"Command: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        # Module not found
        logger.error(f"Step Failed: {step_name} - Error finding module '{module_path}'")
        return False
    except Exception as e:
        logger.error(f"Step Failed: {step_name}")
        logger.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Removed sys.path manipulation
    # if SRC_DIR not in sys.path:
    #     sys.path.insert(0, SRC_DIR)

    parser = argparse.ArgumentParser(description="Run Fanuc RL pipeline.")
    parser.add_argument("--tune_duration", type=int, default=16,
                        help="Tuning duration (mins)")
    parser.add_argument("--train_duration", type=int, default=29,
                        help="Training duration (mins)")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Visual test episodes")
    parser.add_argument("--study_name", type=str, default="ppo_fanuc_pipeline_study",
                        help="Optuna study name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Tuning random seed")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="Skip tuning step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training step")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Skip visual testing step")
    parser.add_argument("--delete_old_params", action="store_true",
                        help=f"Delete existing {BEST_PARAMS_FILE}")

    args = parser.parse_args()

    logger.info("Starting Fanuc RL Pipeline...")
    logger.info(f"Arguments: {vars(args)}")

    pipeline_successful = True

    # Tuning
    if not args.skip_tuning:
        if args.delete_old_params:
            if os.path.exists(BEST_PARAMS_FILE):
                try:
                    os.remove(BEST_PARAMS_FILE)
                    logger.info(f"Deleted existing {BEST_PARAMS_FILE}.")
                except OSError as e:
                    logger.warning(f"Could not delete {BEST_PARAMS_FILE}: {e}")

        # No storage arg needed
        tune_command = [
            "src.rl.tune",
            "--duration", str(args.tune_duration),
            "--study_name", args.study_name,
            "--seed", str(args.seed)
        ]

        if not run_command(tune_command, "Hyperparameter Tuning"):
            logger.error("Tuning failed. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Step: Hyperparameter Tuning")
        if not os.path.exists(BEST_PARAMS_FILE):
             logger.warning(f"Tuning skipped, {BEST_PARAMS_FILE} not found. Using defaults.")

    # Training
    if not args.skip_training:
        # Module path 'src.rl.train'
        train_command = [
            "src.rl.train",
            "--duration", str(args.train_duration)
        ]
        if not run_command(train_command, "Training"):
            logger.error("Training failed. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Step: Training")

    # Visual Testing
    if not args.skip_testing:
        # Module path 'src.rl.test'
        test_command = [
            "src.rl.test",
            "--episodes", str(args.test_episodes)
        ]
        # Don't abort if only visual test fails
        if not run_command(test_command, "Visual Testing"):
            logger.warning("Visual testing step failed.")
            # pipeline_successful = False # Optional failure flag
    else:
        logger.info("Skipping Step: Visual Testing")

    logger.info("Fanuc RL Pipeline Finished.")
    if not pipeline_successful:
         logger.warning("Pipeline completed, non-critical steps failed.") 