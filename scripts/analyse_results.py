# analyse_results.py
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
import collections
from tensorboard.backend.event_processing import event_accumulator # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "best_params.json")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "analysis")


def find_latest_run_dir(base_log_dir: str) -> str | None:
    """Finds the latest timestamped run directory."""
    try:
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


def find_latest_event_file(run_dir: str) -> str | None:
    """Finds the most recent TensorBoard event file."""
    try:
        logger.debug(f"Searching for event files in directory: {run_dir}")

        potential_log_subdirs = [d for d in os.listdir(run_dir)
                               if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("PPO_")]

        search_dirs = []
        if potential_log_subdirs:
            potential_log_subdirs.sort()
            log_subdir_path = os.path.join(run_dir, potential_log_subdirs[0])
            search_dirs.append(log_subdir_path)
            logger.debug(f"Found PPO subdirectory, searching inside: {log_subdir_path}")
        else:
            search_dirs.append(run_dir)
            logger.debug(f"No PPO subdirectory found, searching directly in: {run_dir}")

        event_files_full_paths = []
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                 logger.warning(f"Search directory {search_dir} not found or is not a directory.")
                 continue
            try:
                all_files = os.listdir(search_dir)
                logger.debug(f"Files found by os.listdir in {search_dir}: {all_files}")
                event_files_raw_names = [f for f in all_files if 'events.out.tfevents' in f]
                logger.debug(f"Files matching pattern in {search_dir}: {event_files_raw_names}")
                event_files_full_paths.extend([os.path.join(search_dir, f) for f in event_files_raw_names])
            except FileNotFoundError:
                 logger.warning(f"Could not list directory while searching: {search_dir}")
            except Exception as list_err:
                 logger.error(f"Error listing directory {search_dir}: {list_err}")

        if not event_files_full_paths:
            logger.warning(f"No event files found matching pattern in searched directories for run: {run_dir}")
            return None

        logger.debug(f"Potential event files (full path): {event_files_full_paths}")

        # Filter empty files
        valid_event_files = []
        for f in event_files_full_paths:
             try:
                  if os.path.getsize(f) > 0:
                       valid_event_files.append(f)
             except OSError as size_err:
                  logger.warning(f"Could not get size of {f}: {size_err}")
        logger.debug(f"Valid (non-empty) event files: {valid_event_files}")

        if not valid_event_files:
             logger.warning(f"No valid event files (non-empty) found in: {run_dir}")
             return None

        # Latest valid event file
        latest_file = max(valid_event_files, key=os.path.getctime)
        logger.debug(f"Selected latest event file: {latest_file}")
        return latest_file

    except FileNotFoundError:
        logger.warning(f"Run directory or subdirectory not found during search: {run_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding event file in {run_dir}: {e}")
        return None


def load_tensorboard_data(event_file_path: str) -> pd.DataFrame | None:
    """Loads scalar data from a TensorBoard event file into a pandas DataFrame."""
    if not os.path.exists(event_file_path):
        logger.error(f"Event file not found: {event_file_path}")
        return None

    logger.info(f"Loading data from: {event_file_path}")
    try:
        ea = event_accumulator.EventAccumulator(event_file_path,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        tags = ea.Tags()['scalars']
        if not tags:
             logger.warning(f"No scalar tags found in the event file: {event_file_path}")
             return None
        logger.info(f"Available scalar tags: {tags}")

        data = {}
        max_steps = 0

        for tag in tags:
            try:
                events = ea.Scalars(tag)
                if not events:
                     logger.warning(f"Tag '{tag}' found but contains no scalar events.")
                     continue
                steps = [e.step for e in events]
                values = [e.value for e in events]
                s = pd.Series(values, index=steps, name=tag)
                s = s[~s.index.duplicated(keep='last')]
                data[tag] = s
                max_steps = max(max_steps, steps[-1] if steps else 0)
            except KeyError:
                 logger.warning(f"Tag '{tag}' reported but could not be loaded from EventAccumulator.")
            except Exception as e_inner:
                 logger.warning(f"Could not process tag '{tag}': {e_inner}")


        if not data:
             logger.warning("No scalar data successfully extracted from the event file.")
             return None

        # Create DataFrame, fill missing values
        df = pd.DataFrame(data)
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error loading TensorBoard data from {event_file_path}: {e}")
        return None


def plot_learning_curves(df: pd.DataFrame, output_dir: str):
    """Plots key learning curves from the DataFrame."""
    if df is None or df.empty:
        logger.warning("DataFrame is empty, skipping plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")

    plt.style.use('seaborn-v0_8-darkgrid')

    tags_to_plot = {
        'rollout/ep_rew_mean': 'Mean Episode Reward',
        'rollout/ep_len_mean': 'Mean Episode Length',
        'rollout/collision_rate': 'Collision Rate',
        'rollout/obstacle_collision_rate': 'Obstacle Collision Rate',
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/explained_variance': 'Explained Variance',
        'train/approx_kl': 'Approximate KL Divergence',
        'train/clip_fraction': 'PPO Clip Fraction',
        'train/std': 'Action Standard Deviation',
        'time/fps': 'Training FPS',
        'train/learning_rate': 'Learning Rate',
    }

    for tag, title in tags_to_plot.items():
        if tag in df.columns and not df[tag].isnull().all():
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df[tag], label=tag.split('/')[-1])

                if any(k in tag for k in ['rew_mean', 'ep_len', 'rate']):
                     rolling_window = max(1, min(50, len(df) // 50))
                     rolling_window = min(rolling_window, len(df))
                     if rolling_window > 1:
                         plt.plot(df.index, df[tag].rolling(window=rolling_window, min_periods=1).mean(),
                                  label=f"{tag.split('/')[-1]} (smoothed)", linestyle='--')

                plt.xlabel("Training Steps")
                plt.ylabel(title.split(' ')[-1])
                plt.title(f"{title} during Training")
                plt.legend()
                plt.tight_layout()
                filename = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
                plt.savefig(filename)
                logger.info(f"Saved plot: {filename}")
                plt.close()
            except Exception as plot_err:
                logger.error(f"Failed to plot tag '{tag}': {plot_err}")
                plt.close()
        else:
            logger.warning(f"Tag '{tag}' not found in data or contains only NaNs, skipping individual plot.")

    success_tag = 'rollout/success_rate'
    radius_tag = 'custom/current_min_target_radius'

    if success_tag in df.columns and not df[success_tag].isnull().all() and \
       radius_tag in df.columns and not df[radius_tag].isnull().all():
        logger.info(f"Generating combined plot for {success_tag} and {radius_tag}...")
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color1 = 'tab:blue'
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Success Rate', color=color1)
            lns1 = ax1.plot(df.index, df[success_tag], color=color1, label='Success Rate')
            rolling_window = max(1, min(50, len(df) // 50))
            rolling_window = min(rolling_window, len(df))
            if rolling_window > 1:
                lns2 = ax1.plot(df.index, df[success_tag].rolling(window=rolling_window, min_periods=1).mean(),
                                  color=color1, linestyle='--', label='Success Rate (smoothed)')
            else:
                lns2 = []
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True)

            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Curriculum Min Radius (m)', color=color2)
            lns3 = ax2.plot(df.index, df[radius_tag], color=color2, label='Min Target Radius')
            ax2.tick_params(axis='y', labelcolor=color2)

            lns = lns1 + (lns2 if lns2 else []) + lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='best')

            plt.title('Success Rate vs. Curriculum Minimum Target Radius')
            fig.tight_layout()
            combined_filename = os.path.join(output_dir, 'success_vs_curriculum_radius.png')
            plt.savefig(combined_filename)
            logger.info(f"Saved combined plot: {combined_filename}")
            plt.close(fig)

        except Exception as plot_err:
            logger.error(f"Failed to create combined plot: {plot_err}")
            plt.close()
    else:
        logger.warning(f"Skipping combined Success Rate vs Curriculum Radius plot: One or both tags ('{success_tag}', '{radius_tag}') not found or NaN.")


def load_best_params(filepath: str) -> dict | None:
    """Loads the best parameters from the JSON file."""
    if not os.path.exists(filepath):
        logger.warning(f"{BEST_PARAMS_FILE} not found at {filepath}. Cannot report best params.")
        return None
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
        logger.info(f"Successfully loaded best parameters from {filepath}")
        return params
    except Exception as e:
        logger.error(f"Error loading best parameters from {filepath}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse Fanuc RL training results.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                        help=f"Directory containing logs (default: {DEFAULT_LOG_DIR}).")
    parser.add_argument("--params_file", type=str, default=BEST_PARAMS_FILE,
                        help=f"Path to best params JSON (default: {BEST_PARAMS_FILE}).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--event_file", type=str, default=None,
                        help="Specify specific TensorBoard event file path.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO).")

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("--- Starting Analysis ---")

    best_params = load_best_params(args.params_file)
    if best_params:
        logger.info("Best Hyperparameters Found:")
        param_string = json.dumps(best_params, indent=4)
        for line in param_string.split('\n'):
            logger.info(f"  {line}")
    logger.info("-" * 30)

    event_file = args.event_file
    target_log_dir = args.log_dir

    if not event_file:
        logger.info(f"Event file not specified, searching {target_log_dir} for latest run...")
        latest_run_dir = find_latest_run_dir(target_log_dir)
        if latest_run_dir:
            logger.info(f"Found latest run directory: {latest_run_dir}")
            event_file = find_latest_event_file(latest_run_dir)
        else:
            logger.warning(f"No run directories found in {target_log_dir}. Cannot find event file.")

    if event_file and os.path.exists(event_file):
        tb_data = load_tensorboard_data(event_file)
        logger.info("-" * 30)

        if tb_data is not None:
            plot_learning_curves(tb_data, args.output_dir)
        else:
            logger.warning("Could not load TensorBoard data for plotting.")
    else:
        logger.warning(f"No event file found in {args.log_dir}. Skipping TensorBoard analysis.")

    logger.info("--- Analysis Finished ---")