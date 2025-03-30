# analyse_results.py
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
import collections # Import collections needed for deque if used later
from tensorboard.backend.event_processing import event_accumulator

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Define paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "ppo_logs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "analysis")

# --- Helper Functions ---

def find_latest_event_file(log_dir):
    """Finds the most recent TensorBoard event file in a directory."""
    try:
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
        if not event_files:
            return None
        # Filter out potential temporary files if any exist
        event_files = [f for f in event_files if os.path.getsize(f) > 0]
        if not event_files:
             return None
        return max(event_files, key=os.path.getctime)
    except FileNotFoundError:
        logger.warning(f"Log directory not found: {log_dir}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest event file in {log_dir}: {e}")
        return None


def load_tensorboard_data(event_file_path: str) -> pd.DataFrame | None:
    """Loads scalar data from a TensorBoard event file into a pandas DataFrame."""
    if not os.path.exists(event_file_path):
        logger.error(f"Event file not found: {event_file_path}")
        return None

    logger.info(f"Loading data from: {event_file_path}")
    try:
        # Increased reload_multifile to handle potential multiple event files per run
        ea = event_accumulator.EventAccumulator(event_file_path,
            size_guidance={event_accumulator.SCALARS: 0} # Load all scalar data
        )
        ea.Reload() # Load the events from file

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
                # Use wall_time for x-axis if step is not consistent? Let's stick to step.
                steps = [e.step for e in events]
                values = [e.value for e in events]
                # Handle potential duplicate steps by keeping the last value?
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

        # Create a DataFrame, forward-fill missing values, backward-fill initial NaNs
        # This helps align data from different tags that log at different intervals
        df = pd.DataFrame(data)
        # Sort by index (step) first
        df.sort_index(inplace=True)
        # Optional: Reindex to ensure uniform step intervals if needed, but can be large
        # df = df.reindex(range(df.index.min(), df.index.max() + 1))
        # Fill missing values (might occur if tags log at different frequencies)
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

    # Use a specific backend if needed, e.g., for servers without GUI
    # import matplotlib
    # matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    # Extended list including potential custom tags
    tags_to_plot = {
        'rollout/ep_rew_mean': 'Mean Episode Reward',
        'rollout/ep_len_mean': 'Mean Episode Length',
        'rollout/success_rate': 'Success Rate', # Custom
        'rollout/collision_rate': 'Collision Rate', # Custom
        'rollout/obstacle_collision_rate': 'Obstacle Collision Rate', # Custom
        'custom/current_max_target_radius': 'Curriculum Max Radius', # Custom
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/explained_variance': 'Explained Variance',
        'train/fps': 'Training FPS',
        'train/learning_rate': 'Learning Rate', # Often useful to see schedule
    }

    for tag, title in tags_to_plot.items():
        if tag in df.columns and not df[tag].isnull().all():
            try:
                plt.figure(figsize=(12, 6))
                # Use step as x-axis (index of DataFrame)
                plt.plot(df.index, df[tag], label=tag.split('/')[-1]) # Simpler label

                # Apply smoothing (rolling average) for potentially noisy plots
                # Adjust window size based on data density
                # Don't smooth FPS, loss, LR etc. by default
                if any(k in tag for k in ['rew_mean', 'ep_len', 'rate', 'radius']):
                     rolling_window = max(1, min(50, len(df) // 50)) # Adaptive window
                     # Ensure window is not larger than dataframe size
                     rolling_window = min(rolling_window, len(df))
                     if rolling_window > 1:
                         plt.plot(df.index, df[tag].rolling(window=rolling_window, min_periods=1).mean(),
                                  label=f"{tag.split('/')[-1]} (smoothed)", linestyle='--')

                plt.xlabel("Training Steps")
                plt.ylabel(title.split(' ')[-1]) # Extract unit/type
                plt.title(f"{title} during Training")
                plt.legend()
                plt.tight_layout()
                filename = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
                plt.savefig(filename)
                logger.info(f"Saved plot: {filename}")
                plt.close() # Close the plot figure to free memory
            except Exception as plot_err:
                logger.error(f"Failed to plot tag '{tag}': {plot_err}")
                plt.close() # Ensure plot is closed even if error occurs
        else:
            logger.warning(f"Tag '{tag}' not found in data or contains only NaNs, skipping plot.")

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

# --- Main Execution ---
if __name__ == "__main__":
    # REMOVE sys.path adjustment - not needed when run from scripts/
    # SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    # if SRC_DIR not in sys.path:
    #      sys.path.insert(0, SRC_DIR)

    parser = argparse.ArgumentParser(description="Analyse Fanuc RL training results.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                        help=f"Directory containing logs (default: {DEFAULT_LOG_DIR}).")
    parser.add_argument("--params_file", type=str, default=BEST_PARAMS_FILE,
                        help=f"Path to best params JSON (default: {BEST_PARAMS_FILE}).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--event_file", type=str, default=None,
                        help="Specify specific TensorBoard event file path.")

    args = parser.parse_args()

    logger.info("--- Starting Analysis ---")

    # --- Load Best Params ---
    best_params = load_best_params(args.params_file)
    if best_params:
        logger.info("Best Hyperparameters Found:")
        param_string = json.dumps(best_params, indent=4)
        for line in param_string.split('\n'):
            logger.info(f"  {line}")
    logger.info("-" * 30)

    # --- Load TensorBoard Data ---
    event_file = args.event_file
    if not event_file:
        event_file = find_latest_event_file(args.log_dir)

    if event_file:
        tb_data = load_tensorboard_data(event_file)
        logger.info("-" * 30)

        # --- Plot Data ---
        if tb_data is not None:
            plot_learning_curves(tb_data, args.output_dir)
        else:
            logger.warning("Could not load TensorBoard data for plotting.")
    else:
        logger.warning(f"No event file found in {args.log_dir}. Skipping TensorBoard analysis.")

    logger.info("--- Analysis Finished ---") 