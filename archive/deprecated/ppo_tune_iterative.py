import os
import multiprocessing
import time
import torch
import numpy as np
import json
import argparse
import math # Import math for ceiling function
from typing import Tuple, List, Dict, Any # Add typing imports

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import the custom environment
from fanuc_env import FanucEnv

# --- Tuning Configuration ---
TUNE_TIMESTEPS = 250_000
NUM_EVAL_EPISODES = 20 # Number of episodes for evaluation after each run
RESULTS_FILE = "ppo_tuning_results.json"
TUNING_LOG_DIR = "./ppo_tuning_logs/"
BEST_PARAMS_FILE = "best_params.json" # Central file for best parameters
# --- Iterative Tuning Config ---
MAX_ITERATIONS = 5
CONVERGENCE_THRESHOLD = 500.0 # Increased from 100.0

# --- Baseline Hyperparameters ---
# Resetting based on educated guess for new env (squared reward, curriculum)
BASELINE_PARAMS = {
    "learning_rate": 3e-4,      # Kept default
    "n_steps": 1024,            # Reduced from 2048 (more frequent updates)
    "batch_size": 64,           # Kept default
    "n_epochs": 5,              # Reduced from 10 (less overfitting per update)
    "gamma": 0.99,            # Kept default
    "gae_lambda": 0.95,         # Kept default
    "clip_range": 0.2,          # Kept default
    "ent_coef": 0.005,          # Increased from 0.001 (more exploration)
    "vf_coef": 0.5,             # Kept default
    "max_grad_norm": 0.5,       # Kept default
    "policy": "MlpPolicy",    # Kept default
    # Add policy_kwargs here if you want to tune network architecture, e.g.:
    # "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
}

# --- Parameters to Tune (Relative Method) ---
# Format: {param_name: (type, [factors_or_offsets])}
# Types: 'mult' (multiplicative float), 'int_mult' (multiplicative integer),
#        'add' (additive float), 'int_add' (additive integer)
# Use type hints for clarity
PARAMS_TO_TUNE: Dict[str, Tuple[str, List[float]]] = {
    # param_name: (type, [factors_or_offsets])
    "learning_rate": ("mult", [0.5, 1.5]),  # Multiply baseline
    "n_steps": ("int_mult", [0.5, 2.0]), # Multiply baseline, ensure integer >= min_val
    "batch_size": ("int_mult", [0.5, 2.0]), # Multiply baseline, ensure integer >= min_val
    "n_epochs": ("int_add", [-2, 2]),  # Add to baseline (min 1)
    "gamma": ("add", [-0.005, 0.005]), # Add to baseline (clip 0-1)
    "gae_lambda": ("add", [-0.02, 0.02]),# Add to baseline (clip 0-1)
    "clip_range": ("mult", [0.8, 1.2]), # Multiply baseline
    # For ent_coef, multiplicative might be tricky if baseline is 0. Consider additive or small absolute values?
    # Let's try additive for ent_coef for now.
    "ent_coef": ("add", [0.001, 0.01]), # Add to baseline (min 0)
    "vf_coef": ("mult", [0.8, 1.2]),  # Multiply baseline
    "max_grad_norm": ("mult", [0.8, 1.2]),# Multiply baseline
}
# Minimum values for certain integer parameters
PARAM_MIN_VALUES = {
    "n_steps": 64,
    "batch_size": 8,
    "n_epochs": 1,
}

# --- Helper function to calculate test value ---
def calculate_test_value(baseline_value, param_type, modifier):
    if param_type == "mult":
        return baseline_value * modifier
    elif param_type == "int_mult":
        min_val = PARAM_MIN_VALUES.get(param_name, 1) # Use specific min or default to 1
        # Use math.ceil to ensure we don't round down to 0 for small baseline * factor
        return max(min_val, int(math.ceil(baseline_value * modifier)))
    elif param_type == "add":
        new_val = baseline_value + modifier
        # Apply specific bounds if needed (e.g., gamma, gae_lambda)
        if param_name == "gamma" or param_name == "gae_lambda":
            new_val = np.clip(new_val, 0.0, 1.0)
        if param_name == "ent_coef":
            new_val = max(0.0, new_val) # Ensure non-negative
        return new_val
    elif param_type == "int_add":
        min_val = PARAM_MIN_VALUES.get(param_name, 1)
        return max(min_val, baseline_value + modifier)
    else:
        raise ValueError(f"Unknown parameter tuning type: {param_type}")

# --- Helper function to create run name ---
def create_run_name(param_name, param_type, modifier):
    mod_str = ""
    if param_type == "mult" or param_type == "int_mult":
        mod_str = f"x{modifier}"
    elif param_type == "add" or param_type == "int_add":
        mod_str = f"add{modifier:+}" # Include sign for additive
    # Format modifier for filename (replace . with p, handle +/-)
    mod_str_clean = mod_str.replace('.', 'p').replace('+', 'plus').replace('-', 'minus')
    return f"{param_name}_{mod_str_clean}"

# --- Function to determine best parameters based on relative results ---
def determine_best_params(results, current_baseline):
    print("\n--- Determining Best Parameters from Iteration Results ---")
    if not results:
        print("No results found for this iteration. Using current baseline.")
        return current_baseline.copy()

    best_params = current_baseline.copy()
    results_dict = {res.get("run_name"): res for res in results if "error" not in res and "mean_reward" in res}

    # Get baseline performance for this iteration
    baseline_reward = results_dict.get("baseline", {}).get("mean_reward", -np.inf)
    print(f"Baseline reward for this iteration: {baseline_reward if baseline_reward > -np.inf else 'N/A'}")

    # Iterate through tunable parameters
    for param_name, (param_type, modifiers) in PARAMS_TO_TUNE.items():
        print(f"Checking param: {param_name}")
        best_value = current_baseline[param_name] # Start with the baseline value for this iter
        best_reward_for_param = baseline_reward
        found_better = False

        # Check each tested modifier for this parameter
        for modifier in modifiers:
            run_name = create_run_name(param_name, param_type, modifier)
            run_result = results_dict.get(run_name)
            calculated_value = calculate_test_value(current_baseline[param_name], param_type, modifier) # Value tested

            if run_result:
                reward = run_result.get("mean_reward", -np.inf)
                print(f"  - Modifier: {modifier} ({param_type}), Tested Value: {calculated_value}, Run: {run_name}, Reward: {reward:.2f}")
                # Use >= to prefer modifications slightly if reward is identical to baseline
                if reward >= best_reward_for_param:
                    # Special case: if reward is same as baseline, only update if value is different
                    if reward == baseline_reward and calculated_value == current_baseline[param_name]:
                        continue
                    best_reward_for_param = reward
                    best_value = calculated_value # Store the actual value, not the modifier
                    found_better = True
            else:
                print(f"  - Modifier: {modifier} ({param_type}), Tested Value: {calculated_value}, Run: {run_name} - No results found or run failed.")

        if found_better and best_value != current_baseline[param_name]:
             print(f"  => New best for {param_name}: {best_value} (Reward: {best_reward_for_param:.2f})")
             best_params[param_name] = best_value
        else:
             # Keep the value from the start of the iteration (which was best from *previous* iter)
             print(f"  => Keeping current baseline value for {param_name}: {current_baseline[param_name]}")
             best_params[param_name] = current_baseline[param_name]


    print("--- Finished Determining Best Parameters for this Iteration ---")
    return best_params

def run_tuning_experiment(params, run_name, num_cpu, iteration, run_number_this_iter, total_runs_this_iter, overall_run_number):
    """Runs a single training and evaluation experiment."""
    print(f"\n--- Iteration {iteration} / Test {run_number_this_iter}/{total_runs_this_iter} (Overall Run {overall_run_number}) ---")
    print(f"--- Starting Run: {run_name} ---")
    print(f"Parameters: {params}")

    train_env = None
    eval_env = None
    run_results = {"run_name": run_name, "params": params, "iteration": iteration}

    try:
        # --- Environment Setup ---
        print(f"Using {num_cpu} parallel environments for training.")
        vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
        # Create train envs
        train_env = make_vec_env(lambda: FanucEnv(render_mode=None), n_envs=num_cpu, vec_env_cls=vec_env_cls)

        # Create a separate evaluation environment (single, non-rendered)
        eval_env = FanucEnv(render_mode=None) # Use default settings from FanucEnv

        # --- Agent and Training ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        # Include iteration in log dir path
        run_log_dir = os.path.join(TUNING_LOG_DIR, f"iter_{iteration}", run_name)
        os.makedirs(run_log_dir, exist_ok=True)

        model = PPO(
            params.get("policy", BASELINE_PARAMS["policy"]),
            train_env,
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
            verbose=0, # Reduce verbosity
            tensorboard_log=run_log_dir,
            device=device,
            policy_kwargs=params.get("policy_kwargs", None) # Use policy_kwargs if defined
        )

        print(f"Training for {TUNE_TIMESTEPS} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=TUNE_TIMESTEPS,
            log_interval=100 # Log less frequently
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        run_results["training_time_s"] = training_time

        # --- Evaluation ---
        print(f"Evaluating model over {NUM_EVAL_EPISODES} episodes...")
        # Use wrapper env for evaluation if needed, otherwise direct env
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVAL_EPISODES, deterministic=True)
        print(f"Evaluation results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        run_results["mean_reward"] = mean_reward
        run_results["std_reward"] = std_reward

        # Optionally save the model from this run (can take lots of space)
        # model_save_path = os.path.join(run_log_dir, "tuned_model.zip")
        # model.save(model_save_path)
        # print(f"Model for this run saved to {model_save_path}")

    except Exception as e:
        print(f"Error during run {run_name}: {e}")
        import traceback
        traceback.print_exc()
        run_results["error"] = str(e)
    finally:
        # --- Cleanup ---
        print(f"Cleaning up environments for run {run_name}...")
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            # Since eval_env is not VecEnv, it might not have close(), but FanucEnv does
            try:
                eval_env.close()
            except AttributeError:
                pass # FanucEnv might handle closing pybullet internally if needed
        print(f"--- Finished Run: {run_name} ---")

    return run_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO hyperparameter tuning for Fanuc Env.")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip running the baseline configuration in the first iteration.")
    # Argument to disable baseline update is less relevant now, removing.
    # parser.add_argument("--no_update_baseline", action="store_true", help="Do not update baseline parameters from previous results.")
    parser.add_argument("--start_iteration", type=int, default=0, help="Iteration number to start from (loads previous results).")
    args = parser.parse_args()

    # Determine number of CPUs for parallel training runs
    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count # Use all available cores
    print(f"Using {num_cpu} CPUs for parallel environments during tuning.") # Add print statement

    # Define the initial baseline (hardcoded)
    initial_baseline_params = BASELINE_PARAMS.copy()
    current_baseline_params = initial_baseline_params
    best_params_overall = initial_baseline_params # Track the best params found across iterations
    previous_baseline_reward = -np.inf

    # --- Load Best Params Found So Far (if starting fresh or resuming) ---
    if args.start_iteration == 0:
        if os.path.exists(BEST_PARAMS_FILE):
            try:
                with open(BEST_PARAMS_FILE, 'r') as f:
                    current_baseline_params = json.load(f)
                    best_params_overall = current_baseline_params.copy()
                    print(f"Loaded existing best parameters from {BEST_PARAMS_FILE} as starting baseline.")
            except Exception as e:
                print(f"Warning: Failed to load {BEST_PARAMS_FILE}. Using initial defaults. Error: {e}")
                current_baseline_params = initial_baseline_params
                best_params_overall = initial_baseline_params
        else:
            print(f"No {BEST_PARAMS_FILE} found. Using initial defaults as starting baseline.")
            current_baseline_params = initial_baseline_params
            best_params_overall = initial_baseline_params
    # If resuming (--start_iteration > 0), the baseline will be determined later from the previous iteration's results file

    total_runs_executed_this_session = 0 # Initialize overall counter HERE

    # --- Calculate Total Tests Per Iteration ---
    total_tests_per_iteration = 1 # For baseline
    for _, (_, modifiers) in PARAMS_TO_TUNE.items():
        total_tests_per_iteration += len(modifiers)
    print(f"Calculated {total_tests_per_iteration} tests per iteration (1 baseline + variations).")

    # --- Iteration Loop ---
    for iteration_count in range(args.start_iteration, MAX_ITERATIONS):
        print(f"\n=============== Starting Iteration {iteration_count} ===============")
        experiment_counter_this_iteration = 0 # Reset counter for this iteration

        iteration_results_file = f"ppo_tuning_results_iter_{iteration_count}.json"
        previous_iteration_results_file = f"ppo_tuning_results_iter_{iteration_count - 1}.json"

        all_prev_results = []
        previous_baseline_reward = -np.inf # Reset for safety
        # Load previous iteration's results if not the first iteration
        if iteration_count > 0:
            if os.path.exists(previous_iteration_results_file):
                try:
                    with open(previous_iteration_results_file, 'r') as f:
                        all_prev_results = json.load(f)
                        print(f"Loaded {len(all_prev_results)} results from previous iteration: {previous_iteration_results_file}")

                        # Store the baseline reward from the previous iteration for convergence check
                        prev_baseline_run = next((res for res in all_prev_results if res.get("run_name") == "baseline" and "error" not in res), None)
                        if prev_baseline_run:
                            previous_baseline_reward = prev_baseline_run.get("mean_reward", -np.inf)
                            print(f"Previous iteration baseline reward: {previous_baseline_reward:.2f}")
                        else:
                            print("Could not find baseline reward from previous iteration.")
                            # previous_baseline_reward remains -np.inf

                except Exception as e:
                    print(f"Warning: Could not load previous results from {previous_iteration_results_file}. Error: {e}")
                    print("Stopping due to error loading previous results.")
                    break
            else: # Iteration > 0 but previous file missing
                print(f"Warning: Previous results file not found ({previous_iteration_results_file}). Cannot perform convergence check for this iteration.")
                # baseline determination will use previous iteration's best or initial
        else: # iteration_count == 0
             print("First iteration. Using initial baseline parameters.")

        # Determine the baseline parameters for THIS iteration based on PREVIOUS results
        if iteration_count > 0 and all_prev_results:
            current_baseline_params = determine_best_params(all_prev_results, current_baseline_params) # Use previous best as fallback
        elif iteration_count == 0:
            # Ensure first run uses initial, potentially loaded from BEST_PARAMS_FILE earlier
            pass # current_baseline_params is already set correctly
        # If iteration > 0 and no prev results, current_baseline_params retains its value

        print(f"\nCurrent Baseline Parameters for Iteration {iteration_count}:")
        for key, val in current_baseline_params.items():
            print(f"  {key}: {val}")

        # --- Run Experiments for Current Iteration --- 
        current_iteration_results = []
        # If results file for *this* iteration already exists, load it to potentially skip runs
        if os.path.exists(iteration_results_file):
            try:
                with open(iteration_results_file, 'r') as f:
                    current_iteration_results = json.load(f)
                print(f"Loaded {len(current_iteration_results)} existing results for Iteration {iteration_count} from {iteration_results_file}")
            except Exception as e:
                 print(f"Warning: Could not load existing results from {iteration_results_file}. Overwriting. Error: {e}")
                 current_iteration_results = []

        existing_run_names_this_iter = {res.get("run_name") for res in current_iteration_results}

        # --- Run Baseline for *this* iteration FIRST --- 
        baseline_run_name = "baseline"
        current_baseline_reward = -np.inf # Initialize reward for this iteration's baseline
        baseline_run_completed_this_session = False

        # Check if baseline already exists in loaded results for this iteration
        baseline_existing_result = next((res for res in current_iteration_results if res.get("run_name") == baseline_run_name and "error" not in res), None)

        if baseline_existing_result:
            print(f"--- Found Existing Baseline Result for Iteration {iteration_count} --- ")
            current_baseline_reward = baseline_existing_result.get("mean_reward", -np.inf)
            # skip_current_baseline_run = True # Don't need to run it
        else:
            # Only run baseline if not skipped by args AND not found in existing results
            skip_baseline_arg = args.skip_baseline and iteration_count == 0
            if not skip_baseline_arg:
                experiment_counter_this_iteration += 1
                total_runs_executed_this_session += 1 # Increment the overall counter
                print(f"--- Preparing Baseline for Iteration {iteration_count} --- ") # Clarify stage
                baseline_results = run_tuning_experiment(
                    current_baseline_params,
                    baseline_run_name,
                    num_cpu,
                    iteration_count,
                    experiment_counter_this_iteration, # Pass counters
                    total_tests_per_iteration,       # Pass counters
                    total_runs_executed_this_session # Pass counters
                )
                if baseline_results:
                    current_iteration_results.append(baseline_results)
                    existing_run_names_this_iter.add(baseline_run_name)
                    baseline_run_completed_this_session = True
                    if "error" not in baseline_results:
                         current_baseline_reward = baseline_results.get("mean_reward", -np.inf)
                    # Save incrementally
                    try:
                        with open(iteration_results_file, 'w') as f:
                            json.dump(current_iteration_results, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not save baseline results to {iteration_results_file}. Error: {e}")
            else:
                 print(f"--- Skipping Baseline run for Iteration {iteration_count} (--skip_baseline or already exists) --- ")

        # --- Convergence Check (Moved After Baseline Run) --- 
        if iteration_count > 0:
            print(f"\nConvergence Check: Current Baseline Reward = {current_baseline_reward:.2f}, Previous Baseline Reward = {previous_baseline_reward:.2f}")
            improvement = -np.inf
            if current_baseline_reward > -np.inf and previous_baseline_reward > -np.inf:
                 improvement = current_baseline_reward - previous_baseline_reward
            elif current_baseline_reward > -np.inf: # Current succeeded, previous failed
                 improvement = np.inf # Consider any success an infinite improvement over failure
            else: # No improvement or both failed
                 improvement = 0.0 # Or handle case where current failed but prev succeeded? Treat as negative infinity?
                 # Let's stick to 0.0, assuming we only care about positive improvement.

            # --- Revised Convergence Check --- 
            # Stop only if improvement is small but non-negative (plateau)
            # Continue if improvement is large OR if performance decreased (improvement < 0)
            if 0 <= improvement < CONVERGENCE_THRESHOLD:
                print(f"\nConvergence likely reached! Improvement ({improvement:.2f}) is non-negative and less than threshold ({CONVERGENCE_THRESHOLD}).")
                # Prevent stopping if baseline check was based on old results or skipped baseline run
                if not baseline_run_completed_this_session and not baseline_existing_result:
                     print("However, the baseline for the current iteration was skipped or failed this session. Running variations anyway.")
                elif previous_baseline_reward == -np.inf:
                     print("However, the previous baseline failed or was not found. Running variations anyway.")
                else:
                     print(f"Stopping after Iteration {iteration_count - 1}.")
                     break # Exit the iteration loop
            else:
                print(f"Improvement ({improvement:.2f}) meets threshold. Continuing tuning.")

        # --- Run Parameter Variations (Rest of the loop) --- 
        for param_name, param_config in PARAMS_TO_TUNE.items():
            param_type, modifiers = param_config # Unpack here
            print(f"\n--- Iteration {iteration_count} / Preparing variations for Parameter: {param_name} ---")
            baseline_value_for_param = current_baseline_params[param_name]

            for modifier in modifiers:
                # Calculate the actual value to be tested
                test_value = calculate_test_value(baseline_value_for_param, param_type, modifier)

                # Skip if calculated value is essentially the same as baseline
                # Use tolerance for floats
                is_same_as_baseline = False
                # Check if both are numeric before using isclose
                if isinstance(test_value, (int, float)) and isinstance(baseline_value_for_param, (int, float)):
                    # Use np.isclose for float comparison
                    if np.isclose(float(test_value), float(baseline_value_for_param)): # Cast to float for safety
                        is_same_as_baseline = True
                elif test_value == baseline_value_for_param: # Fallback for non-numeric or exact integer match
                    is_same_as_baseline = True

                if is_same_as_baseline:
                    print(f"Skipping {param_name} modifier {modifier} -> {test_value} (same as current baseline: {baseline_value_for_param})")
                    continue

                # Create run name based on modifier
                run_name = create_run_name(param_name, param_type, modifier)

                # Check if already run *within this iteration*
                if run_name in existing_run_names_this_iter:
                    print(f"Skipping run {run_name} (already run in this iteration)")
                    continue

                experiment_counter_this_iteration += 1
                total_runs_executed_this_session += 1
                run_params = current_baseline_params.copy() # Start from baseline
                run_params[param_name] = test_value # Set the calculated value

                experiment_results = run_tuning_experiment(
                    run_params,
                    run_name,
                    num_cpu,
                    iteration_count,
                    experiment_counter_this_iteration, # Pass counters
                    total_tests_per_iteration,       # Pass counters
                    total_runs_executed_this_session # Pass counters
                )
                if experiment_results:
                    current_iteration_results.append(experiment_results)
                    existing_run_names_this_iter.add(run_name)
                    # Save results incrementally within the iteration
                    try:
                        with open(iteration_results_file, 'w') as f:
                            json.dump(current_iteration_results, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not save intermediate results to {iteration_results_file}. Error: {e}")

        # --- Post-Iteration Analysis and Update ---
        print(f"\n--- Analysing Results for Iteration {iteration_count} --- ")
        print(f"Results for this iteration saved to {iteration_results_file}")

        # Determine the best params found *within this iteration* using the updated function
        if current_iteration_results:
             iteration_best_params = determine_best_params(current_iteration_results, current_baseline_params) # Pass the baseline used FOR THIS iteration

             # Update the overall best parameters and save to the central file
             print(f"Updating {BEST_PARAMS_FILE} with best parameters from Iteration {iteration_count}.")
             best_params_overall = iteration_best_params # Keep track for next iteration's baseline
             try:
                 with open(BEST_PARAMS_FILE, 'w') as f:
                     json.dump(best_params_overall, f, indent=4)
             except Exception as e:
                 print(f"Warning: Could not save best parameters to {BEST_PARAMS_FILE}. Error: {e}")

             # Set baseline for the *next* iteration
             current_baseline_params = best_params_overall.copy()
        else:
            print("No successful runs in this iteration to determine best parameters.")
            # Keep current_baseline_params as is for the next iteration

    # --- Post-Loop Summary ---
    final_iteration = iteration_count if iteration_count == MAX_ITERATIONS else iteration_count -1
    if final_iteration < args.start_iteration:
         print("\nNo iterations were run or completed successfully.")
    else:
        print(f"\n--- Final Tuning Summary (Based on Iteration {final_iteration}) ---")
        final_results_file = f"ppo_tuning_results_iter_{final_iteration}.json"
        if os.path.exists(final_results_file):
            try:
                with open(final_results_file, 'r') as f:
                    final_results = json.load(f)

                # Determine final best params from the last completed iteration
                # Load the final overall best parameters directly
                final_best_params = best_params_overall
                print("\nFinal Recommended Baseline Parameters (from latest iteration results):")
                for key, val in final_best_params.items():
                    print(f"  {key}: {val}")

                # Print summary sorted by mean reward from the final iteration
                print("\nSummary (Sorted by Mean Reward - Last Iteration):")
                successful_runs = [res for res in final_results if 'error' not in res and 'mean_reward' in res]
                failed_runs = [res for res in final_results if 'error' in res]

                for result in sorted(successful_runs, key=lambda x: x.get('mean_reward', -np.inf), reverse=True):
                     print(f"  Run: {result['run_name']:<25} - Mean Reward: {result.get('mean_reward', 'N/A'):>8.2f} +/- {result.get('std_reward', 'N/A'):.2f}")

                if failed_runs:
                    print("\nFailed Runs (Last Iteration):")
                    for result in failed_runs:
                         print(f"  Run: {result['run_name']} - ERROR: {result['error']}")

            except Exception as e:
                print(f"Error reading or processing final results file {final_results_file}: {e}")
        else:
            print(f"Final results file not found: {final_results_file}") 