import os
import multiprocessing
import time
import torch
import numpy as np
import json
import argparse
import math
from typing import Tuple, List, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from fanuc_env import FanucEnv

# Tuning configuration
TUNE_TIMESTEPS = 250_000
NUM_EVAL_EPISODES = 20
RESULTS_FILE = "ppo_tuning_results.json"
TUNING_LOG_DIR = "./ppo_tuning_logs/"
BEST_PARAMS_FILE = "best_params.json"

# Iterative tuning settings
MAX_ITERATIONS = 5
CONVERGENCE_THRESHOLD = 500.0

# Baseline hyperparameters
BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

# Parameters for tuning
PARAMS_TO_TUNE: Dict[str, Tuple[str, List[float]]] = {
    "learning_rate": ("mult", [0.5, 1.5]),
    "n_steps": ("int_mult", [0.5, 2.0]),
    "batch_size": ("int_mult", [0.5, 2.0]),
    "n_epochs": ("int_add", [-2, 2]),
    "gamma": ("add", [-0.005, 0.005]),
    "gae_lambda": ("add", [-0.02, 0.02]),
    "clip_range": ("mult", [0.8, 1.2]),
    "ent_coef": ("add", [0.001, 0.01]),
    "vf_coef": ("mult", [0.8, 1.2]),
    "max_grad_norm": ("mult", [0.8, 1.2]),
}

# Minimum parameter values
PARAM_MIN_VALUES = {
    "n_steps": 64,
    "batch_size": 8,
    "n_epochs": 1,
}

# Calculate test value for tuning
def calculate_test_value(baseline_value, param_type, modifier):
    if param_type == "mult":
        return baseline_value * modifier
    elif param_type == "int_mult":
        min_val = PARAM_MIN_VALUES.get(param_name, 1)
        return max(min_val, int(math.ceil(baseline_value * modifier)))
    elif param_type == "add":
        new_val = baseline_value + modifier
        if param_name == "gamma" or param_name == "gae_lambda":
            new_val = np.clip(new_val, 0.0, 1.0)
        if param_name == "ent_coef":
            new_val = max(0.0, new_val)
        return new_val
    elif param_type == "int_add":
        min_val = PARAM_MIN_VALUES.get(param_name, 1)
        return max(min_val, baseline_value + modifier)
    else:
        raise ValueError(f"Unknown parameter tuning type: {param_type}")

# Create run name for experiment
def create_run_name(param_name, param_type, modifier):
    mod_str = ""
    if param_type == "mult" or param_type == "int_mult":
        mod_str = f"x{modifier}"
    elif param_type == "add" or param_type == "int_add":
        mod_str = f"add{modifier:+}"
    mod_str_clean = mod_str.replace('.', 'p').replace('+', 'plus').replace('-', 'minus')
    return f"{param_name}_{mod_str_clean}"

# Determine best parameters from results
def determine_best_params(results, current_baseline):
    print("\n--- Determining Best Parameters from Iteration Results ---")
    if not results:
        print("No results found for this iteration. Using current baseline.")
        return current_baseline.copy()

    best_params = current_baseline.copy()
    results_dict = {res.get("run_name"): res for res in results if "error" not in res and "mean_reward" in res}

    baseline_reward = results_dict.get("baseline", {}).get("mean_reward", -np.inf)
    print(f"Baseline reward for this iteration: {baseline_reward if baseline_reward > -np.inf else 'N/A'}")

    for param_name, (param_type, modifiers) in PARAMS_TO_TUNE.items():
        print(f"Checking param: {param_name}")
        best_value = current_baseline[param_name]
        best_reward_for_param = baseline_reward
        found_better = False

        for modifier in modifiers:
            run_name = create_run_name(param_name, param_type, modifier)
            run_result = results_dict.get(run_name)
            calculated_value = calculate_test_value(current_baseline[param_name], param_type, modifier)

            if run_result:
                reward = run_result.get("mean_reward", -np.inf)
                print(f"  - Modifier: {modifier} ({param_type}), Tested Value: {calculated_value}, Run: {run_name}, Reward: {reward:.2f}")
                if reward >= best_reward_for_param:
                    if reward == baseline_reward and calculated_value == current_baseline[param_name]:
                        continue
                    best_reward_for_param = reward
                    best_value = calculated_value
                    found_better = True
            else:
                print(f"  - Modifier: {modifier} ({param_type}), Tested Value: {calculated_value}, Run: {run_name} - No results found or run failed.")

        if found_better and best_value != current_baseline[param_name]:
             print(f"  => New best for {param_name}: {best_value} (Reward: {best_reward_for_param:.2f})")
             best_params[param_name] = best_value
        else:
             print(f"  => Keeping current baseline value for {param_name}: {current_baseline[param_name]}")
             best_params[param_name] = current_baseline[param_name]

    print("--- Finished Determining Best Parameters for this Iteration ---")
    return best_params

# Run single tuning experiment
def run_tuning_experiment(params, run_name, num_cpu, iteration, run_number_this_iter, total_runs_this_iter, overall_run_number):
    """Runs a single training and evaluation experiment."""
    print(f"\n--- Iteration {iteration} / Test {run_number_this_iter}/{total_runs_this_iter} (Overall Run {overall_run_number}) ---")
    print(f"--- Starting Run: {run_name} ---")
    print(f"Parameters: {params}")

    train_env = None
    eval_env = None
    run_results = {"run_name": run_name, "params": params, "iteration": iteration}

    try:
        print(f"Using {num_cpu} parallel environments for training.")
        vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
        train_env = make_vec_env(lambda: FanucEnv(render_mode=None), n_envs=num_cpu, vec_env_cls=vec_env_cls)

        eval_env = FanucEnv(render_mode=None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
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
            verbose=0,
            tensorboard_log=run_log_dir,
            device=device,
            policy_kwargs=params.get("policy_kwargs", None)
        )

        print(f"Training for {TUNE_TIMESTEPS} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=TUNE_TIMESTEPS,
            log_interval=100
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        run_results["training_time_s"] = training_time

        print(f"Evaluating model over {NUM_EVAL_EPISODES} episodes...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVAL_EPISODES, deterministic=True)
        print(f"Evaluation results: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        run_results["mean_reward"] = mean_reward
        run_results["std_reward"] = std_reward

    except Exception as e:
        print(f"Error during run {run_name}: {e}")
        import traceback
        traceback.print_exc()
        run_results["error"] = str(e)
    finally:
        print(f"Cleaning up environments for run {run_name}...")
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            try:
                eval_env.close()
            except AttributeError:
                pass
        print(f"--- Finished Run: {run_name} ---")

    return run_results

if __name__ == "__main__":
    # Command-line argument setup
    parser = argparse.ArgumentParser(description="Run PPO hyperparameter tuning for Fanuc Env.")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip running the baseline configuration in the first iteration.")
    parser.add_argument("--start_iteration", type=int, default=0, help="Iteration number to start from (loads previous results).")
    args = parser.parse_args()

    # CPU configuration for parallel processing
    cpu_count = multiprocessing.cpu_count()
    num_cpu = cpu_count
    print(f"Using {num_cpu} CPUs for parallel environments during tuning.")

    # Initial parameter setup
    initial_baseline_params = BASELINE_PARAMS.copy()
    current_baseline_params = initial_baseline_params
    best_params_overall = initial_baseline_params
    previous_baseline_reward = -np.inf

    # Load existing best parameters if starting from iteration 0
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
    total_runs_executed_this_session = 0

    # Calculate total tests per iteration
    total_tests_per_iteration = 1
    for _, (_, modifiers) in PARAMS_TO_TUNE.items():
        total_tests_per_iteration += len(modifiers)
    print(f"Calculated {total_tests_per_iteration} tests per iteration (1 baseline + variations).")

    # Iteration loop for tuning
    for iteration_count in range(args.start_iteration, MAX_ITERATIONS):
        print(f"\n=============== Starting Iteration {iteration_count} ===============")
        experiment_counter_this_iteration = 0

        # Results file paths for current and previous iterations
        iteration_results_file = f"ppo_tuning_results_iter_{iteration_count}.json"
        previous_iteration_results_file = f"ppo_tuning_results_iter_{iteration_count - 1}.json"

        # Load results from previous iteration if applicable
        all_prev_results = []
        previous_baseline_reward = -np.inf
        if iteration_count > 0:
            if os.path.exists(previous_iteration_results_file):
                try:
                    with open(previous_iteration_results_file, 'r') as f:
                        all_prev_results = json.load(f)
                        print(f"Loaded {len(all_prev_results)} results from previous iteration: {previous_iteration_results_file}")

                        prev_baseline_run = next((res for res in all_prev_results if res.get("run_name") == "baseline" and "error" not in res), None)
                        if prev_baseline_run:
                            previous_baseline_reward = prev_baseline_run.get("mean_reward", -np.inf)
                            print(f"Previous iteration baseline reward: {previous_baseline_reward:.2f}")
                        else:
                            print("Could not find baseline reward from previous iteration.")
                except Exception as e:
                    print(f"Warning: Could not load previous results from {previous_iteration_results_file}. Error: {e}")
                    print("Stopping due to error loading previous results.")
                    break
            else:
                print(f"Warning: Previous results file not found ({previous_iteration_results_file}). Cannot perform convergence check for this iteration.")
        else:
             print("First iteration. Using initial baseline parameters.")

        # Update baseline parameters based on previous results
        if iteration_count > 0 and all_prev_results:
            current_baseline_params = determine_best_params(all_prev_results, current_baseline_params)
        elif iteration_count == 0:
            pass

        print(f"\nCurrent Baseline Parameters for Iteration {iteration_count}:")
        for key, val in current_baseline_params.items():
            print(f"  {key}: {val}")

        current_iteration_results = []
        if os.path.exists(iteration_results_file):
            try:
                with open(iteration_results_file, 'r') as f:
                    current_iteration_results = json.load(f)
                print(f"Loaded {len(current_iteration_results)} existing results for Iteration {iteration_count} from {iteration_results_file}")
            except Exception as e:
                 print(f"Warning: Could not load existing results from {iteration_results_file}. Overwriting. Error: {e}")
                 current_iteration_results = []

        existing_run_names_this_iter = {res.get("run_name") for res in current_iteration_results}

        baseline_run_name = "baseline"
        current_baseline_reward = -np.inf
        baseline_run_completed_this_session = False

        baseline_existing_result = next((res for res in current_iteration_results if res.get("run_name") == baseline_run_name and "error" not in res), None)

        # Run baseline experiment if not skipped
        if baseline_existing_result:
            print(f"--- Found Existing Baseline Result for Iteration {iteration_count} --- ")
            current_baseline_reward = baseline_existing_result.get("mean_reward", -np.inf)
        else:
            skip_baseline_arg = args.skip_baseline and iteration_count == 0
            if not skip_baseline_arg:
                experiment_counter_this_iteration += 1
                total_runs_executed_this_session += 1
                print(f"--- Preparing Baseline for Iteration {iteration_count} --- ")
                baseline_results = run_tuning_experiment(
                    current_baseline_params,
                    baseline_run_name,
                    num_cpu,
                    iteration_count,
                    experiment_counter_this_iteration,
                    total_tests_per_iteration,
                    total_runs_executed_this_session
                )
                if baseline_results:
                    current_iteration_results.append(baseline_results)
                    existing_run_names_this_iter.add(baseline_run_name)
                    baseline_run_completed_this_session = True
                    if "error" not in baseline_results:
                         current_baseline_reward = baseline_results.get("mean_reward", -np.inf)
                    try:
                        with open(iteration_results_file, 'w') as f:
                            json.dump(current_iteration_results, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not save baseline results to {iteration_results_file}. Error: {e}")
            else:
                 print(f"--- Skipping Baseline run for Iteration {iteration_count} (--skip_baseline or already exists) --- ")

        # Check for convergence based on reward improvement
        if iteration_count > 0:
            print(f"\nConvergence Check: Current Baseline Reward = {current_baseline_reward:.2f}, Previous Baseline Reward = {previous_baseline_reward:.2f}")
            improvement = -np.inf
            if current_baseline_reward > -np.inf and previous_baseline_reward > -np.inf:
                 improvement = current_baseline_reward - previous_baseline_reward
            elif current_baseline_reward > -np.inf:
                 improvement = np.inf
            else:
                 improvement = 0.0

            if 0 <= improvement < CONVERGENCE_THRESHOLD:
                print(f"\nConvergence likely reached! Improvement ({improvement:.2f}) is non-negative and less than threshold ({CONVERGENCE_THRESHOLD}).")
                if not baseline_run_completed_this_session and not baseline_existing_result:
                     print("However, the baseline for the current iteration was skipped or failed this session. Running variations anyway.")
                elif previous_baseline_reward == -np.inf:
                     print("However, the previous baseline failed or was not found. Running variations anyway.")
                else:
                     print(f"Stopping after Iteration {iteration_count - 1}.")
                     break
            else:
                print(f"Improvement ({improvement:.2f}) meets threshold. Continuing tuning.")

        # Run parameter variation experiments
        for param_name, param_config in PARAMS_TO_TUNE.items():
            param_type, modifiers = param_config
            print(f"\n--- Iteration {iteration_count} / Preparing variations for Parameter: {param_name} ---")
            baseline_value_for_param = current_baseline_params[param_name]

            for modifier in modifiers:
                test_value = calculate_test_value(baseline_value_for_param, param_type, modifier)

                is_same_as_baseline = False
                if isinstance(test_value, (int, float)) and isinstance(baseline_value_for_param, (int, float)):
                    if np.isclose(float(test_value), float(baseline_value_for_param)):
                        is_same_as_baseline = True
                elif test_value == baseline_value_for_param:
                    is_same_as_baseline = True

                if is_same_as_baseline:
                    print(f"Skipping {param_name} modifier {modifier} -> {test_value} (same as current baseline: {baseline_value_for_param})")
                    continue

                run_name = create_run_name(param_name, param_type, modifier)

                if run_name in existing_run_names_this_iter:
                    print(f"Skipping run {run_name} (already run in this iteration)")
                    continue

                experiment_counter_this_iteration += 1
                total_runs_executed_this_session += 1
                run_params = current_baseline_params.copy()
                run_params[param_name] = test_value

                experiment_results = run_tuning_experiment(
                    run_params,
                    run_name,
                    num_cpu,
                    iteration_count,
                    experiment_counter_this_iteration,
                    total_tests_per_iteration,
                    total_runs_executed_this_session
                )
                if experiment_results:
                    current_iteration_results.append(experiment_results)
                    existing_run_names_this_iter.add(run_name)
                    try:
                        with open(iteration_results_file, 'w') as f:
                            json.dump(current_iteration_results, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not save intermediate results to {iteration_results_file}. Error: {e}")

        # Analyse iteration results and update best parameters
        print(f"\n--- Analysing Results for Iteration {iteration_count} --- ")
        print(f"Results for this iteration saved to {iteration_results_file}")

        if current_iteration_results:
             iteration_best_params = determine_best_params(current_iteration_results, current_baseline_params)

             print(f"Updating {BEST_PARAMS_FILE} with best parameters from Iteration {iteration_count}.")
             best_params_overall = iteration_best_params
             try:
                 with open(BEST_PARAMS_FILE, 'w') as f:
                     json.dump(best_params_overall, f, indent=4)
             except Exception as e:
                 print(f"Warning: Could not save best parameters to {BEST_PARAMS_FILE}. Error: {e}")

             current_baseline_params = best_params_overall.copy()
        else:
            print("No successful runs in this iteration to determine best parameters.")

    # Summarise final tuning results
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

                final_best_params = best_params_overall
                print("\nFinal Recommended Baseline Parameters (from latest iteration results):")
                for key, val in final_best_params.items():
                    print(f"  {key}: {val}")

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