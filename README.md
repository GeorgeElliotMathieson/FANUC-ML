# FANUC Robot Arm RL Training, Tuning, and Deployment

This project trains a FANUC LRMate 200iC robot arm simulation to reach random target positions within its workspace using Proximal Policy Optimisation (PPO), and provides scripts for deploying the trained model to a real robot.

**Key Features:**

*   **PyBullet Simulation:** Uses the PyBullet physics engine for efficient robot simulation.
*   **Obstacle Avoidance:** The environment dynamically places obstacles during training.
*   **Continuous Curriculum Learning:** Gradually increases difficulty to aid learning.
*   **Hyperparameter Tuning (Optuna):** Employs Optuna for optimising PPO hyperparameters.
*   **Stable Baselines3:** Leverages SB3 for PPO implementation.
*   **Time-Based Training & Tuning:** Duration is controlled by time limits.
*   **Modular Structure:** Codebase organized into `src` (with `rl` and `deployment` submodules), `scripts`, `config`, `assets`, and `output`.
*   **Automated Archiving:** Automatically archives older training runs from `output/ppo_logs/` to `archive/archived_ppo_logs/` to keep the output directory focused on recent runs.
*   **Real Robot Deployment:** Includes scripts (`scripts/deploy_real.py`) and modules (`src/deployment/`) for interfacing with and controlling a real FANUC robot (**requires significant user adaptation and safety verification**).
*   **Analysis Tools:** Includes a script to analyse training logs.

## Prerequisites

*   Python 3.8+ (3.8 - 3.10 recommended)
*   PyBullet physics simulator
*   PyTorch >= 2.0.0
*   Stable Baselines3[extra] >= 2.0.0
*   NumPy >= 1.22.0
*   Gymnasium >= 0.28.0 (replacement for OpenAI Gym)
*   Optuna >= 3.0.0
*   Pandas (for analysis script)
*   Matplotlib (for analysis script)
*   TensorBoard (for viewing logs and running analysis script)

## Installation

1.  **Clone the Project Repository:**
    ```bash
    # git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Obtain Robot Model Data:**
    Place the necessary URDF and mesh files for your specific FANUC robot model (e.g., LRMate 200iC) into the `assets/robot_model/` directory. Ensure the filenames and relative paths match those expected by the scripts (e.g., `assets/robot_model/urdf/Fanuc.urdf`).
    *(Example source: You might obtain these from a repository like [sezan92/Fanuc](https://github.com/sezan92/Fanuc), but only the model files are needed).* 
    ```bash
    # Example using the external repo (run in project root):
    # git clone https://github.com/sezan92/Fanuc.git temp_fanuc
    # mkdir -p assets/robot_model/urdf
    # # Move only necessary files (adjust based on repo structure)
    # mv temp_fanuc/urdf/Fanuc.urdf assets/robot_model/urdf/
    # mv temp_fanuc/meshes/* assets/robot_model/meshes/ # Create meshes dir if needed
    # rm -rf temp_fanuc
    # Ensure assets/robot_model/.git is removed or ignored if needed
    ```

3.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate (Linux/macOS: source venv/bin/activate | Windows: venv\Scripts\activate)
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```
.
├── archive/                   # Archived old runs and deprecated code
│   ├── archived_ppo_logs/     # Older PPO training runs (moved automatically by src/rl/train.py)
│   └── deprecated/
├── assets/                    # Static assets
│   └── robot_model/           # Robot model data (URDF, meshes)
├── config/                    # Configuration files
│   ├── best_params.json       # Best hyperparameters from tuning
│   ├── workspace_config.json  # Estimated workspace limits
│   ├── robot_config.py        # Central robot parameters (limits, etc.)
│   └── transfer_params.json   # (Optional) Sim-to-real calibration parameters
├── output/                    # Generated outputs from current runs
│   ├── analysis/              # Analysis plots/data
│   ├── optuna_study/          # Optuna study database/logs
│   └── ppo_logs/              # Recent PPO training runs (logs, models) - Older runs automatically moved to archive/
├── scripts/                   # Runnable scripts
│   ├── run_pipeline.py        # Main RL pipeline (tune->train->test)
│   ├── analyse_results.py     # Analyse RL training results
│   ├── check_workspace.py     # Utility: Estimate workspace
│   ├── joint_limit_demo.py    # Utility: Visualise joint limits
│   └── deploy_real.py         # **Script for real robot deployment**
├── src/                       # Core source code library
│   ├── __init__.py
│   ├── rl/                    # RL-related modules
│   │   ├── __init__.py
│   │   ├── fanuc_env.py       # RL Environment (Simulation)
│   │   ├── tune.py            # RL Hyperparameter Tuning
│   │   ├── train.py           # RL Agent Training (includes auto-archiving logic)
│   │   └── test.py            # RL Agent Visual Testing
│   └── deployment/            # **Real-robot deployment related modules**
│       ├── __init__.py
│       ├── robot_api.py       # Real Robot Communication API
│       └── transfer_learning.py # Sim-to-Real Model Adaptation
├── .gitignore                 # Git ignore file
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation file
```

## Usage

### Recommended Workflow: The RL Pipeline (`scripts/run_pipeline.py`)

Run the full simulation workflow (tune -> train -> test):
```bash
python scripts/run_pipeline.py [OPTIONS]
```
*Key Options:* `--tune_duration`, `--train_duration`, `--test_episodes`, `--skip_tuning`, etc. (See script help `python scripts/run_pipeline.py -h` for details).

### Running Individual RL Steps

Run steps independently (from project root directory):
*   **Tuning:** `python -m src.rl.tune [OPTIONS]`
*   **Training:** `python -m src.rl.train [OPTIONS]` (Note: This script also triggers automatic archiving of old runs from `output/ppo_logs/`)
*   **Visual Testing:** `python -m src.rl.test [OPTIONS]`
*(See script help `-h` for specific options)*

### Analysis Script (`scripts/analyse_results.py`)

Analyse the latest training run results:
```bash
python scripts/analyse_results.py [OPTIONS]
```
*(Options: `--log_dir`, `--params_file`, `--output_dir`, `--event_file`)*

### Utility Scripts

*   `scripts/check_workspace.py`: Estimates robot reach and saves to `config/workspace_config.json`.
*   `scripts/joint_limit_demo.py`: Visually demonstrates joint limits in simulation.

### **Real Robot Deployment (`scripts/deploy_real.py`)**

This script attempts to load a trained model and control a real FANUC robot.

** **CRITICAL WARNINGS** **
*   **ADAPTATION REQUIRED:** This script requires **SIGNIFICANT modification** to work with your specific robot hardware, network configuration, safety protocols, and task definition. Review the `TODO` comments within the script and associated modules (`src/deployment/robot_api.py`, `src/deployment/transfer_learning.py`).
*   **SAFETY FIRST:** Operating real robotic hardware carries inherent risks. **Implement and rigorously test ALL necessary safety measures** (physical barriers, emergency stops, velocity limits, collision checks, operator supervision) before running this script on a real robot. The provided safety checks are placeholders and likely insufficient on their own.
*   **VERIFY API:** Ensure the commands and parsing logic in `src/deployment/robot_api.py` match your controller's exact Socket Messaging interface.
*   **CALIBRATION:** Sim-to-real transfer typically requires calibration. The provided `RobotTransfer` class and calibration routines are basic placeholders.
*   **(Dependency Note):** PyBullet is still required for deployment, as it is used by `scripts/deploy_real.py` for Forward Kinematics (FK) calculations needed for safety checks and state estimation.

**(Conceptual Workflow):** The script outlines steps for connecting, optionally calibrating, running a control loop (get state, predict action, safety check, send command), handling targets, and managing potential errors. These steps require user implementation.

**Example Usage (Requires Adaptation & Extreme Caution):**
```bash
# VERIFY ALL PARAMETERS AND SAFETY BEFORE RUNNING
python scripts/deploy_real.py --robot_ip <YOUR_ROBOT_IP> --model_path <PATH_TO_MODEL.zip> [--calibrate] [--skip_safety]
```
*(See script help `-h` for options like `--loop_rate`, `--control_mode`)*

## Configuration Files

Located in the `config/` directory:
*   `best_params.json`: Stores the best hyperparameters found by Optuna tuning (`src/rl/tune.py`). Used by `src/rl/train.py` and `src/rl/test.py`.
*   `workspace_config.json`: Stores estimated simulation workspace limits (min/max reach, reach at Z midpoint) generated by `scripts/check_workspace.py`. Used by `src/rl/fanuc_env.py`.
*   `robot_config.py`: Central definition of core robot parameters (joint count, joint limits, velocity limits, link names) used across simulation and deployment code.
*   `transfer_params.json` (Optional): Intended to store calibration parameters for sim-to-real transfer, potentially generated by a calibration routine in `scripts/deploy_real.py` and used by `src/deployment/transfer_learning.py`. Expected keys might include `state_mean`, `state_std`, `action_scale`, `action_offset` (as NumPy arrays/lists).

## Environment Details (`src/rl/fanuc_env.py`)

*   **Action Space:** Continuous, 5 dimensions representing normalised target velocities for the 5 controllable joints (J1-J5). Values range from -1 to 1, scaled by velocity limits internally.
*   **Observation Space:** 21 dimensions:
    *   Joint positions (5)
    *   Joint velocities (5)
    *   Relative vector from end-effector (EE) to target (3)
    *   Normalised joint positions relative to limits (-1 to 1) (5)
    *   Relative vector from EE to nearest obstacle (3) (Zero vector if no obstacles)
*   **Reward Function:**
    *   Dense negative squared distance to target.
    *   Potential-based shaping reward for reducing the base rotation angle error towards the target azimuth (scaled by `angle_bonus_factor`).
    *   Large positive bonus (`+100`) for reaching the target within accuracy (`_target_accuracy`).
    *   Large negative penalty (`-100`) for colliding with an obstacle (terminates episode).
    *   *(Note: Self-collision penalty was present earlier but seems removed/inactive in the latest check; obstacle collision is the primary penalty)*
*   **Curriculum Learning:**
    *   Uses a **decreasing minimum radius** approach.
    *   The **maximum target radius remains fixed** throughout training (`fixed_max_target_radius`).
    *   Starts with the **minimum target radius equal to the maximum radius**, forcing initial targets to the outer edge of the workspace.
    *   Calculates the success rate over the last `success_rate_window_size` (20) episodes.
    *   If the success rate exceeds `success_rate_threshold` (0.75), the **`current_min_target_radius` is decreased** by `radius_decrease_step` (0.05m), down towards a final minimum value (`final_min_target_radius`, default: 10x base reach).
    *   The target position is randomly sampled within the `current_min_target_radius` and the `fixed_max_target_radius`.
*   **Obstacle Avoidance:**
    *   Places `num_obstacles` (currently 1) static sphere obstacles.
    *   Obstacles are randomly positioned within the workspace but outside safe zones near the robot base and the current target.
    *   Collision with an obstacle results in termination and a penalty.

## Hyperparameter Tuning (`src/rl/tune.py`)

*   Uses Optuna to find optimal hyperparameters for PPO.
*   The `objective` function defines the search space for parameters like `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, `angle_bonus_factor`, and network architecture size (`net_arch_size`).
*   Optimises based on the mean reward achieved during evaluation (`NUM_EVAL_EPISODES=30`) after a trial trains for `TUNE_TIMESTEPS`.
*   **Intermediate Evaluation & Pruning:** Implements an Optuna callback (`TrialCallback`) to perform evaluations every `EVAL_FREQ` (50k) steps *during* a trial. These intermediate results are reported to Optuna, allowing the configured pruner (`MedianPruner`) to stop unpromising trials early, saving significant tuning time.
*   **Default Storage:** Automatically saves study results to a default SQLite database (`output/optuna_study/default_fanuc_study.db`) unless overridden by the `--storage` argument. This allows for easy resuming of studies and use with visualization tools.
*   **Extended Logging:** Logs the standard deviation of the evaluation reward (`std_reward`) and the actual trial duration (`duration_seconds`) to the Optuna study's user attributes for more detailed analysis.
*   **Dashboard Compatibility:** The default SQLite storage (`output/optuna_study/default_fanuc_study.db`) makes the study directly compatible with the `optuna-dashboard` tool or IDE extensions. Simply point the dashboard tool/extension to the `.db` file.
*   Prunes trials early if `batch_size > n_steps`.

## Future Work / Improvements

*   Refine real-robot deployment script (`deploy_real.py`) with robust status checking and error handling.
*   Implement more sophisticated sim-to-real calibration techniques.
*   Implement more complex obstacle scenarios (multiple obstacles, moving obstacles).
*   Experiment with different RL algorithms (e.g., SAC for continuous control).
*   Refine reward shaping further.
*   Improve curriculum learning (e.g., adaptive step sizes, different metrics, switching strategies).
*   Add more sophisticated collision checking (e.g., checking all links, not just EE for self-collision penalty).
*   Investigate sim-to-real transfer possibilities.
*   Integrate inverse kinematics as an option or baseline.

## Acknowledgements

*   The FANUC LRMate 200iC URDF model is based on the work found at [https://github.com/sezan92/Fanuc.git](https://github.com/sezan92/Fanuc.git).
*   This project heavily utilises the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [Optuna](https://github.com/optuna/optuna) libraries.
*   Built using the [PyBullet](https://pybullet.org/) physics engine. 