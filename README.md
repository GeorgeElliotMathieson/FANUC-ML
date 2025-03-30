# FANUC Robot Arm PPO Training with Obstacles and Tuning

This project trains a FANUC LRMate 200iC robot arm to reach random target positions within its workspace using Proximal Policy Optimisation (PPO). The training incorporates several advanced features:

*   **Obstacle Avoidance:** The environment can dynamically place obstacles, requiring the agent to learn collision-free paths.
*   **Continuous Curriculum Learning:** The difficulty (target reach distance) gradually increases as the agent's success rate improves.
*   **Hyperparameter Tuning (Optuna):** Uses the Optuna library for efficient Bayesian optimisation of PPO hyperparameters.
*   **Time-Based Training:** Training duration is controlled by a time limit rather than a fixed number of steps.
*   **Parallel Environments:** Leverages multiple CPU cores for faster training.
*   **PyBullet Simulation:** Uses the PyBullet physics engine for simulation.

The core logic focuses on learning forward kinematics without relying on pre-calculated inverse kinematics solutions.

## Prerequisites

*   Python 3.8+
*   PyBullet physics simulator
*   PyTorch
*   Stable Baselines3 reinforcement learning library
*   NumPy
*   Gymnasium (formerly OpenAI Gym)
*   Optuna

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    # This step might have been done already.
    # git clone <your-repo-url> # Or wherever this code resides
    # cd <your-repo-directory>
    ```

2.  **Clone the FANUC robot model repository (Required):**
    *(This assumes the 'Fanuc' directory containing the robot model will be placed in the same root directory as this project's files)*
    ```bash
    # If you don't have the Fanuc model yet:
    git clone https://github.com/sezan92/Fanuc.git
    ```
    *Note: The original environment from the Fanuc repository seems ROS-based. This implementation uses PyBullet with the provided URDF for simplicity. ROS integration is not included here.*

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

The typical workflow involves three main steps: Tuning, Training, and Testing.

1.  **Step 1: Hyperparameter Tuning (Optuna)**
    *   Run the tuning script to find optimal hyperparameters using Optuna.
    *   Specify the duration for the tuning process.
    *   The best parameters found will be saved to `best_params.json`.
    ```bash
    # Example: Tune for 10 minutes (600 seconds)
    python ppo_tune.py --duration 600 --study_name "ppo_fanuc_study_1"
    ```
    *   `--duration`: Tuning time in seconds.
    *   `--study_name`: A unique name for the Optuna study (allows resuming).
    *   `--storage`: Optional database URL (e.g., `sqlite:///optuna_study.db`) to store and resume studies across runs. Defaults to in-memory.

2.  **Step 2: Training the Agent**
    *   Run the training script using the best parameters found during tuning (loaded automatically from `best_params.json`).
    *   Specify the desired training duration.
    *   The final trained model will be saved to `./ppo_fanuc_logs/ppo_fanuc_model.zip`.
    ```bash
    # Example: Train for 30 minutes (1800 seconds)
    python ppo_train.py --duration 1800
    ```
    *   `--duration`: Training time in seconds.

3.  **Step 3: Visual Testing**
    *   Run the visual test script to observe the trained agent's performance in the rendered environment.
    *   Loads the latest saved model automatically.
    ```bash
    # Example: Run 10 test episodes
    python ppo_visual_test.py --episodes 10
    ```
    *   `--episodes`: Number of visual test episodes.
    *   `--model_path`: Optionally specify a path to a specific model `.zip` file.

## Additional Scripts

*   **`check_workspace.py`:** Estimates the robot's minimum and maximum reach by sampling random joint configurations. Saves results to `workspace_config.json`, which is used by the environment. Run this if you modify the robot model or suspect reach issues.
    ```bash
    python check_workspace.py
    ```
*   **`joint_limit_demo.py`:** Visually demonstrates the movement range of each controllable joint based on the limits defined in the environment. Useful for verifying joint limits.
    ```bash
    python joint_limit_demo.py
    ```

## Project Structure

*   `ppo_tune.py`: Script for hyperparameter tuning using Optuna.
*   `ppo_train.py`: Main script for training the PPO agent using tuned parameters.
*   `ppo_visual_test.py`: Script for visually testing the trained agent.
*   `fanuc_env.py`: Custom Gymnasium environment (simulation, rewards, curriculum, obstacles).
*   `check_workspace.py`: Utility script to estimate robot workspace limits.
*   `joint_limit_demo.py`: Utility script to visualise joint limits.
*   `requirements.txt`: Python package dependencies.
*   `README.md`: This file.
*   `best_params.json`: Stores the best hyperparameters found by Optuna (created after tuning).
*   `workspace_config.json`: Stores estimated workspace limits (created by `check_workspace.py`).
*   `ppo_fanuc_logs/`: Directory for saving trained models and TensorBoard logs.
*   `optuna_logs/`: Directory for saving detailed Optuna trial logs (optional).
*   `Fanuc/`: Directory containing the robot model (URDF, meshes, etc.) - cloned from [https://github.com/sezan92/Fanuc.git](https://github.com/sezan92/Fanuc.git).

## Customisation

*   **Environment Parameters:** Modify `fanuc_env.py` to change:
    *   Reward shaping (e.g., `angle_bonus_factor`).
    *   Curriculum settings (`initial_max_target_radius`, `success_rate_threshold`, etc.).
    *   Obstacle properties (`num_obstacles`, `obstacle_radius`, placement logic).
    *   Simulation settings (max steps, accuracy).
*   **Tuning Parameters:** Adjust `ppo_tune.py`:
    *   `TUNE_TIMESTEPS`: Timesteps per Optuna trial.
    *   Hyperparameter search spaces within the `objective` function.
*   **Training:** Modify `ppo_train.py`:
    *   Checkpoint save frequency (`save_freq`).
*   **URDF/Robot Model:** Changes to the `Fanuc/urdf/Fanuc.urdf` may require re-running `check_workspace.py` and potentially adjusting joint indices/limits in `fanuc_env.py`. 