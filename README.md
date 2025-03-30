# FANUC Robot Arm PPO Training

This project trains a FANUC LRMate 200iD robot arm to reach random target positions within its workspace using Proximal Policy Optimisation (PPO) with parallel environments. The training focuses on learning forward kinematics without relying on inverse kinematics solutions.

## Prerequisites

*   Python 3.8+
*   PyBullet physics simulator
*   PyTorch
*   Stable Baselines3 reinforcement learning library
*   NumPy
*   Gymnasium (formerly OpenAI Gym)

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    # This step might have been done already.
    # git clone https://github.com/your-username/FANUC-ML.git # Or wherever this code resides
    # cd FANUC-ML
    ```

2.  **Clone the FANUC robot model repository:**
    *(This assumes the 'Fanuc' directory containing the robot model is in the same root directory as this project's files)*
    ```bash
    # If you don't have it yet:
    # git clone https://github.com/sezan92/Fanuc.git
    ```
    *Note: The original environment seems ROS-based. This implementation uses PyBullet with the provided URDF for simplicity. ROS integration is not included here.*

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure the virtual environment is activated.**
2.  **Run the training script:**
    ```bash
    python ppo_train.py
    ```
3.  The script will initialise the parallel environments, load the PPO agent, and start the training process. Training progress (mean reward, episode length, etc.) will be printed to the console.
4.  Once training is complete, the trained model will be saved as `ppo_fanuc_model.zip`.

## Project Structure

*   `ppo_train.py`: Main script for setting up and running the PPO training.
*   `fanuc_env.py`: Custom Gymnasium environment for the FANUC robot simulation using PyBullet.
*   `requirements.txt`: Python package dependencies.
*   `README.md`: This file.
*   `Fanuc/`: Directory containing the robot model (URDF, meshes, etc.) - cloned from [https://github.com/sezan92/Fanuc.git](https://github.com/sezan92/Fanuc.git).

## Customisation

*   **Hyperparameters:** Adjust PPO hyperparameters (learning rate, batch size, number of environments, etc.) in `ppo_train.py`.
*   **Reward Function:** Modify the reward shaping in `fanuc_env.py` to potentially improve learning speed or final performance.
*   **Environment:** Change simulation parameters, target generation logic, or observation/action spaces in `fanuc_env.py`. 