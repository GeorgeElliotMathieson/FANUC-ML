import os
import multiprocessing
import time
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the custom environment
from fanuc_env import FanucEnv

# Define a function to create the environment. This is needed for parallelisation.
def make_env(rank, seed=0):
    """Utility function for multiprocessed env creation.

    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Use a different seed for each environment
        env = FanucEnv(render_mode=None) # No rendering during parallel training
        # Note: Proper seeding would involve env.reset(seed=seed + rank)
        # but Gymnasium's API might handle this slightly differently now.
        # For simplicity, relying on initial random state differences for now.
        # Consider adding explicit seeding later if needed.
        return env
    # set_global_seeds(seed + rank) # Deprecated in SB3
    return _init

if __name__ == "__main__":
    # --- Hyperparameters ---
    # Determine the number of CPUs available, leave one free for system processes
    cpu_count = multiprocessing.cpu_count()
    num_cpu = max(1, cpu_count - 1) # Number of parallel environments
    # num_cpu = 4 # Or set manually

    total_timesteps = 200_000 # Adjust as needed, 1M is a starting point
    learning_rate = 3e-4
    n_steps = 2048       # Steps per environment per PPO update
    batch_size = 64        # Minibatch size for PPO updates
    n_epochs = 10          # Number of epochs when optimizing the surrogate loss
    gamma = 0.99         # Discount factor
    gae_lambda = 0.95    # Factor for Generalized Advantage Estimation
    clip_range = 0.2       # Clipping parameter for PPO
    ent_coef = 0.0         # Entropy coefficient (can try 0.01)
    vf_coef = 0.5        # Value function coefficient
    max_grad_norm = 0.5    # Max gradient norm for clipping

    save_freq = 50_000     # Save a checkpoint every N steps
    log_dir = "./ppo_fanuc_logs/"
    save_path = os.path.join(log_dir, "ppo_fanuc_model") # Base path for saving model
    os.makedirs(log_dir, exist_ok=True)

    # --- Environment Setup ---
    print(f"Using {num_cpu} parallel environments.")

    # Create the vectorized environment using SubprocVecEnv for true parallelism
    # If SubprocVecEnv causes issues (e.g., on Windows without proper freezing),
    # fallback to DummyVecEnv (runs environments sequentially in one process).
    vec_env_cls = SubprocVecEnv if num_cpu > 1 else DummyVecEnv
    train_env = make_vec_env(lambda: FanucEnv(render_mode=None), n_envs=num_cpu, vec_env_cls=vec_env_cls)

    # --- Agent and Training ---
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define the PPO model
    # Policy network defaults: MlpPolicy (Multi-Layer Perceptron)
    # Network architecture can be customised via policy_kwargs
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1, # Print training progress
        tensorboard_log=log_dir,
        device=device
    )

    # --- Callbacks ---
    # Save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // num_cpu, 1), # Adjust freq based on num envs
        save_path=log_dir,
        name_prefix="rl_model"
    )

    # --- Train the agent ---
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=1 # Log every update
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- Save the final model ---
        print("Training finished. Saving final model...")
        model.save(save_path)
        train_env.close() # Important to close the parallel environments
        end_time = time.time()
        print(f"Model saved to {save_path}.zip")
        print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

    # --- Optional: Load and test the trained model ---
    # print("Loading trained model for testing...")
    # loaded_model = PPO.load(save_path)

    # # Create a single test environment with rendering
    # test_env = FanucEnv(render_mode='human')
    # obs, _ = test_env.reset()
    # print("Testing model...")
    # for _ in range(5000): # Test for a fixed number of steps
    #     action, _states = loaded_model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = test_env.step(action)
    #     test_env.render()
    #     if terminated or truncated:
    #         print("Test episode finished. Resetting.")
    #         obs, _ = test_env.reset()
    #         time.sleep(1) # Pause before next episode

    # test_env.close()
    print("Script finished.") 