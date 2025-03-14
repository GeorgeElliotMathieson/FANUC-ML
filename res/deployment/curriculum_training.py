# curriculum_training.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from update_rl_env import DomainRandomizedRLEnv

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Environment setup helpers
def make_env(rank, seed=0, randomize=True, task_difficulty=0):
    """
    Helper function to create an environment with proper settings
    
    Args:
        rank: process rank for multi-processing
        seed: random seed
        randomize: whether to use domain randomization
        task_difficulty: difficulty level (0-1) for curriculum learning
    """
    def _init():
        env = DomainRandomizedRLEnv(render=False, randomize=randomize)
        # Adjust workspace size based on difficulty
        env.workspace_size = 0.2 + 0.5 * task_difficulty
        # Set random seed
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Curriculum learning
def train_with_curriculum():
    n_envs = 4  # Number of parallel environments
    n_curriculum_stages = 5
    timesteps_per_stage = 200000
    
    # Start with easy tasks
    task_difficulty = 0.0
    
    # Initialize the model with the first difficulty level
    env = SubprocVecEnv([make_env(i, task_difficulty=task_difficulty) for i in range(n_envs)])
    
    # Create evaluation environment (no randomization for consistent evaluation)
    eval_env = DummyVecEnv([make_env(0, randomize=False, task_difficulty=1.0)])
    
    # Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    
    # Iterate through curriculum stages
    for stage in range(n_curriculum_stages):
        # Update difficulty
        task_difficulty = stage / (n_curriculum_stages - 1)
        print(f"Training on difficulty level: {task_difficulty:.2f}")
        
        # Create new environments with the updated difficulty
        env.close()
        env = SubprocVecEnv([make_env(i, task_difficulty=task_difficulty) for i in range(n_envs)])
        
        # Update the environment reference in the model
        model.set_env(env)
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=f"./models/stage_{stage}/",
            name_prefix="fanuc_model"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/stage_{stage}/best/",
            log_path=f"./logs/stage_{stage}/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train for this stage
        model.learn(
            total_timesteps=timesteps_per_stage,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=f"PPO_stage_{stage}"
        )
        
        # Save the model for this curriculum stage
        model.save(f"./models/fanuc_model_stage_{stage}")
    
    # Final evaluation
    print("Final evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save the final model
    model.save("./models/fanuc_final_model")
    
    # Clean up
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_with_curriculum()