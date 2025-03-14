# train_rl.py
import os
import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
import time

from rl_env import FANUCRLEnv

# Create output directory
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Create and vectorize the environment
def make_env():
    return FANUCRLEnv(render=False)  # No rendering during training for speed

env = DummyVecEnv([make_env])

# Define the model
# PPO is a good starting point for robotic control tasks
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

# Create callback for saving models
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path="./models/",
    name_prefix="fanuc_model"
)

# Train the model
total_timesteps = 1000000  # Adjust based on complexity and available time
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    tb_log_name="PPO"
)

# Save the final model
model.save("./models/fanuc_final_model")

# Evaluate the model
print("Evaluating model...")
mean_reward, std_reward = evaluate_policy(model, make_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test the model
print("Testing model in environment...")
env = make_env()
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)  # Small delay for visualization
    print(f"Distance to target: {info['distance']:.4f}, Reward: {reward:.4f}")

env.close()