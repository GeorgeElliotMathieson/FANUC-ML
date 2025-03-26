#!/usr/bin/env python3
# train_robot_rl_ppo_directml.py
# Training FANUC robot for end-effector positioning using PPO algorithm
# with DirectML support for AMD RX 6700S GPU

import os
import time
import math
import json
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import psutil
import warnings

# Import from existing modules to reuse functionality
from res.rml.python.train_robot_rl_positioning_revamped import (
    RobotPositioningRevampedEnv,
    CustomFeatureExtractor,
    visualize_target,
    JointLimitEnforcingEnv
)

from res.rml.python.train_robot_rl_positioning import (
    get_shared_pybullet_client, 
    FANUCRobotEnv, 
    load_workspace_data,
    determine_reachable_workspace,
    adjust_camera_for_robots
)

# Ensure output directories exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

# Configure DirectML for AMD GPU support
def setup_directml():
    """
    Configure TorchDynamo and DirectML backend for AMD GPU.
    This allows PyTorch to run on AMD GPUs.
    """
    try:
        import torch_directml
        dml = torch_directml.device()
        print(f"DirectML device initialized: {dml}")
        print(f"Using DirectML version: {torch_directml.__version__}")
        
        # Create a demo tensor to verify everything works
        test_tensor = torch.ones((2, 3), device=dml)
        print(f"Test tensor created on DirectML device: {test_tensor.device}")
        print("DirectML setup successful!")
        
        return dml
    except ImportError:
        print("ERROR: torch_directml package not found.")
        print("Please install it with: pip install torch-directml")
        print("Falling back to CPU...")
        return torch.device("cpu")
    except Exception as e:
        print(f"ERROR setting up DirectML: {e}")
        print("Falling back to CPU...")
        return torch.device("cpu")

# PPO implementation adapted for DirectML
class DirectMLPPO:
    """
    PPO implementation that works with DirectML on AMD GPUs
    """
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        tensorboard_log=None,
        verbose=0,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        
        # Initialize buffer for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Get environment spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Initialize actor and critic networks
        self.policy_network = self._build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {
            "iterations": [],
            "mean_rewards": [],
            "std_rewards": [],
            "mean_lengths": [],
            "distances": [],
            "successes": []
        }
    
    def _build_network(self):
        """
        Build the policy network for the PPO algorithm
        """
        # Create actor-critic network
        class ActorCritic(nn.Module):
            def __init__(self, observation_space, action_space):
                super().__init__()
                
                # Feature extractor
                self.feature_extractor = CustomFeatureExtractor(observation_space)
                feature_dim = self.feature_extractor.features_dim
                
                # Actor (policy) network
                n_actions = action_space.shape[0]
                self.actor = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_actions * 2)  # Mean and log_std
                )
                
                # Critic (value) network
                self.critic = nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
            
            def forward(self, obs):
                # Extract features
                features = self.feature_extractor(obs)
                
                # Actor output (action distribution parameters)
                actor_output = self.actor(features)
                n_actions = self.actor[-1].out_features // 2
                mu, log_std = torch.chunk(actor_output, 2, dim=-1)
                
                # Bound the log_std for numerical stability
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)
                
                # Create normal distribution
                dist = torch.distributions.Normal(mu, std)
                
                # Value estimate
                value = self.critic(features)
                
                return dist, value
            
            def get_action(self, obs, deterministic=False):
                dist, value = self(obs)
                
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
                
                # Clip the action to the action space bounds
                action = torch.clamp(action, -1.0, 1.0)
                
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                return action, log_prob, value
            
            def evaluate_actions(self, obs, actions):
                dist, value = self(obs)
                log_prob = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                return log_prob, value, entropy
        
        # Create the network and move to the specified device
        network = ActorCritic(self.observation_space, self.action_space).to(self.device)
        return network
    
    def collect_rollouts(self):
        """
        Collect a batch of experiences
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        episode_distances = []
        episode_successes = 0
        
        for step in range(self.n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy_network.get_action(state_tensor)
            
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy()
            value = value.cpu().numpy().flatten()
            
            # Execute action in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store trajectory data
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(terminated or truncated)
            
            # Update episode stats
            episode_reward += reward
            episode_length += 1
            
            # If episode ended
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Extract info about distance and success
                if 'distance' in info:
                    episode_distances.append(info['distance'])
                if 'success' in info and info['success']:
                    episode_successes += 1
                
                # Reset episode stats
                episode_reward = 0
                episode_length = 0
                
                # Reset the environment
                state, _ = self.env.reset()
            else:
                state = next_state
        
        # Compute statistics
        if episode_rewards:
            self.episode_rewards.extend(episode_rewards)
            self.episode_lengths.extend(episode_lengths)
            
            # Update training metrics
            self.training_metrics["iterations"].append(len(self.training_metrics["iterations"]) + 1)
            self.training_metrics["mean_rewards"].append(np.mean(episode_rewards))
            self.training_metrics["std_rewards"].append(np.std(episode_rewards))
            self.training_metrics["mean_lengths"].append(np.mean(episode_lengths))
            
            if episode_distances:
                self.training_metrics["distances"].append(np.mean(episode_distances))
            else:
                self.training_metrics["distances"].append(0.0)
                
            success_rate = episode_successes / len(episode_rewards) if episode_rewards else 0
            self.training_metrics["successes"].append(success_rate)
            
            if self.verbose > 0:
                print(f"Episodes: {len(episode_rewards)} | "
                      f"Mean reward: {np.mean(episode_rewards):.2f} | "
                      f"Mean length: {np.mean(episode_lengths):.2f} | "
                      f"Success rate: {success_rate:.2f} | "
                      f"Mean distance: {np.mean(episode_distances) if episode_distances else 0:.4f}m")
        
        return np.array(self.states), np.array(self.actions), np.array(self.rewards), \
               np.array(self.values), np.array(self.log_probs), np.array(self.dones)
    
    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        """
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        # Get last value estimate
        state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, next_value = self.policy_network.get_action(state_tensor)
        next_value = next_value.cpu().numpy()
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """
        Update policy using PPO loss
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # PPO update for n_epochs
        for epoch in range(self.n_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # Get new log probs and values
                new_log_probs, values, entropy = self.policy_network.evaluate_actions(batch_states, batch_actions)
                values = values.squeeze()
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped PPO loss
                clip_adv = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(ratio * batch_advantages, clip_adv).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def learn(self, total_timesteps, callback=None):
        """
        Main training loop
        """
        timesteps_elapsed = 0
        
        while timesteps_elapsed < total_timesteps:
            # Collect rollouts
            states, actions, rewards, values, old_log_probs, dones = self.collect_rollouts()
            timesteps_elapsed += self.n_steps
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards, values, dones)
            
            # Update policy
            self.update_policy(states, actions, old_log_probs, advantages, returns)
            
            # Call callback if provided
            if callback is not None:
                callback(self)
        
        return self
    
    def save(self, path):
        """
        Save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load a saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        print(f"Model loaded from {path}")
    
    def plot_training_progress(self, save_path=None):
        """
        Plot training progress metrics
        """
        if not self.training_metrics["iterations"]:
            print("No training data to plot yet")
            return
        
        plt.figure(figsize=(15, 12))
        
        # Plot rewards
        plt.subplot(3, 2, 1)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["mean_rewards"])
        plt.fill_between(
            self.training_metrics["iterations"],
            np.array(self.training_metrics["mean_rewards"]) - np.array(self.training_metrics["std_rewards"]),
            np.array(self.training_metrics["mean_rewards"]) + np.array(self.training_metrics["std_rewards"]),
            alpha=0.2
        )
        plt.title("Mean Episode Reward")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(3, 2, 2)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["mean_lengths"])
        plt.title("Mean Episode Length")
        plt.xlabel("Iteration")
        plt.ylabel("Steps")
        plt.grid(True)
        
        # Plot distances
        plt.subplot(3, 2, 3)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["distances"])
        plt.title("Mean Distance to Target")
        plt.xlabel("Iteration")
        plt.ylabel("Distance (m)")
        plt.grid(True)
        
        # Plot success rate
        plt.subplot(3, 2, 4)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["successes"])
        plt.title("Success Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training progress plot saved to: {save_path}")
        else:
            plt.show()

# Environment creation utility
def create_ppo_envs(num_envs=1, viz_speed=0.0, parallel_viz=False, eval_env=False):
    """
    Create environments for PPO training with visualization
    """
    envs = []
    
    for i in range(num_envs):
        # Calculate position offset for parallel visualization
        offset_x = 0.0
        if parallel_viz and num_envs > 1:
            offset_x = (i - (num_envs - 1) / 2) * 1.5
        
        # Determine if this env should render
        use_gui = viz_speed > 0.0 or eval_env
        
        # Create the environment
        env = RobotPositioningRevampedEnv(
            gui=use_gui,
            gui_delay=viz_speed,
            viz_speed=viz_speed,
            verbose=False,
            parallel_viz=parallel_viz,
            rank=i,
            offset_x=offset_x
        )
        
        # Wrap environment to enforce joint limits
        env = JointLimitEnforcingEnv(env)
        
        # Add monitoring wrapper
        log_dir = f"./logs/ppo_directml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        env = gym.wrappers.Monitor(
            env,
            log_dir + f"/env_{i}",
            video_callable=None if eval_env else False,
            force=True
        )
        
        envs.append(env)
    
    # If only one environment, return it directly
    if num_envs == 1:
        return envs[0]
    
    # Otherwise, create a vectorized environment
    return gym.vector.SyncVectorEnv(envs)

# Main training function
def train_robot_with_ppo_directml(args):
    """
    Train the robot using PPO algorithm with DirectML on AMD GPU
    """
    print("\n" + "="*80)
    print("FANUC Robot Positioning with PPO and DirectML")
    print("Optimized for AMD RX 6700S GPU")
    print("="*80 + "\n")
    
    # Setup DirectML device
    device = setup_directml() if args.gpu else torch.device("cpu")
    print(f"Using device: {device}\n")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_directml_{timestamp}"
    log_dir = f"./logs/{run_name}"
    model_dir = f"./models/{run_name}"
    plot_dir = f"./plots/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create environments
    print(f"Creating {args.parallel} parallel environments...")
    env = create_ppo_envs(
        num_envs=args.parallel,
        viz_speed=args.viz_speed,
        parallel_viz=True
    )
    
    # Create separate evaluation environment
    if args.eval_freq > 0:
        eval_env = create_ppo_envs(
            num_envs=1,
            viz_speed=args.viz_speed,
            eval_env=True
        )
    
    # Initialize model
    if args.load:
        print(f"Loading pre-trained model from {args.load}")
        model = DirectMLPPO(
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
            tensorboard_log=log_dir,
            verbose=args.verbose
        )
        model.load(args.load)
    else:
        print("Creating new PPO model")
        model = DirectMLPPO(
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
            tensorboard_log=log_dir,
            verbose=args.verbose
        )
    
    # Training callback for saving and evaluation
    def training_callback(model):
        # Current iteration
        iteration = len(model.training_metrics["iterations"])
        
        # Save model periodically
        if args.save_freq > 0 and iteration > 0 and iteration % args.save_freq == 0:
            save_path = f"{model_dir}/model_{iteration}.pt"
            model.save(save_path)
            
            # Plot training progress
            plot_path = f"{plot_dir}/training_progress_{iteration}.png"
            model.plot_training_progress(plot_path)
        
        # Evaluate model
        if args.eval_freq > 0 and iteration > 0 and iteration % args.eval_freq == 0:
            # TODO: Implement evaluation
            pass
    
    # Train the model
    if not args.eval_only:
        print(f"Starting training for {args.steps} timesteps...")
        model.learn(total_timesteps=args.steps, callback=training_callback)
        
        # Save final model
        final_model_path = f"{model_dir}/final_model.pt"
        model.save(final_model_path)
        
        # Plot final training progress
        final_plot_path = f"{plot_dir}/final_training_progress.png"
        model.plot_training_progress(final_plot_path)
        
        print(f"Training completed! Model saved to {final_model_path}")
    
    # Evaluation
    if args.eval_only or args.final_eval:
        print("Running evaluation...")
        # TODO: Implement more sophisticated evaluation
    
    return model, env

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train FANUC robot using PPO with DirectML on AMD GPU")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency in timesteps")
    parser.add_argument("--save-freq", type=int, default=50000, help="Model saving frequency in timesteps")
    parser.add_argument("--parallel", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--load", type=str, default=None, help="Path to pre-trained model to continue training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on a pre-trained model")
    parser.add_argument("--final-eval", action="store_true", help="Run final evaluation after training")
    parser.add_argument("--viz-speed", type=float, default=0.0, help="Visualization speed (delay in seconds)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="Disable GPU usage")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable GPU usage")
    
    # PPO specific parameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for updates")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of optimization epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    
    args = parser.parse_args()
    
    # Train the model
    model, env = train_robot_with_ppo_directml(args)
    
    # Close environments
    env.close()

if __name__ == "__main__":
    main() 