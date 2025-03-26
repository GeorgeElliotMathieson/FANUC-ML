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
import gc

# Configure CPU threading for PyTorch
# Use all available cores for best performance
num_threads = psutil.cpu_count(logical=True)
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
print(f"Using {num_threads} CPU threads for PyTorch")

# Report system memory 
mem = psutil.virtual_memory()
print(f"System memory: {mem.total / (1024.0**3):.1f} GB total, {mem.available / (1024.0**3):.1f} GB available")

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
        
        # Check version if available
        try:
            print(f"Using DirectML version: {torch_directml.__version__}")
        except AttributeError:
            print("DirectML version information not available")
        
        # Create a demo tensor to verify everything works
        test_tensor = torch.ones((2, 3), device=dml)
        print(f"Test tensor created on DirectML device: {test_tensor.device}")
        print("DirectML setup successful!")
        
        # Enable DirectML optimizations if available
        try:
            # Check if enable_optimizations attribute exists
            if hasattr(torch_directml, 'enable_optimizations'):
                torch_directml.enable_optimizations(True)
                print("DirectML optimizations enabled")
            else:
                print("DirectML optimizations not available in this version")
        except Exception as e:
            print(f"Could not enable DirectML optimizations: {e}")
        
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
        
        # Initialize buffer for storing trajectories - using optimized numpy arrays
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=getattr(env, 'num_envs', 1)
        )
        
        # Get environment spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Initialize actor and critic networks
        self.policy_network = self._build_network()
        
        # Use AMSGrad for better convergence
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=learning_rate,
            eps=1e-5,  # Prevent division by zero
            amsgrad=True  # Better stability with AMSGrad
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,  # Reset every 100 iterations
            eta_min=learning_rate * 0.1  # Minimum learning rate = 10% of initial
        )
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {
            "iterations": [],
            "mean_rewards": [],
            "std_rewards": [],
            "mean_lengths": [],
            "distances": [],
            "successes": [],
            "eval_rewards": [],
            "eval_lengths": [],
            "eval_distances": [],
            "eval_successes": []
        }
        
        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()
        self.processed_steps = 0
    
    def _build_network(self):
        """
        Build the policy network for the PPO algorithm with optimizations
        for better GPU utilization and performance.
        """
        # Create actor-critic network
        class ActorCritic(nn.Module):
            def __init__(self, observation_space, action_space, device="cpu"):
                super().__init__()
                
                self.device = device
                self.is_gpu = str(device) != "cpu"
                
                # Register buffer for action scaling (more efficient than recreating tensors)
                self.register_buffer("action_scale", torch.tensor(1.0))
                
                # Feature extractor with performance optimizations
                if hasattr(observation_space, 'shape'):
                    self.feature_extractor = CustomFeatureExtractor(observation_space)
                    feature_dim = self.feature_extractor.features_dim
                else:
                    # Fallback for strange observation spaces
                    feature_dim = 64
                    print(f"Warning: Using fallback feature dimension {feature_dim}")
                
                # Use network architecture optimized for GPU execution
                # - Use layers that are optimized for GPU computation
                # - Use explicit batch dimensions for better utilization
                
                # Actor (policy) network with shared trunk for better parameter sharing
                self.shared_trunk = nn.Sequential(
                    nn.LayerNorm(feature_dim),  # Normalization for better training stability
                    nn.Linear(feature_dim, 256),
                    nn.LeakyReLU(0.01),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(0.01),
                )
                
                # Action mean and std branches
                n_actions = action_space.shape[0]
                self.action_mean = nn.Linear(256, n_actions)
                self.action_log_std = nn.Linear(256, n_actions)
                
                # Critic (value) network - also using the shared trunk
                self.value_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.01),
                    nn.Linear(128, 1)
                )
                
                # Initialize weights for better training
                self._init_weights()
                
                # Script critical parts of the network for speed if JIT is available
                if hasattr(torch, 'jit') and not str(device).startswith('privateuseone'):
                    try:
                        # JIT scripting can dramatically improve performance 
                        self.get_value = torch.jit.script(self._get_value)
                        print("JIT scripting enabled for performance")
                    except Exception as e:
                        print(f"JIT scripting failed: {e}, using standard execution")
            
            def _init_weights(self):
                """Initialize weights for more stable training"""
                # Initialize shared trunk
                for layer in self.shared_trunk:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=1.0)
                        nn.init.constant_(layer.bias, 0.0)
                
                # Initialize actor output heads
                nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
                nn.init.constant_(self.action_mean.bias, 0.0)
                nn.init.orthogonal_(self.action_log_std.weight, gain=0.01)
                nn.init.constant_(self.action_log_std.bias, -0.5)  # Start with smaller variance
                
                # Initialize critic
                for layer in self.value_head:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=1.0)
                        nn.init.constant_(layer.bias, 0.0)
            
            def _get_value(self, features):
                """Optimized value calculation using shared features"""
                return self.value_head(features)
            
            def forward(self, obs):
                """
                Forward pass with optimized memory access patterns and compute
                utilization for better GPU performance.
                """
                # Extract features
                with torch.no_grad() if not self.training else torch.enable_grad():
                    features = self.feature_extractor(obs)
                
                # Shared forward pass through trunk
                shared_features = self.shared_trunk(features)
                
                # Actor outputs (action distribution parameters)
                mu = self.action_mean(shared_features)
                log_std = self.action_log_std(shared_features)
                
                # Bound the log_std for numerical stability
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)
                
                # Create normal distribution
                dist = torch.distributions.Normal(mu, std)
                
                # Value estimate - use the separate function for potential JIT optimization
                if hasattr(self, 'get_value'):
                    value = self.get_value(shared_features)
                else:
                    value = self.value_head(shared_features)
                
                return dist, value
            
            def get_action(self, obs, deterministic=False):
                """
                Get action with optimized sampling and processing
                for better GPU utilization.
                """
                # Compute distribution and value 
                with torch.no_grad():  # Save memory by not tracking gradients here
                    dist, value = self(obs)
                
                if deterministic:
                    action = dist.mean
                else:
                    # Replace dist.sample() with reparameterization trick to avoid torch.normal
                    # which falls back to CPU with DirectML
                    if self.is_gpu:
                        # GPU optimization - avoid torch.normal CPU fallback
                        eps = torch.rand_like(dist.scale, device=dist.scale.device) * 2 - 1  # Uniform(-1, 1)
                        action = dist.mean + dist.scale * eps  # Approximation of normal sampling
                    else:
                        # On CPU we can use the default sampling
                        try:
                            action = dist.sample()
                        except:
                            # Fallback for any sampling issues
                            eps = torch.rand_like(dist.scale) * 2 - 1
                            action = dist.mean + dist.scale * eps
                
                # Clip the action to the action space bounds
                action = torch.clamp(action, -1.0, 1.0)
                
                # Handle batched observations
                if len(obs.shape) > 1 and obs.shape[0] > 1:
                    # For vectorized environments, sum log_probs across action dimensions
                    # but keep batch dimension intact
                    log_prob = dist.log_prob(action).sum(dim=-1)
                else:
                    # Original behavior for single environments
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                return action, log_prob, value
            
            def evaluate_actions(self, obs, actions):
                """
                Evaluate actions with memory efficiency optimizations
                for better GPU utilization.
                """
                dist, value = self(obs)
                
                # Compute log probability
                log_prob = dist.log_prob(actions).sum(dim=-1)
                
                # Compute entropy
                entropy = dist.entropy().mean()
                
                # Ensure value has the correct shape for loss calculation
                # This should be a 1D tensor with batch_size elements
                value = value.view(-1)
                
                return log_prob, value, entropy
        
        # Create the network and move to the specified device
        network = ActorCritic(self.observation_space, self.action_space, device=self.device).to(self.device)
        return network
    
    def collect_rollouts(self):
        """
        Collect a batch of experiences using optimized buffer
        """
        # Reset buffer
        self.rollout_buffer.reset()
        
        # Handle both regular environments and vectorized environments
        is_vector_env = hasattr(self.env, 'num_envs') and self.env.num_envs > 1
        
        # Store initial stats
        start_time = time.time()
        
        if is_vector_env:
            # For vectorized environments
            num_envs = self.env.num_envs
            observations = self.env.reset()
            
            # Vectors for episode tracking
            episode_rewards = [0] * num_envs
            episode_lengths = [0] * num_envs
            all_episode_rewards = []
            all_episode_lengths = []
            all_episode_distances = []
            episode_successes = 0
            
            # Display current iteration
            current_iteration = len(self.training_metrics["iterations"]) + 1
            print(f"\n--- Starting iteration {current_iteration} with {num_envs} parallel robots ---")
            
            # Pre-allocate buffers for better memory efficiency
            last_observations = observations.copy()
            
            for step in range(self.n_steps // num_envs):
                # Process each environment's state
                observations_tensor = torch.FloatTensor(observations).to(self.device)
                
                # Get actions from policy (using no_grad for memory efficiency)
                with torch.no_grad():
                    actions, log_probs, values = self.policy_network.get_action(observations_tensor)
                
                # Convert to numpy and execute actions
                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy().flatten()
                
                # Step environments
                next_observations, rewards, dones, infos = self.env.step(actions_np)
                
                # Store each env transition in buffer
                for env_idx in range(num_envs):
                    # Update episode stats
                    episode_rewards[env_idx] += rewards[env_idx]
                    episode_lengths[env_idx] += 1
                    
                    # Store in buffer
                    self.rollout_buffer.add(
                        obs=last_observations[env_idx],
                        action=actions_np[env_idx],
                        reward=rewards[env_idx],
                        done=dones[env_idx],
                        value=values_np[env_idx],
                        log_prob=log_probs_np[env_idx],
                    )
                    
                    # Track completed episodes
                    if dones[env_idx]:
                        all_episode_rewards.append(episode_rewards[env_idx])
                        all_episode_lengths.append(episode_lengths[env_idx])
                        
                        # Extract episode metrics
                        if 'distance' in infos[env_idx]:
                            all_episode_distances.append(infos[env_idx]['distance'])
                        if 'success' in infos[env_idx] and infos[env_idx]['success']:
                            episode_successes += 1
                        
                        # Reset episode stats
                        episode_rewards[env_idx] = 0
                        episode_lengths[env_idx] = 0
                
                # Save observations for next iteration
                last_observations = next_observations.copy()
                observations = next_observations
            
            # Compute final value estimate for incomplete episodes
            final_obs_tensor = torch.FloatTensor(observations).to(self.device)
            with torch.no_grad():
                _, _, final_values = self.policy_network.get_action(final_obs_tensor)
                final_values = final_values.cpu().numpy().flatten()
            
            # Store last value for advantage calculation
            self.rollout_buffer.compute_returns_and_advantages(last_values=final_values)
            
            # Calculate statistics
            if all_episode_rewards:
                self.episode_rewards.extend(all_episode_rewards)
                self.episode_lengths.extend(all_episode_lengths)
                
                # Update training metrics
                self.training_metrics["iterations"].append(len(self.training_metrics["iterations"]) + 1)
                self.training_metrics["mean_rewards"].append(np.mean(all_episode_rewards))
                self.training_metrics["std_rewards"].append(np.std(all_episode_rewards) if len(all_episode_rewards) > 1 else 0)
                self.training_metrics["mean_lengths"].append(np.mean(all_episode_lengths))
                
                # Store distance and success metrics
                if all_episode_distances:
                    self.training_metrics["distances"].append(np.mean(all_episode_distances))
                else:
                    self.training_metrics["distances"].append(0.0)
                
                success_rate = episode_successes / len(all_episode_rewards) if all_episode_rewards else 0
                self.training_metrics["successes"].append(success_rate)
                
                # Track FPS
                elapsed = time.time() - start_time
                steps_completed = self.n_steps // num_envs * num_envs
                fps = steps_completed / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                
                # Report metrics
                if self.verbose > 0:
                    print(f"Episodes: {len(all_episode_rewards)} | "
                          f"Mean reward: {np.mean(all_episode_rewards):.2f} | "
                          f"Mean length: {np.mean(all_episode_lengths):.2f} | "
                          f"Success rate: {success_rate:.2f} | "
                          f"Mean distance: {np.mean(all_episode_distances) if all_episode_distances else 0:.4f}m | "
                          f"FPS: {fps:.1f}")
        
        else:
            # Single environment case 
            observation, _ = self.env.reset()
            
            # Track episode stats
            episode_reward = 0
            episode_length = 0
            episode_rewards = []
            episode_lengths = []
            episode_distances = []
            episode_successes = 0
            
            # Display current iteration
            current_iteration = len(self.training_metrics["iterations"]) + 1
            print(f"\n--- Starting iteration {current_iteration} ---")
            
            # Store initial observation
            last_observation = observation
            
            for step in range(self.n_steps):
                # Convert observation to tensor for the policy network
                observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                
                # Get action from policy
                with torch.no_grad():
                    action, log_prob, value = self.policy_network.get_action(observation_tensor)
                
                # Convert to numpy for environment step
                action_np = action.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy().flatten()
                
                # Make sure action is properly shaped for the environment (flattened)
                action_step = action_np.flatten()
                
                # Execute action in environment
                next_observation, reward, terminated, truncated, info = self.env.step(action_step)
                done = terminated or truncated
                
                # Store in buffer
                self.rollout_buffer.add(
                    obs=last_observation,
                    action=action_np.reshape(-1),
                    reward=reward,
                    done=done,
                    value=value_np.item(),
                    log_prob=log_prob_np.item(),
                )
                
                # Update episode stats
                episode_reward += reward
                episode_length += 1
                
                # If episode ended
                if done:
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
                    next_observation, _ = self.env.reset()
                
                # Save for next iteration
                last_observation = observation
                observation = next_observation
            
            # Get final value estimate for incomplete episode
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, final_value = self.policy_network.get_action(observation_tensor)
                final_value = final_value.cpu().numpy().item()
            
            # Complete returns and advantages calculation
            self.rollout_buffer.compute_returns_and_advantages(last_values=np.array([final_value]))
            
            # Compute and store statistics
            if episode_rewards:
                self.episode_rewards.extend(episode_rewards)
                self.episode_lengths.extend(episode_lengths)
                
                # Update training metrics
                self.training_metrics["iterations"].append(len(self.training_metrics["iterations"]) + 1)
                self.training_metrics["mean_rewards"].append(np.mean(episode_rewards))
                self.training_metrics["std_rewards"].append(np.std(episode_rewards) if len(episode_rewards) > 1 else 0)
                self.training_metrics["mean_lengths"].append(np.mean(episode_lengths))
                
                if episode_distances:
                    self.training_metrics["distances"].append(np.mean(episode_distances))
                else:
                    self.training_metrics["distances"].append(0.0)
                    
                success_rate = episode_successes / len(episode_rewards) if episode_rewards else 0
                self.training_metrics["successes"].append(success_rate)
                
                # Track FPS
                elapsed = time.time() - start_time
                fps = self.n_steps / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                
                # Report metrics
                if self.verbose > 0:
                    print(f"Episodes: {len(episode_rewards)} | "
                         f"Mean reward: {np.mean(episode_rewards):.2f} | "
                         f"Mean length: {np.mean(episode_lengths):.2f} | "
                         f"Success rate: {success_rate:.2f} | "
                         f"Mean distance: {np.mean(episode_distances) if episode_distances else 0:.4f}m | "
                         f"FPS: {fps:.1f}")
        
        # Return buffer
        return self.rollout_buffer
    
    def update_policy(self, rollout_buffer):
        """
        Update policy using PPO loss with data from rollout buffer
        """
        # Determine if we can use mixed precision 
        use_mixed_precision = str(self.device).startswith('privateuseone')
        
        # Temporarily disable mixed precision until feature extractor is fixed
        use_mixed_precision = False
        
        # Start timing
        update_start_time = time.time()
        
        # Get samples from buffer
        observations, actions, old_values, old_log_probs, advantages, returns = rollout_buffer.get_samples()
        
        # Convert to tensors with optimized memory handling
        is_gpu = str(self.device) != "cpu"
        
        if is_gpu:
            # Try to optimize GPU transfer with pinned memory
            try:
                # Pin memory for faster transfer to GPU
                observations = torch.from_numpy(observations).pin_memory().to(self.device, non_blocking=True)
                actions = torch.from_numpy(actions).pin_memory().to(self.device, non_blocking=True)
                old_log_probs = torch.from_numpy(old_log_probs).pin_memory().to(self.device, non_blocking=True)
                advantages = torch.from_numpy(advantages).pin_memory().to(self.device, non_blocking=True)
                returns = torch.from_numpy(returns).pin_memory().to(self.device, non_blocking=True)
            except:
                # Fall back to normal transfer if pinned memory fails
                observations = torch.FloatTensor(observations).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
        else:
            # Standard CPU tensors
            observations = torch.FloatTensor(observations).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
        
        # If using mixed precision, convert to half precision for better performance
        if use_mixed_precision:
            observations = observations.half()
            actions = actions.half()
        
        # Normalize advantages for better stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset - DataLoader changes for better performance
        dataset = torch.utils.data.TensorDataset(observations, actions, old_log_probs, advantages, returns)
        
        # Adjust batch size for better GPU utilization
        optimal_batch_size = min(self.batch_size, 256)
        
        # DataLoader with performance optimizations
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=optimal_batch_size, 
                shuffle=True,
                pin_memory=is_gpu,  # Use pinned memory only for GPU
                num_workers=0  # Avoid worker overhead for RL
            )
        except:
            # Fallback to basic DataLoader
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
        
        # Track metrics across epochs
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        # PPO update for n_epochs
        for epoch in range(self.n_epochs):
            policy_losses = []
            value_losses = []
            entropies = []
            
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # Get new log probs and values - use mixed precision if available
                if use_mixed_precision:
                    with torch.autocast(device_type='cpu', dtype=torch.float16):
                        new_log_probs, values, entropy = self.policy_network.evaluate_actions(batch_states, batch_actions)
                        
                        # Convert back to float32 for stable loss calculation
                        new_log_probs = new_log_probs.float()
                        values = values.float()
                        entropy = entropy.float()
                else:
                    new_log_probs, values, entropy = self.policy_network.evaluate_actions(batch_states, batch_actions)
                
                # Ensure tensors have the proper shapes
                batch_returns = batch_returns.view(-1)
                
                # Make sure dimensions match
                if values.shape != batch_returns.shape:
                    if values.numel() == batch_returns.numel():
                        batch_returns = batch_returns.reshape(values.shape)
                    else:
                        min_size = min(values.numel(), batch_returns.numel())
                        values = values.flatten()[:min_size]
                        batch_returns = batch_returns.flatten()[:min_size]
                
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
                
                # Optimize with memory efficiency
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record metrics for this batch
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
            
            # Average metrics for this epoch
            epoch_policy_losses.append(np.mean(policy_losses))
            epoch_value_losses.append(np.mean(value_losses))
            epoch_entropies.append(np.mean(entropies))
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        # Calculate total update time
        update_time = time.time() - update_start_time
        if self.verbose > 1:
            print(f"Policy update took {update_time:.3f}s")
        
        # Explicitly clear memory to reduce fragmentation (GPU only)
        if is_gpu and hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
        # Return training metrics with timing information
        return {
            "policy_loss": np.mean(epoch_policy_losses),
            "value_loss": np.mean(epoch_value_losses),
            "entropy": np.mean(epoch_entropies),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "update_time": update_time
        }
    
    def learn(self, total_timesteps, callback=None):
        """
        Main training loop
        """
        timesteps_elapsed = 0
        iteration = 0
        
        print(f"Will run {total_timesteps} total timesteps in iterations of {self.n_steps} steps each")
        
        # Track overall training time
        training_start_time = time.time()
        
        while timesteps_elapsed < total_timesteps:
            # Print current progress
            if iteration > 0:
                elapsed_time = time.time() - training_start_time
                eta = (elapsed_time / timesteps_elapsed) * (total_timesteps - timesteps_elapsed)
                print(f"Progress: {timesteps_elapsed}/{total_timesteps} timesteps ({(timesteps_elapsed/total_timesteps)*100:.1f}%) | "
                     f"Elapsed: {elapsed_time/60:.1f}m | ETA: {eta/60:.1f}m")
            
            # Collect rollouts - returns the buffer with processed advantages and returns
            rollout_buffer = self.collect_rollouts()
            timesteps_elapsed += self.n_steps
            iteration += 1
            
            # Update policy and get metrics
            update_metrics = self.update_policy(rollout_buffer)
            
            # Print detailed metrics if verbose
            if self.verbose > 0:
                print(f"Policy update complete - Loss: {update_metrics['policy_loss']:.4f}, "
                     f"Value Loss: {update_metrics['value_loss']:.4f}, "
                     f"Entropy: {update_metrics['entropy']:.4f}, "
                     f"Learning Rate: {update_metrics['learning_rate']:.6f}")
            
            # Call callback if provided - this is where evaluation happens
            if callback is not None:
                print("\nExecuting callback - this should trigger evaluation if scheduled")
                callback(self)
        
        # Print total training time
        total_time = time.time() - training_start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Average FPS: {total_timesteps/total_time:.1f}")
        
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
        
        # Create a 2x2 grid of plots for training metrics only
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
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
        plt.subplot(2, 2, 2)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["mean_lengths"])
        
        plt.title("Mean Episode Length")
        plt.xlabel("Iteration")
        plt.ylabel("Steps")
        plt.grid(True)
        
        # Plot distances
        plt.subplot(2, 2, 3)
        plt.plot(self.training_metrics["iterations"], self.training_metrics["distances"])
        
        plt.title("Mean Distance to Target")
        plt.xlabel("Iteration")
        plt.ylabel("Distance (m)")
        plt.grid(True)
        
        # Plot success rate
        plt.subplot(2, 2, 4)
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

# Add the RolloutBuffer class before this point
class RolloutBuffer:
    """
    Optimized rollout buffer for storing and processing transitions efficiently
    """
    def __init__(self, buffer_size, observation_space, action_space, gamma=0.99, gae_lambda=0.95, n_envs=1):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        
        # Initialize dynamic storage
        self.reset()
    
    def reset(self):
        """Reset buffer counters and data"""
        self.pos = 0
        self.full = False
        
        obs_shape = self.observation_space.shape
        action_shape = self.action_space.shape
        
        # Allocate all buffers at once with proper shapes
        self.observations = np.zeros((self.buffer_size, self.n_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Add a new transition to the buffer"""
        # Handle vectorized observations
        if self.n_envs > 1:
            # For vectorized environments, obs should already be shaped right
            # Add batch dimension of 1 if it's a single vector environment
            if len(obs.shape) == len(self.observation_space.shape):
                obs = np.expand_dims(obs, 0)
            
            # Same for actions
            if len(action.shape) == len(self.action_space.shape):
                action = np.expand_dims(action, 0)
        else:
            # Single environment case
            # Make sure obs and action have the right shapes
            if len(obs.shape) < len(self.observation_space.shape) + 1:
                obs = np.expand_dims(obs, 0)
            if len(action.shape) < len(self.action_space.shape) + 1:
                action = np.expand_dims(action, 0)
        
        # Store data at current position
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        # Increment position
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_values):
        """Compute returns and advantages using GAE"""
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)
        
        # Go backwards through buffer
        for step in reversed(range(self.buffer_size)):
            # Get next value
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            # Delta (TD error)
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            
            # GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Compute returns (value targets)
        self.returns = self.advantages + self.values
        
        # Normalize advantages
        if self.full:  # Only normalize if buffer is full
            adv_mean = np.mean(self.advantages)
            adv_std = np.std(self.advantages)
            self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
    
    def get_samples(self):
        """Get all samples from the buffer, properly shaped for training"""
        # Reshape samples for training (flatten across environments)
        if self.full:
            indices = np.arange(self.buffer_size)
        else:
            indices = np.arange(self.pos)
        
        # Reshape data to work with both single and vectorized environments
        observations = self.observations[indices].reshape(-1, *self.observation_space.shape)
        actions = self.actions[indices].reshape(-1, *self.action_space.shape)
        values = self.values[indices].reshape(-1)
        log_probs = self.log_probs[indices].reshape(-1)
        advantages = self.advantages[indices].reshape(-1)
        returns = self.returns[indices].reshape(-1)
        
        return observations, actions, values, log_probs, advantages, returns

# Environment creation utility
def create_ppo_envs(num_envs=1, viz_speed=0.0, parallel_viz=False):
    """
    Create environments for PPO training with visualization
    
    Args:
        num_envs: Number of environments to create for parallel training
        viz_speed: Speed of visualization (delay in seconds)
        parallel_viz: Whether to use parallel visualization
    
    Returns:
        The training environment
    """
    # Get visualization flag
    use_gui = viz_speed > 0.0
    
    # If using GUI, always enable parallel_viz to prevent robots from overlapping
    if use_gui:
        parallel_viz = True
        
        # Get the shared client for visualization
        client_id = get_shared_pybullet_client(render=True)
        
        # Create multiple robots with proper spacing in a grid
        if num_envs > 1:
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(num_envs)))
            
            # Calculate spacing between robots
            spacing = 1.2  # Distance between robots in meters
            
            # Calculate positions for each robot in a grid
            positions = []
            for i in range(num_envs):
                # Calculate row and column in the grid
                row = i // grid_size
                col = i % grid_size
                
                # Calculate x and y offsets (centered around the origin)
                x_offset = (col - (grid_size - 1) / 2) * spacing
                y_offset = (row - (grid_size - 1) / 2) * spacing
                
                positions.append((x_offset, y_offset))
                
            # Create and configure environments
            envs = []
            for i in range(num_envs):
                x_offset, y_offset = positions[i]
                env = RobotPositioningRevampedEnv(
                    gui=use_gui,
                    gui_delay=viz_speed,
                    viz_speed=viz_speed,
                    verbose=False,
                    parallel_viz=True,
                    rank=i,
                    offset_x=x_offset
                )
                
                # Apply y-offset to the robot's position
                if y_offset != 0.0:
                    # Get the robot's position
                    robot_pos, robot_orn = p.getBasePositionAndOrientation(
                        env.robot.robot_id, 
                        physicsClientId=env.robot.client
                    )
                    
                    # Set the new position
                    p.resetBasePositionAndOrientation(
                        env.robot.robot_id,
                        [robot_pos[0], y_offset, robot_pos[2]],
                        robot_orn,
                        physicsClientId=env.robot.client
                    )
                    
                    # If verbose, print the robot's new position
                    if i == 0:
                        print(f"Robot {i} positioned at [{robot_pos[0] + x_offset:.2f}, {y_offset:.2f}, {robot_pos[2]:.2f}]")
                
                # Wrap environment to enforce joint limits
                env = JointLimitEnforcingEnv(env)
                
                # Add metrics recording wrapper
                env = gym.wrappers.RecordEpisodeStatistics(env)
                
                envs.append(env)
                
            # Adjust camera to get a good view of all robots
            # Use a slightly increased distance to ensure all robots are visible
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0 + 0.5 * grid_size,  # Increased distance for grid layout
                cameraYaw=45,                           # Angled view
                cameraPitch=-30,                       # Looking down slightly
                cameraTargetPosition=[0, 0, 0.5],      # Center of the grid
                physicsClientId=client_id
            )
            
            # Create a DummyVecEnv to treat the environments as one
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv([lambda env=env: env for env in envs])
        else:
            # Single environment case
            env = RobotPositioningRevampedEnv(
                gui=use_gui,
                gui_delay=viz_speed,
                viz_speed=viz_speed,
                verbose=False,
                parallel_viz=True,
                rank=0,
                offset_x=0.0
            )
            
            # Adjust camera for single robot
            adjust_camera_for_robots(client_id, 1)
            
            # Wrap environment
            env = JointLimitEnforcingEnv(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            
            return env
    else:
        # Non-GUI case - can create multiple environments without visualization
        if num_envs > 1:
            envs = []
            for i in range(num_envs):
                env = RobotPositioningRevampedEnv(
                    gui=False,
                    verbose=False,
                    rank=i
                )
                
                # Wrap environment
                env = JointLimitEnforcingEnv(env)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                
                envs.append(env)
                
            # Create a DummyVecEnv
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv([lambda env=env: env for env in envs])
        else:
            # Single environment without GUI
            env = RobotPositioningRevampedEnv(
                gui=False,
                verbose=False,
                rank=0
            )
            
            # Wrap environment
            env = JointLimitEnforcingEnv(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            
            return env

# Main training function
def train_robot_with_ppo_directml(args):
    """
    Train the robot using PPO algorithm with DirectML on AMD GPU
    """
    print("\n" + "="*80)
    print("FANUC Robot Positioning with PPO and DirectML")
    print("Optimized for AMD RX 6700S GPU")
    print("="*80 + "\n")
    
    # Set up performance measurements
    training_start_time = time.time()
    iteration_times = []
    fps_values = []
    
    # Pre-clean memory
    force_gc()
    
    # Setup DirectML device
    device = setup_directml() if args.gpu else torch.device("cpu")
    print(f"Using device: {device}\n")
    
    # Add timer to measure performance of key operations
    class Timer:
        def __init__(self, name):
            self.name = name
            self.start_time = None
            self.total_time = 0
            self.calls = 0
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            end_time = time.time()
            elapsed = end_time - self.start_time
            self.total_time += elapsed
            self.calls += 1
            if args[0] is not None:
                print(f"Error in {self.name}: {args[1]}")
        
        def report(self):
            avg_time = self.total_time / max(1, self.calls)
            return f"{self.name}: {self.total_time:.2f}s total, {avg_time*1000:.2f}ms avg ({self.calls} calls)"
    
    # Dictionary to store timers
    timers = {
        "env_creation": Timer("Environment Creation"),
        "model_creation": Timer("Model Creation"),
        "training": Timer("Training"),
        "evaluation": Timer("Evaluation"),
        "saving": Timer("Model Saving")
    }
    
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
    
    # Create environments with timing
    with timers["env_creation"]:
        print(f"Creating {args.parallel} parallel training robots...")
        env = create_ppo_envs(
            num_envs=args.parallel,
            viz_speed=args.viz_speed,
            parallel_viz=True
        )
    
    # Initialize model with timing
    with timers["model_creation"]:
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
    
    # Training callback for saving and monitoring
    def training_callback(model):
        # Current iteration - note: this is 1-indexed
        iteration = len(model.training_metrics["iterations"])
        
        print(f"Callback called at iteration {iteration}")
        
        # Save model periodically
        if args.save_freq > 0 and iteration > 0 and iteration % args.save_freq == 0:
            with timers["saving"]:
                save_path = f"{model_dir}/model_{iteration}.pt"
                model.save(save_path)
                
                # Plot training progress
                plot_path = f"{plot_dir}/training_progress_{iteration}.png"
                model.plot_training_progress(plot_path)
            
            # Print metrics summary
            mean_reward = model.training_metrics["mean_rewards"][-1]
            mean_length = model.training_metrics["mean_lengths"][-1]
            mean_distance = model.training_metrics["distances"][-1]
            success_rate = model.training_metrics["successes"][-1]
            
            print(f"\nTraining metrics at iteration {iteration}:")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Mean episode length: {mean_length:.2f}")
            print(f"  Mean distance to target: {mean_distance:.4f}m")
            print(f"  Success rate: {success_rate:.2f}")
            
            # Report performance
            if fps_values:
                avg_fps = np.mean(fps_values[-10:]) if len(fps_values) >= 10 else np.mean(fps_values)
                print(f"  Average FPS: {avg_fps:.1f}")
                print(f"  Average iteration time: {np.mean(iteration_times[-10:]):.2f}s")
            
            # Force garbage collection
            force_gc()
    
    # Train the model
    if not args.eval_only:
        print(f"Starting training for {args.steps} timesteps...")
        
        with timers["training"]:
            # Track iterations manually to calculate timing
            timesteps_elapsed = 0
            iteration = 0
            
            print(f"Will run {args.steps} total timesteps in iterations of {model.n_steps} steps each")
            
            while timesteps_elapsed < args.steps:
                iteration_start = time.time()
                
                # Print current progress
                if iteration > 0:
                    elapsed_time = time.time() - training_start_time
                    eta = (elapsed_time / timesteps_elapsed) * (args.steps - timesteps_elapsed)
                    print(f"Progress: {timesteps_elapsed}/{args.steps} timesteps ({(timesteps_elapsed/args.steps)*100:.1f}%) | "
                         f"Elapsed: {elapsed_time/60:.1f}m | ETA: {eta/60:.1f}m")
                
                # Collect rollouts - returns the buffer with processed advantages and returns
                rollout_buffer = model.collect_rollouts()
                
                # Track performance of this iteration
                rollout_time = time.time() - iteration_start
                fps = model.n_steps / rollout_time if rollout_time > 0 else 0
                fps_values.append(fps)
                
                # Update timesteps
                timesteps_elapsed += model.n_steps
                iteration += 1
                
                # Update policy and get metrics
                update_metrics = model.update_policy(rollout_buffer)
                
                # Print detailed metrics if verbose
                if args.verbose > 0:
                    print(f"Policy update complete - Loss: {update_metrics['policy_loss']:.4f}, "
                         f"Value Loss: {update_metrics['value_loss']:.4f}, "
                         f"Entropy: {update_metrics['entropy']:.4f}, "
                         f"Learning Rate: {update_metrics['learning_rate']:.6f}")
                
                # Call callback if provided - this is where evaluation happens
                if iteration % 2 == 0:  # Only call back every 2 iterations to reduce overhead
                    print("\nExecuting callback - this should trigger evaluation if scheduled")
                    training_callback(model)
                
                # Record iteration time
                iteration_time = time.time() - iteration_start
                iteration_times.append(iteration_time)
                
                # Print performance information
                if args.verbose > 0:
                    print(f"Iteration {iteration} completed in {iteration_time:.2f}s ({fps:.1f} FPS)")
                
                # Periodically force garbage collection to prevent memory fragmentation
                if iteration % 5 == 0:
                    force_gc()
        
        # Save final model
        with timers["saving"]:
            final_model_path = f"{model_dir}/final_model.pt"
            model.save(final_model_path)
            
            # Plot final training progress
            final_plot_path = f"{plot_dir}/final_training_progress.png"
            model.plot_training_progress(final_plot_path)
        
        print(f"Training completed! Model saved to {final_model_path}")
        
        # Run a showcase episode if visualization is enabled
        if args.viz_speed > 0.0:
            with timers["evaluation"]:
                print("\nRunning showcase episode for visualization...")
                showcase_model(model, env)
    
    # For evaluation-only mode
    if args.eval_only:
        with timers["evaluation"]:
            print("\nRunning showcase evaluation...")
            showcase_model(model, env, max_steps=300)
    
    # Print performance summary
    total_time = time.time() - training_start_time
    print("\nPerformance Summary:")
    for name, timer in timers.items():
        print(f"  {timer.report()}")
    print(f"  Total time: {total_time:.2f}s")
    
    if fps_values:
        print(f"  Average FPS: {np.mean(fps_values):.1f}")
        print(f"  Peak FPS: {np.max(fps_values):.1f}")
    
    # Return model and environment
    return model, env

# Utility function to force garbage collection
def force_gc():
    """Force garbage collection to free memory"""
    gc_start = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'directml') and torch.directml.is_available():
        # DirectML specific memory cleanup if available
        try:
            import torch_directml
            if hasattr(torch_directml, 'empty_cache'):
                torch_directml.empty_cache()
        except (ImportError, AttributeError):
            pass
    gc_time = time.time() - gc_start
    
    # Only print if it took more than 0.1 seconds
    if gc_time > 0.1:
        print(f"Memory cleaned in {gc_time:.2f}s")

def showcase_model(model, env, max_steps=200):
    """
    Run a showcase episode that demonstrates the model's performance
    with a longer step limit for visualization purposes.
    
    Args:
        model: The trained model to evaluate
        env: The evaluation environment
        max_steps: Maximum number of steps to run
    """
    # Check if we have a vector environment
    is_vector_env = hasattr(env, 'num_envs') and env.num_envs > 1
    
    if is_vector_env:
        # For vectorized environment, we'll focus on the first robot for showcase
        print("\nShowcase episode started (using first robot in environment)")
        print("Target position set - robot attempting to reach target...")
        
        # Reset all environments
        states = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            # Convert states to tensor
            states_tensor = torch.FloatTensor(states).to(model.device)
            
            # Get action from policy (deterministic for evaluation)
            with torch.no_grad():
                actions, _, _ = model.policy_network.get_action(states_tensor, deterministic=True)
            
            # Convert to numpy and ensure proper shape
            actions_np = actions.cpu().numpy()
            
            # Execute actions in all environments
            next_states, rewards, dones, infos = env.step(actions_np)
            
            # Update counters - focus on first environment for reporting
            episode_reward += rewards[0]
            step_count += 1
            
            # Print progress at intervals
            if step_count % 20 == 0:
                distance = infos[0].get('distance', 'unknown')
                print(f"  Step {step_count}: Reward={rewards[0]:.2f}, Distance={distance}")
            
            # Check if first environment is done
            done = dones[0]
            
            # Update states
            states = next_states
        
        # Print final results for the first environment
        success = infos[0].get('success', False)
        distance = infos[0].get('distance', 'unknown')
        print(f"\nShowcase episode completed after {step_count} steps")
        print(f"Final distance to target: {distance}")
        print(f"Episode success: {'Yes' if success else 'No'}")
        print(f"Total reward: {episode_reward:.2f}")
        
        # If we didn't succeed but reached the step limit, print a message
        if not success and step_count >= max_steps:
            print("Note: Reached maximum step limit without reaching target")
    
    else:
        # Original code for single environment case
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print("\nShowcase episode started")
        print("Target position set - robot attempting to reach target...")
        
        while not done and step_count < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)
            
            # Get action from policy (deterministic for evaluation)
            with torch.no_grad():
                action, _, _ = model.policy_network.get_action(state_tensor, deterministic=True)
            
            # Convert to numpy and ensure proper shape
            action_np = action.cpu().numpy().flatten()
            
            # Execute action in environment
            next_state, reward, terminated, truncated, info = env.step(action_np)
            
            # Update counters
            episode_reward += reward
            step_count += 1
            
            # Print progress at intervals
            if step_count % 20 == 0:
                distance = info.get('distance', 'unknown')
                print(f"  Step {step_count}: Reward={reward:.2f}, Distance={distance}")
            
            # Check if episode ended
            done = terminated or truncated
            
            # Update state
            state = next_state
        
        # Print final results
        success = info.get('success', False)
        distance = info.get('distance', 'unknown')
        print(f"\nShowcase episode completed after {step_count} steps")
        print(f"Final distance to target: {distance}")
        print(f"Episode success: {'Yes' if success else 'No'}")
        print(f"Total reward: {episode_reward:.2f}")
        
        # If we didn't succeed but reached the step limit, print a message
        if not success and step_count >= max_steps:
            print("Note: Reached maximum step limit without reaching target")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train FANUC robot using PPO with DirectML on AMD GPU")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=50000, help="Model saving frequency in timesteps")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel robots for training (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--load", type=str, default=None, help="Path to pre-trained model to continue training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on a pre-trained model")
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