#!/usr/bin/env python3
# directml_ppo.py
# DirectML-optimized implementation of PPO algorithm components for AMD GPUs

import torch
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Track synchronization to avoid excessive synchronization
_last_sync_time = time.time()
_sync_interval = 0.01  # seconds

def sync_dml(dml_device, force=False):
    """
    Synchronize the DirectML device by forcing a small operation to complete.
    This helps ensure timing accuracy and prevents memory leaks.
    
    Args:
        dml_device: The DirectML device to synchronize
        force: Whether to force synchronization regardless of timing
    
    Returns:
        bool: Whether synchronization was performed
    """
    global _last_sync_time, _sync_interval
    
    current_time = time.time()
    if force or (current_time - _last_sync_time) > _sync_interval:
        # DirectML has no direct synchronize method, so create a small tensor and move it to CPU
        # This forces a synchronization point
        dummy = torch.tensor([1.0], device=dml_device)
        _ = dummy.cpu()  # Force synchronization
        _last_sync_time = current_time
        return True
    return False

def to_dml(tensor, dml_device):
    """
    Move a tensor or collection of tensors to the DirectML device.
    Handles various data types and structures.
    
    Args:
        tensor: The tensor or collection to move
        dml_device: The DirectML device to move to
    
    Returns:
        The tensor or collection moved to the DirectML device
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [to_dml(t, dml_device) for t in tensor]
    if isinstance(tensor, tuple):
        return tuple(to_dml(t, dml_device) for t in tensor)
    if isinstance(tensor, dict):
        return {k: to_dml(v, dml_device) for k, v in tensor.items()}
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, device=dml_device)
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dml_device)
    return tensor

def to_cpu(tensor):
    """
    Move a tensor or collection of tensors to the CPU.
    Handles various data types and structures.
    
    Args:
        tensor: The tensor or collection to move
    
    Returns:
        The tensor or collection moved to the CPU
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [to_cpu(t) for t in tensor]
    if isinstance(tensor, tuple):
        return tuple(to_cpu(t) for t in tuple)
    if isinstance(tensor, dict):
        return {k: to_cpu(v) for k, v in tensor.items()}
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    return tensor

def compute_advantages_dml(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    dml_device: torch.device
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE) on DirectML device.
    This is a key component of PPO that benefits from GPU acceleration.
    
    Args:
        rewards: Array of episode rewards [batch_size]
        values: Array of state values [batch_size] 
        dones: Array of episode terminations [batch_size]
        next_values: Array of next state values [batch_size]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        dml_device: DirectML device
    
    Returns:
        advantages: Computed advantage values as a numpy array
    """
    # Move data to DirectML device
    rewards_t = torch.tensor(rewards, device=dml_device, dtype=torch.float32)
    values_t = torch.tensor(values, device=dml_device, dtype=torch.float32)
    dones_t = torch.tensor(dones, device=dml_device, dtype=torch.float32)  # Ensure float type
    next_values_t = torch.tensor(next_values, device=dml_device, dtype=torch.float32)
    
    # Calculate TD errors (temporal difference)
    deltas = rewards_t + gamma * next_values_t * (1.0 - dones_t) - values_t
    
    # Initialize advantages tensor on device
    advantages = torch.zeros_like(deltas, device=dml_device)
    
    # Iteratively compute GAE
    last_gae = torch.zeros(1, device=dml_device)
    
    # Reverse iteration for efficiency
    for t in reversed(range(len(deltas))):
        # GAE calculation: δ_t + γλA_{t+1}
        last_gae = deltas[t] + gamma * gae_lambda * (1.0 - dones_t[t]) * last_gae
        advantages[t] = last_gae
    
    # Force sync to ensure computation is complete
    sync_dml(dml_device, force=True)
    
    # Return as numpy array
    return advantages.cpu().numpy()

def normalize_advantages_dml(advantages: np.ndarray, dml_device: torch.device) -> np.ndarray:
    """
    Normalize advantages on DirectML device.
    Helps stabilize training by standardizing the advantage distribution.
    
    Args:
        advantages: Advantage values to normalize
        dml_device: DirectML device
    
    Returns:
        normalized_advantages: Normalized advantage values as a numpy array
    """
    # Move to DirectML
    adv_t = torch.tensor(advantages, device=dml_device)
    
    # Calculate mean and std 
    eps = 1e-8  # Small value to avoid division by zero
    mean = torch.mean(adv_t)
    std = torch.std(adv_t) + eps
    
    # Normalize
    normalized_adv = (adv_t - mean) / std
    
    # Return as numpy array
    return normalized_adv.cpu().numpy()

def compute_ppo_loss_dml(
    obs_batch: np.ndarray,
    actions_batch: np.ndarray,
    old_values: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    policy_network: torch.nn.Module,
    value_network: torch.nn.Module,
    clip_range: float,
    entropy_coef: float,
    value_coef: float,
    dml_device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO loss on DirectML device.
    Combines policy loss, value loss, and entropy into the total PPO objective.
    
    Args:
        obs_batch: Batch of observations
        actions_batch: Batch of actions taken
        old_values: Previous value estimates
        old_log_probs: Log probabilities of actions from old policy
        advantages: Computed advantage values
        returns: Computed returns (rewards-to-go)
        policy_network: Actor network for policy
        value_network: Critic network for value estimation
        clip_range: PPO clipping parameter
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        dml_device: DirectML device
        
    Returns:
        loss: Total PPO loss
        stats: Dictionary of statistics for logging
    """
    # Move data to DirectML
    obs = torch.tensor(obs_batch, dtype=torch.float32, device=dml_device)
    actions = torch.tensor(actions_batch, device=dml_device)
    old_values = torch.tensor(old_values, dtype=torch.float32, device=dml_device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=dml_device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=dml_device)
    returns = torch.tensor(returns, dtype=torch.float32, device=dml_device)
    
    # Move networks to DirectML device (if they aren't already)
    policy_network.to(dml_device)
    value_network.to(dml_device)
    
    # Instead of using the actor/critic networks directly, we'll use the policy's methods
    # which handle the proper input/output formats
    with torch.no_grad():
        _, current_values, _ = policy_network.evaluate_actions(obs)
    
    # Compute policy (actor) loss with clipping
    log_prob, entropy = policy_network.evaluate_actions(obs, actions)[:2]
    entropy = entropy.mean()
    
    # Calculate ratios and clipped loss
    ratio = torch.exp(log_prob - old_log_probs.flatten())
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.max(policy_loss1, policy_loss2).mean()
    
    # Compute value (critic) loss with clipping
    value_pred = current_values
    value_loss_unclipped = (value_pred - returns) ** 2
    value_clipped = old_values + torch.clamp(value_pred - old_values, -clip_range, clip_range)
    value_loss_clipped = (value_clipped - returns) ** 2
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean() * 0.5
    
    # Combine all losses
    loss = policy_loss - entropy_coef * entropy + value_coef * value_loss
    
    # Collect statistics
    with torch.no_grad():
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean().item()
        approx_kl = 0.5 * ((log_prob - old_log_probs.flatten()) ** 2).mean().item()
        
    # Create stats dictionary
    stats = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'clip_fraction': clip_fraction,
        'approx_kl': approx_kl,
        'explained_variance': 0.0  # Calculate later if needed
    }
    
    # Force sync at the end
    sync_dml(dml_device)
    
    return loss, stats 

def collect_rollouts_dml(
    env,
    policy,
    rollout_buffer,
    n_rollout_steps: int,
    dml_device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: str = "cpu"
):
    """
    Collect experiences using the current policy and fill a rollout buffer.
    Optimized implementation for DirectML that minimizes CPU-GPU transfers.
    
    Args:
        env: The vectorized environment
        policy: Current policy
        rollout_buffer: Buffer to fill with rollouts
        n_rollout_steps: Number of steps to collect per environment
        dml_device: DirectML device for optimized operations
        gamma: Discount factor
        gae_lambda: GAE lambda parameter 
        device: Device for compatibility with original implementation
    
    Returns:
        bool: Whether the rollout was completed successfully
    """
    assert rollout_buffer is not None, "Rollout buffer cannot be None"
    
    # Reset the rollout buffer
    rollout_buffer.reset()
    
    # Get the last observation from the environment or reset if needed
    obs = env.reset()
    
    # Store references for faster access
    n_envs = env.num_envs
    
    # Prepare tracking variables
    episode_starts = np.ones((n_envs,), dtype=bool)
    
    # Convert policy network to DirectML for forward passes
    policy_on_dml = policy
    policy_on_dml.to(dml_device)
    
    # Begin the rollout loop - collect one less than the buffer size
    # because we need space for computing advantages later
    for step in range(n_rollout_steps):
        # Convert observation to tensor and move to DirectML
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(dml_device)
        
        # Get action from policy
        with torch.no_grad():
            actions, values, log_probs = policy_on_dml.forward(obs_tensor)
            
            # Move tensors back to CPU for environment interaction
            actions_np = actions.cpu().numpy()
            values_np = values.cpu()  # Keep as tensor for buffer
            log_probs_np = log_probs.cpu()  # Keep as tensor for buffer
        
        # Execute actions in the environment
        new_obs, rewards, dones, infos = env.step(actions_np)
        
        # Add data to rollout buffer for all environments at once
        rollout_buffer.add(
            obs=obs,
            action=actions_np,
            reward=rewards,
            episode_start=episode_starts,
            value=values_np,
            log_prob=log_probs_np
        )
        
        # Update for next step
        obs = new_obs
        episode_starts = dones
    
    # Compute value estimates for bootstrapping
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(dml_device)
        # Get value estimates for last observation
        values = policy_on_dml.predict_values(obs_tensor)
        # Convert to CPU and ensure proper shape for RolloutBuffer
        last_values = values.cpu()
    
    # Compute advantages and returns using the buffer's method
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    
    # Ensure all DirectML operations are complete
    sync_dml(dml_device, force=True)
    
    return True

def optimize_policy_dml(
    policy,
    optimizer,
    rollout_buffer,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    target_kl: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    dml_device: torch.device
):
    """
    Optimize the policy using collected rollout data with DirectML acceleration.
    
    Args:
        policy: Policy network to optimize
        optimizer: Optimizer for policy parameters
        rollout_buffer: Buffer containing collected experiences
        n_epochs: Number of optimization epochs per update
        batch_size: Minibatch size for optimization
        clip_range: PPO clipping parameter
        target_kl: Target KL divergence threshold for early stopping
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum norm for gradient clipping
        dml_device: DirectML device to use for computation
    
    Returns:
        dict: Statistics from the optimization
    """
    # Get the data from rollout buffer
    data = rollout_buffer.get()
    
    # Extract data directly from buffer attributes instead
    observations = rollout_buffer.observations
    actions = rollout_buffer.actions
    old_values = rollout_buffer.values
    old_log_probs = rollout_buffer.log_probs
    advantages = rollout_buffer.advantages
    returns = rollout_buffer.returns
    
    # Get observation and action dimensions
    obs_shape = observations.shape
    act_shape = actions.shape
    
    # Flatten if needed to match policy expectations
    # SB3 buffers typically store data as [n_steps, n_envs, feature_dim]
    # We need to reshape to [n_steps * n_envs, feature_dim]
    if len(obs_shape) > 2:
        observations = observations.reshape((-1, *obs_shape[2:]))
        
    if len(act_shape) > 2:
        actions = actions.reshape((-1, *act_shape[2:]))
        
    # If values and log_probs are multi-dimensional, flatten them too
    if len(old_values.shape) > 1:
        old_values = old_values.flatten()
        
    if len(old_log_probs.shape) > 1:
        old_log_probs = old_log_probs.flatten()
        
    # Do the same for advantages and returns
    if len(advantages.shape) > 1:
        advantages = advantages.flatten()
        
    if len(returns.shape) > 1:
        returns = returns.flatten()
    
    # Initialize statistics tracking
    pg_losses, value_losses, entropies = [], [], []
    clip_fractions = []
    approx_kl_divs = []
    
    # Print shapes for debugging
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Values shape: {old_values.shape}")
    print(f"Log probs shape: {old_log_probs.shape}")
    print(f"Advantages shape: {advantages.shape}")
    print(f"Returns shape: {returns.shape}")
    
    # Check sizes of policy inputs - helpful for debugging model architecture
    print(f"Policy observation space: {policy.observation_space}")
    print(f"Policy action space: {policy.action_space}")
    
    # Convert policy to DirectML for training
    policy.to(dml_device)
    
    # Optimization loop
    for epoch in range(n_epochs):
        # Shuffle the data
        indices = np.random.permutation(len(observations))
        
        # Iterate through mini-batches
        for start_idx in range(0, len(observations), batch_size):
            end_idx = min(start_idx + batch_size, len(observations))
            batch_indices = indices[start_idx:end_idx]
            
            # Get minibatch data
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]
            batch_values = old_values[batch_indices]
            batch_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Convert to tensors and move to DirectML
            batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=dml_device)
            batch_actions = torch.tensor(batch_actions, dtype=torch.float32, device=dml_device)
            batch_values = torch.tensor(batch_values, dtype=torch.float32, device=dml_device).flatten()
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=dml_device).flatten()
            batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=dml_device).flatten()
            batch_returns = torch.tensor(batch_returns, dtype=torch.float32, device=dml_device).flatten()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Evaluate actions and get new log probs and entropy
            log_prob, entropy, values = policy.evaluate_actions(batch_obs, batch_actions)
            log_prob = log_prob.flatten()
            values = values.flatten()
            entropy = entropy.mean()
            
            # Compute policy (actor) loss with clipping
            ratio = torch.exp(log_prob - batch_log_probs)
            policy_loss1 = -batch_advantages * ratio
            policy_loss2 = -batch_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()
            
            # Compute value (critic) loss with clipping
            value_loss_unclipped = (values - batch_returns) ** 2
            value_clipped = batch_values + torch.clamp(values - batch_values, -clip_range, clip_range)
            value_loss_clipped = (value_clipped - batch_returns) ** 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean() * 0.5
            
            # Combine all losses
            loss = policy_loss - entropy_coef * entropy + value_coef * value_loss
            
            # Optimization step
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            
            # Calculate approximate KL divergence for early stopping
            with torch.no_grad():
                clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean().item()
                approx_kl = 0.5 * ((log_prob - batch_log_probs) ** 2).mean().item()
            
            # Store statistics
            pg_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            clip_fractions.append(clip_fraction)
            approx_kl_divs.append(approx_kl)
        
        # Check for early stopping
        if len(approx_kl_divs) > 0:
            mean_kl = np.mean(approx_kl_divs)
            if mean_kl > 3.0 * target_kl:
                print(f"Early stopping at epoch {epoch+1}/{n_epochs}: KL divergence {mean_kl:.4f} > target {3.0 * target_kl:.4f}")
                break
    
    # Compute explained variance between value predictions and returns
    explained_var = explained_variance(
        rollout_buffer.values.flatten(), 
        rollout_buffer.returns.flatten()
    )
    
    # Gather and return statistics
    stats = {
        "loss/policy_gradient": np.mean(pg_losses) if len(pg_losses) > 0 else 0.0,
        "loss/value_function": np.mean(value_losses) if len(value_losses) > 0 else 0.0,
        "loss/entropy": np.mean(entropies) if len(entropies) > 0 else 0.0,
        "rollout/explained_variance": explained_var,
        "rollout/clip_fraction": np.mean(clip_fractions) if len(clip_fractions) > 0 else 0.0,
        "rollout/approx_kl": np.mean(approx_kl_divs) if len(approx_kl_divs) > 0 else 0.0
    }
    
    return stats

def explained_variance(y_pred, y_true):
    """
    Compute the explained variance.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        float: Explained variance
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    
    return 1 - np.var(y_true - y_pred) / var_y 

class DirectMLPPO:
    """
    DirectML-optimized Proximal Policy Optimization algorithm (PPO).
    This is a drop-in replacement for Stable Baselines 3 PPO that uses DirectML
    for improved performance on AMD GPUs.
    
    Paper: https://arxiv.org/abs/1707.06347
    
    Args:
        policy: The policy model to use
        env: The environment to learn from
        learning_rate: Learning rate
        n_steps: The number of steps to run for each environment per update
        batch_size: Minibatch size for optimization
        n_epochs: Number of epochs for optimization
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance in GAE
        clip_range: Clipping parameter for PPO
        clip_range_vf: Clipping parameter for the value function (None to disable)
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: The maximum value for the gradient clipping
        target_kl: Target KL divergence threshold for early stopping
        verbose: The verbosity level: 0 no output, 1 info, 2 debug
    """
    def __init__(
        self,
        policy,
        env,
        dml_device=None,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        verbose=0,
        seed=None,
        **kwargs
    ):
        # Try to import torch_directml
        try:
            import torch_directml
            if dml_device is None:
                dml_device = torch_directml.device()
        except ImportError:
            raise ImportError(
                "DirectML PPO requires torch_directml to be installed. "
                "Please install it with: pip install torch-directml"
            )
        
        # Store DirectML device
        self.dml_device = dml_device
        
        # Store for compatibility with Stable Baselines
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        # Increase the target KL divergence threshold to a more permissive value
        # This prevents early stopping during early training stages when policy changes are large
        self.target_kl = 0.1 if target_kl == 0.01 else target_kl  # Default to 0.1 instead of 0.01
        self.verbose = verbose
        self.seed = seed
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Store environment properties
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        
        # Initialize rollout buffer
        # Import from stable-baselines3
        try:
            from stable_baselines3.common.buffers import RolloutBuffer
            self.rollout_buffer = RolloutBuffer(
                buffer_size=n_steps,  # This is per-environment
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=torch.device("cpu"),  # Keep buffer on CPU
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_envs=env.num_envs  # Specify number of environments
            )
        except ImportError:
            raise ImportError(
                "DirectML PPO requires stable-baselines3 to be installed. "
                "Please install it with: pip install stable-baselines3"
            )
        
        # Initialize policy
        if isinstance(policy, str):
            # Policy is specified as a string, e.g., "MlpPolicy"
            try:
                from stable_baselines3.ppo import PPO
                # Create base PPO with policy
                temp_ppo = PPO(policy, env, verbose=0)
                # Extract policy
                self.policy = temp_ppo.policy
                # Cleanup temporary PPO
                del temp_ppo
            except ImportError:
                raise ImportError(
                    "Could not create policy from string. "
                    "Please pass a policy instance instead."
                )
        else:
            # Policy is passed directly
            self.policy = policy
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate,
            eps=1e-5
        )
        
        # Learning rate schedule
        self._current_learning_rate = learning_rate
        
        # Initialize the policy on DirectML for faster inference
        self.policy.to(self.dml_device)
        
        # Setup for rollouts
        self.num_timesteps = 0
        self._last_obs = None
        self._last_episode_starts = None
        
        # Start verbosity outputs
        if self.verbose > 0:
            print(f"DirectML PPO initialized with device: {dml_device}")
            print(f"Batch size: {batch_size}, Steps per update: {n_steps}")
            print(f"Learning rate: {learning_rate}")
        
    def _update_learning_rate(self):
        """
        Update the learning rate according to the schedule.
        In this implementation, we use a constant learning rate.
        """
        # No learning rate schedule in this implementation
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._current_learning_rate
    
    def collect_rollouts(self, callback=None):
        """
        Collect experiences using the current policy and fill a rollout buffer.
        
        Args:
            callback: Callback that will be called at each step
                (if None, no callback will be called)
        
        Returns:
            bool: Whether rollout collection was successful
        """
        # Reset if needed
        if self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        
        # Use our DirectML-optimized rollout collection
        return collect_rollouts_dml(
            env=self.env,
            policy=self.policy,
            rollout_buffer=self.rollout_buffer,
            n_rollout_steps=self.n_steps,
            dml_device=self.dml_device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
    
    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        
        Returns:
            dict: Statistics from the update
        """
        # Update learning rate
        self._update_learning_rate()
        
        # Use our DirectML-optimized policy update
        stats = optimize_policy_dml(
            policy=self.policy,
            optimizer=self.optimizer,
            rollout_buffer=self.rollout_buffer,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            clip_range=self.clip_range,
            target_kl=self.target_kl,
            entropy_coef=self.ent_coef,
            value_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            dml_device=self.dml_device
        )
        
        return stats
    
    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="DirectML_PPO",
        reset_num_timesteps=True,
        progress_bar=False
    ):
        """
        Return a trained model.
        
        Args:
            total_timesteps: The total number of samples to train on
            callback: Callback that will be called at each step
                (if None, no callback will be called)
            log_interval: The number of timesteps before logging
            tb_log_name: The name of the run for tensorboard log
            reset_num_timesteps: Whether to reset the current timestep number
            progress_bar: Display a progress bar during training
            
        Returns:
            self: The trained model
        """
        # For tracking
        iteration = 0
        start_time = time.time()
        
        # Reset the environment if needed
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        
        # Main training loop
        while self.num_timesteps < total_timesteps:
            # Collect a batch of experiences
            if self.verbose > 0 and iteration % log_interval == 0:
                print(f"Collecting rollouts... (iteration {iteration})")
            
            # Collect experiences
            self.collect_rollouts(callback=callback)
            
            # Increment the timesteps
            self.num_timesteps += self.env.num_envs * self.n_steps
            iteration += 1
            
            # Update the policy
            if self.verbose > 0 and iteration % log_interval == 0:
                print(f"Updating policy... (timestep {self.num_timesteps}/{total_timesteps})")
            
            # Train the policy with the collected data
            stats = self.train()
            
            # Log statistics
            if self.verbose > 0 and iteration % log_interval == 0:
                # Calculate frames per second
                fps = int(self.num_timesteps / (time.time() - start_time))
                
                # Print training stats
                print(f"Timesteps: {self.num_timesteps}")
                print(f"FPS: {fps}")
                print(f"Policy loss: {stats['loss/policy_gradient']:.5f}")
                print(f"Value loss: {stats['loss/value_function']:.5f}")
                print(f"Entropy: {stats['loss/entropy']:.5f}")
                print(f"Clip fraction: {stats['rollout/clip_fraction']:.5f}")
                print(f"Approx KL: {stats['rollout/approx_kl']:.5f}")
                print(f"Explained variance: {stats['rollout/explained_variance']:.5f}")
                print("-" * 50)
        
        return self
    
    def predict(
        self, 
        observation, 
        state=None, 
        deterministic=False
    ):
        """
        Get the policy action from an observation.
        
        Args:
            observation: The input observation
            state: The last states (ignored if no state)
            deterministic: Whether to use stochastic or deterministic actions
            
        Returns:
            actions, states: The model's action and next state if stateful
        """
        # Convert to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation).float()
        
        # Move to DirectML
        observation = observation.to(self.dml_device)
        
        # Get action from policy
        with torch.no_grad():
            actions = self.policy.act(observation, deterministic=deterministic)
        
        # Return as numpy arrays
        return actions.cpu().numpy(), None
    
    def save(self, path):
        """
        Save the model to the given path.
        
        Args:
            path: Path to save the model
        """
        # Make sure model is on CPU before saving
        cpu_state_dict = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        
        # Save the model
        data = {
            "policy_state_dict": cpu_state_dict,
            "policy_class": type(self.policy).__name__,
            "version": "1.0.0",
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl
        }
        
        # Save the data
        torch.save(data, path)
        
        if self.verbose > 0:
            print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, env=None, dml_device=None, **kwargs):
        """
        Load the model from the given path.
        
        Args:
            path: Path to load the model from
            env: The environment to use (can be None if only inference)
            device: The device to load the model on
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            model: The loaded model
        """
        # Try to import torch_directml
        try:
            import torch_directml
            if dml_device is None:
                dml_device = torch_directml.device()
        except ImportError:
            print("Warning: DirectML not available. Using CPU.")
            dml_device = torch.device("cpu")
        
        # Load the data
        data = torch.load(path, map_location="cpu")
        
        # Create a new instance of the model
        if env is None:
            # Cannot create a model without an environment
            raise ValueError("Environment must be provided when loading a model.")
        
        # Create the model
        model = cls(
            policy="MlpPolicy",  # Placeholder, will be replaced
            env=env,
            dml_device=dml_device,
            learning_rate=data.get("learning_rate", 3e-4),
            n_steps=data.get("n_steps", 2048),
            batch_size=data.get("batch_size", 64),
            n_epochs=data.get("n_epochs", 10),
            gamma=data.get("gamma", 0.99),
            gae_lambda=data.get("gae_lambda", 0.95),
            clip_range=data.get("clip_range", 0.2),
            ent_coef=data.get("ent_coef", 0.0),
            vf_coef=data.get("vf_coef", 0.5),
            max_grad_norm=data.get("max_grad_norm", 0.5),
            target_kl=data.get("target_kl", 0.01),
            **kwargs
        )
        
        # Load the policy state dict
        model.policy.load_state_dict(data["policy_state_dict"])
        model.policy.to(dml_device)
        
        return model 