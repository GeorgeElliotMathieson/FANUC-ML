"""
Custom PPO implementation for robot control.
"""

import torch
from stable_baselines3 import PPO
from src.core.models.networks import CustomActorCriticPolicy

class CustomPPO(PPO):
    """
    Custom PPO implementation with our improved policy architecture
    that inherently respects joint limits during action sampling.
    """
    def __init__(self, policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 clip_range_vf=None, normalize_advantage=True, ent_coef=0.0,
                 vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1,
                 target_kl=None, tensorboard_log=None, policy_kwargs=None,
                 verbose=0, seed=None, device='auto', _init_setup_model=True):
        """
        Initialize the CustomPPO.
        
        Args:
            policy: Policy network class or string
            env: Training environment
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            clip_range_vf: Clipping parameter for the value function
            normalize_advantage: Whether to normalize advantages
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum norm for gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample frequency for gSDE
            target_kl: Target KL divergence threshold for early stopping
            tensorboard_log: Path for tensorboard log
            policy_kwargs: Arguments to be passed to the policy on creation
            verbose: Verbosity level
            seed: Random seed
            device: Device to run the model on
            _init_setup_model: Whether to build the network at initialization
        """
        # Remove create_eval_env which is causing a linter error
        super(CustomPPO, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        ) 