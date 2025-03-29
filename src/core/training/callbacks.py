"""
Training callbacks for monitoring and saving models.
"""

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class SaveModelCallback(BaseCallback):
    """Callback for saving models during training."""
    
    def __init__(self, save_freq=None, check_freq=None, save_path=None, verbose=1):
        """
        Initialize the callback.
        
        Args:
            save_freq: Save frequency in timesteps
            check_freq: Check frequency in timesteps
            save_path: Path to save models
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.check_freq = check_freq or save_freq
        self.save_path = save_path
        
    def _on_step(self):
        """Called at each step during training."""
        if self.check_freq and self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print(f"Timestep {self.n_calls}: Checking model...")
                
            if self.save_freq and self.n_calls % self.save_freq == 0:
                if self.save_path:
                    path = os.path.join(self.save_path, f"model_{self.n_calls}")
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"Saved model to {path}")
        
        return True

class TrainingMonitorCallback(BaseCallback):
    """Callback for monitoring training progress."""
    
    def __init__(self, log_interval=100, verbose=1, plot_interval=10000, plot_dir=None, model_dir=None):
        """
        Initialize the callback.
        
        Args:
            log_interval: Log interval in timesteps
            verbose: Verbosity level
            plot_interval: Plot interval in timesteps
            plot_dir: Directory for saving plots
            model_dir: Directory for saving models
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        
        # Metrics to track
        self.timesteps = []
        self.rewards = []
        self.episode_lengths = []
        self.learning_rates = []
        self.explained_variances = []
        self.values_info = []
        self.success_rates = []
        self.distances = []
        self.joint_limit_violations = []
        
        # Start time
        self.start_time = time.time()
        
    def _on_step(self):
        """Called at each step during training."""
        # Get info from the most recent episode
        if self.log_interval and self.n_calls % self.log_interval == 0:
            # Extract metrics from model
            rewards = []
            lengths = []
            successes = []
            distances = []
            limit_violations = []
            
            # Check if we have recent episode info
            if len(self.model.ep_info_buffer) > 0:
                for ep_info in self.model.ep_info_buffer:
                    rewards.append(ep_info.get("r", 0))
                    lengths.append(ep_info.get("l", 0))
                    successes.append(ep_info.get("success", False))
                    distances.append(ep_info.get("distance", float("inf")))
                    limit_violations.append(not ep_info.get("positions_within_limits", True))
            
            # Check if we have current learning rate
            if hasattr(self.model, "learning_rate"):
                if callable(self.model.learning_rate):
                    lr = self.model.learning_rate(self.num_timesteps)
                else:
                    lr = self.model.learning_rate
            else:
                lr = None
                
            # Check if we have explained variance
            explained_var = None
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                if "train/explained_variance" in self.model.logger.name_to_value:
                    explained_var = self.model.logger.name_to_value["train/explained_variance"]
            
            # Get value function statistics if available
            value_info = None
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                if "train/value_loss" in self.model.logger.name_to_value:
                    value_info = {
                        "value_loss": self.model.logger.name_to_value.get("train/value_loss", 0),
                        "value_mean": self.model.logger.name_to_value.get("train/value_mean", 0),
                        "value_std": self.model.logger.name_to_value.get("train/value_std", 0)
                    }
            
            # Append metrics
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(np.mean(rewards) if len(rewards) > 0 else 0)
            self.episode_lengths.append(np.mean(lengths) if len(lengths) > 0 else 0)
            self.learning_rates.append(lr)
            self.explained_variances.append(explained_var)
            self.values_info.append(value_info)
            self.success_rates.append(np.mean(successes) if len(successes) > 0 else 0)
            self.distances.append(np.mean(distances) if len(distances) > 0 else float("inf"))
            self.joint_limit_violations.append(np.mean(limit_violations) if len(limit_violations) > 0 else 0)
            
            # Log metrics
            if self.verbose > 0:
                elapsed_time = time.time() - self.start_time
                fps = int(self.num_timesteps / elapsed_time)
                
                print(f"Steps: {self.num_timesteps} ({fps} fps)")
                if len(rewards) > 0:
                    print(f"  Mean reward: {np.mean(rewards):.2f}")
                    print(f"  Mean episode length: {np.mean(lengths):.1f}")
                    print(f"  Success rate: {np.mean(successes):.2%}")
                    if not np.isinf(np.mean(distances)):
                        print(f"  Mean distance: {np.mean(distances)*100:.2f} cm")
                    print(f"  Joint limit violations: {np.mean(limit_violations):.2%}")
                if lr is not None:
                    print(f"  Learning rate: {lr:.6f}")
                if explained_var is not None:
                    print(f"  Explained variance: {explained_var:.4f}")
                print(f"  Elapsed time: {datetime.timedelta(seconds=int(elapsed_time))}")
                
            # Plot metrics
            if self.plot_interval and self.n_calls % self.plot_interval == 0 and self.plot_dir:
                plot_path = os.path.join(self.plot_dir, f"training_progress_{self.num_timesteps}.png")
                self.plot_training_progress(plot_path)
                
                # Save model
                if self.model_dir:
                    checkpoint_path = os.path.join(self.model_dir, f"checkpoint_{self.num_timesteps}")
                    self.model.save(checkpoint_path)
                    if self.verbose > 0:
                        print(f"Saved checkpoint to {checkpoint_path}")
                        
        return True
        
    def plot_training_progress(self, save_path):
        """
        Plot training progress metrics.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.timesteps) < 2:
            return
            
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("Training Progress", fontsize=16)
        
        # Plot rewards
        axs[0, 0].plot(self.timesteps, self.rewards)
        axs[0, 0].set_title("Mean Episode Reward")
        axs[0, 0].set_xlabel("Timesteps")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].grid(True)
        
        # Plot episode lengths
        axs[0, 1].plot(self.timesteps, self.episode_lengths)
        axs[0, 1].set_title("Mean Episode Length")
        axs[0, 1].set_xlabel("Timesteps")
        axs[0, 1].set_ylabel("Steps")
        axs[0, 1].grid(True)
        
        # Plot success rate
        axs[1, 0].plot(self.timesteps, self.success_rates)
        axs[1, 0].set_title("Success Rate")
        axs[1, 0].set_xlabel("Timesteps")
        axs[1, 0].set_ylabel("Success Rate")
        axs[1, 0].set_ylim([0, 1])
        axs[1, 0].grid(True)
        
        # Plot mean distance
        non_inf_distances = [d if not np.isinf(d) else 1.0 for d in self.distances]
        axs[1, 1].plot(self.timesteps, non_inf_distances)
        axs[1, 1].set_title("Mean Distance to Target (m)")
        axs[1, 1].set_xlabel("Timesteps")
        axs[1, 1].set_ylabel("Distance (m)")
        axs[1, 1].grid(True)
        
        # Plot learning rate
        if any(lr is not None for lr in self.learning_rates):
            clean_lr = [lr if lr is not None else 0 for lr in self.learning_rates]
            axs[2, 0].plot(self.timesteps, clean_lr)
            axs[2, 0].set_title("Learning Rate")
            axs[2, 0].set_xlabel("Timesteps")
            axs[2, 0].set_ylabel("Learning Rate")
            axs[2, 0].grid(True)
        
        # Plot joint limit violations
        axs[2, 1].plot(self.timesteps, self.joint_limit_violations)
        axs[2, 1].set_title("Joint Limit Violations")
        axs[2, 1].set_xlabel("Timesteps")
        axs[2, 1].set_ylabel("Violation Rate")
        axs[2, 1].set_ylim([0, 1])
        axs[2, 1].grid(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        plt.close(fig)

class JointLimitMonitorCallback(BaseCallback):
    """Callback for monitoring joint limit violations."""
    
    def __init__(self, log_interval=5000, verbose=0):
        """
        Initialize the callback.
        
        Args:
            log_interval: Log interval in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        
    def _init_callback(self):
        """Initialize counters."""
        # Initialize counters
        self.total_steps = 0
        self.violation_steps = 0
        self.total_violations_by_joint = {}
        self.episode_violations = 0
        self.episode_steps = 0
        
    def _on_step(self):
        """Called at each step during training."""
        # Check if we have environment info
        if self.locals and "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]  # Get info from first environment
            
            # Check for positions_within_limits
            if "positions_within_limits" in info:
                within_limits = info["positions_within_limits"]
                if not within_limits:
                    self.violation_steps += 1
                    self.episode_violations += 1
                    
                    # Check for joint positions and joint limits
                    if "joint_positions" in info and hasattr(self.training_env.envs[0].unwrapped, "robot"):
                        robot = self.training_env.envs[0].unwrapped.robot
                        if hasattr(robot, "joint_limits"):
                            joint_positions = info["joint_positions"]
                            for i, pos in enumerate(joint_positions):
                                if i in robot.joint_limits:
                                    limit_low, limit_high = robot.joint_limits[i]
                                    if pos < limit_low or pos > limit_high:
                                        if i not in self.total_violations_by_joint:
                                            self.total_violations_by_joint[i] = 0
                                        self.total_violations_by_joint[i] += 1
            
            self.total_steps += 1
            self.episode_steps += 1
            
            # Check for episode termination
            if "terminal_observation" in info:
                # Reset episode counters
                self.episode_violations = 0
                self.episode_steps = 0
                
        # Log periodically
        if self.log_interval and self.n_calls % self.log_interval == 0:
            if self.total_steps > 0:
                violation_rate = self.violation_steps / self.total_steps
                
                # Sort joints by violation count
                sorted_violations = sorted(
                    self.total_violations_by_joint.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if self.verbose > 0:
                    print("\nJoint Limit Monitor:")
                    print(f"  Steps: {self.total_steps}")
                    print(f"  Violations: {self.violation_steps}")
                    print(f"  Violation rate: {violation_rate:.2%}")
                    
                    if sorted_violations:
                        print("  Violations by joint:")
                        for joint, count in sorted_violations:
                            print(f"    Joint {joint}: {count} violations ({count/self.total_steps:.2%})")
                            
        return True 