#!/usr/bin/env python3
# train_robot_rl_positioning_revamped.py
# A completely revamped approach to training a robot for accurate end-effector positioning

import os
import time
import math
import json
import argparse
import numpy as np
import gc  # Explicit garbage collection
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import imageio  # For video recording if available
except ImportError:
    print("Warning: imageio not found, video recording will be disabled")
    imageio = None
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
from stable_baselines3.common.preprocessing import get_action_dim
import torch as th
from stable_baselines3.common.utils import set_random_seed

# Global shared variables
_WORKSPACE_POSITIONS = None
_WORKSPACE_JOINT_CONFIGS = None
_SHARED_MODEL = None
_MODEL_VERSION = 0
_TARGET_REACHED_FLAG = False
_BEST_TIMEOUT_DISTANCE = float('inf')
_BEST_TIMEOUT_ROBOT_RANK = -1
_BEST_TIMEOUT_REWARD = float('-inf')
_LAST_TARGET_RANDOMIZATION_TIME = 0.0
_MODEL_UPDATE_FLAG = False
_MODEL_UPDATE_ROBOT_RANK = 0

# Visualization functions
def visualize_target(position, client_id):
    """Create a visual marker for the target position"""
    target_visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 0.7],  # Green with transparency
        physicsClientId=client_id
    )
    
    target_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual_id,
        basePosition=position,
        physicsClientId=client_id
    )
    
    return target_body_id

# Import shared functionality from the original implementation
# This will be replaced with our own implementations in subsequent edits
from src.core.train_robot_rl_positioning import (
    get_shared_pybullet_client, 
    load_workspace_data,
    determine_reachable_workspace,
    adjust_camera_for_robots
)

# Import the robot environment directly from robot_sim.py
from src.envs.robot_sim import FANUCRobotEnv

# Utility function to strictly enforce joint limits from URDF
def ensure_joint_limits(robot, joint_positions):
    """
    Strictly enforce joint limits from the URDF model.
    
    Args:
        robot: The robot environment instance
        joint_positions: Array of joint positions to check and enforce
        
    Returns:
        Array of joint positions with limits enforced
    """
    limited_positions = joint_positions.copy()
    
    for i, pos in enumerate(joint_positions):
        if i in robot.joint_limits:
            limit_low, limit_high = robot.joint_limits[i]
            # Strictly enforce limits
            if pos < limit_low:
                limited_positions[i] = limit_low
                print(f"WARNING: Joint {i} below limit ({pos:.4f} < {limit_low:.4f}), enforcing limit")
            elif pos > limit_high:
                limited_positions[i] = limit_high
                print(f"WARNING: Joint {i} above limit ({pos:.4f} > {limit_high:.4f}), enforcing limit")
    
    return limited_positions

# Custom environment wrapper to enforce joint limits
class JointLimitEnforcingEnv(gym.Wrapper):
    """
    Environment wrapper that works with JointLimitedBox action space to enforce joint limits.
    With JointLimitedBox, this wrapper is mostly for backward compatibility and monitoring.
    """
    def __init__(self, env):
        super().__init__(env)
        self.robot = env.robot
        
        # Check if we're using the JointLimitedBox action space
        if isinstance(env.action_space, JointLimitedBox):
            print("Using JointLimitedBox action space - joint limits are inherently enforced")
            self.using_joint_limited_box = True
        else:
            print("LEGACY MODE: Joint limits will be enforced by the environment wrapper")
            self.using_joint_limited_box = False
    
    def step(self, action):
        # If using JointLimitedBox, the limits are already enforced by the action space
        if self.using_joint_limited_box:
            # Just pass the action through to the underlying environment
            return self.env.step(action)
        
        # Legacy mode: manually enforce limits
        else:
            # Extract the underlying robot action from the environment's step method
            if hasattr(self.env, 'robot') and hasattr(self.env.robot, 'step'):
                # Get current joint positions
                state = self.robot._get_state()
                current_joint_positions = state[:self.robot.dof*2:2]
                
                # For delta joint position control, calculate new positions
                new_joint_positions = []
                for i, delta in enumerate(action):
                    # Current position plus delta
                    new_pos = current_joint_positions[i] + delta
                    new_joint_positions.append(new_pos)
                
                # Enforce joint limits
                limited_positions = ensure_joint_limits(self.robot, new_joint_positions)
                
                # Create zero velocities for the robot step
                zero_velocities = [0.0] * len(limited_positions)
                
                # Call the original step method with enforced limits
                next_state = self.robot.step((limited_positions, zero_velocities))
                
                # Pass to parent step with the original action (the environment will re-apply limits internally)
                return self.env.step(action)
            else:
                # Fallback if the environment doesn't match our expected structure
                return self.env.step(action)

# Add this new class near the top of the file after the imports
class JointLimitedBox(spaces.Box):
    """
    A gym.spaces.Box variant that inherently respects joint limits.
    
    This space represents actions as normalized values in [-1, 1], which are then
    mapped to the corresponding joint limits. This ensures that any action sampled
    from this space is always within the physical limits of the robot.
    
    Additional features:
    - Provides metadata about joint limits for policy networks
    - Implements custom sampling that respects safe margins near limits
    - Includes helper methods for action normalization/unnormalization
    """
    def __init__(self, robot, shape=(5,), dtype=np.float32):
        """
        Initialize JointLimitedBox action space.
        
        Args:
            robot: The FANUCRobotEnv instance with joint_limits dictionary
            shape: The shape of the action space (default is 5 for the FANUC robot)
            dtype: The data type of the action space
        """
        super().__init__(low=-1.0, high=1.0, shape=shape, dtype=dtype)
        self.robot = robot
        
        # Store joint limits for quick access
        self.joint_limits = {}
        for i in range(shape[0]):
            if i in robot.joint_limits:
                self.joint_limits[i] = robot.joint_limits[i]
        
        # Calculate midpoints and ranges for each joint for efficient unnormalization
        self.joint_mids = {}
        self.joint_ranges = {}
        
        for joint_idx, limits in self.joint_limits.items():
            limit_low, limit_high = limits
            mid = (limit_high + limit_low) / 2.0
            rng = (limit_high - limit_low) / 2.0
            
            self.joint_mids[joint_idx] = mid
            self.joint_ranges[joint_idx] = rng
        
        # Safety margin factor (reduces the effective range to avoid getting too close to limits)
        self.safety_margin = 0.97  # Use 97% of the range to avoid hitting exact limits
    
    def sample(self):
        """
        Sample a random action that respects joint limits with a safety margin.
        This ensures that random exploration doesn't push joints to their absolute limits.
        """
        # Sample uniformly but with slightly reduced range to leave safety margin
        action = np.random.uniform(
            low=-1.0 * self.safety_margin, 
            high=1.0 * self.safety_margin, 
            size=self.shape
        ).astype(self.dtype)
        
        return action
    
    def contains(self, x):
        """
        Check if the action is within the space's bounds, respecting joint limits.
        Since all normalized actions in [-1, 1] are valid by design, we just check
        the normalized action range.
        """
        if isinstance(x, list):
            x = np.array(x, dtype=self.dtype)
        
        return np.all(np.logical_and(x >= self.low, x <= self.high))
    
    def normalize_action(self, joint_positions):
        """
        Convert real joint positions to normalized actions in [-1, 1].
        
        Args:
            joint_positions: List of joint positions in radians
            
        Returns:
            Normalized action array with values in [-1, 1]
        """
        normalized = np.zeros(self.shape, dtype=self.dtype)
        
        for joint_idx in range(len(joint_positions)):
            if joint_idx in self.joint_limits:
                mid = self.joint_mids[joint_idx]
                rng = self.joint_ranges[joint_idx]
                
                # Normalize to [-1, 1]
                normalized[joint_idx] = (joint_positions[joint_idx] - mid) / rng
                
                # Ensure we're strictly within bounds
                normalized[joint_idx] = np.clip(normalized[joint_idx], -1.0, 1.0)
            else:
                # For joints without defined limits, use the position directly
                normalized[joint_idx] = np.clip(joint_positions[joint_idx], -1.0, 1.0)
        
        return normalized
    
    def unnormalize_action(self, normalized_action):
        """
        Convert normalized actions in [-1, 1] to actual joint positions.
        
        Args:
            normalized_action: Array of normalized actions in [-1, 1]
            
        Returns:
            Array of joint positions in radians within joint limits
        """
        # Ensure the action is clipped to [-1, 1] for safety
        clipped_action = np.clip(normalized_action, -1.0, 1.0)
        
        # Initialize joint positions
        joint_positions = np.zeros(self.shape, dtype=np.float32)
        
        # Convert normalized values to actual joint positions
        for joint_idx in range(len(joint_positions)):
            if joint_idx in self.joint_limits:
                mid = self.joint_mids[joint_idx]
                rng = self.joint_ranges[joint_idx]
                
                # Unnormalize to joint position
                joint_positions[joint_idx] = mid + clipped_action[joint_idx] * rng
            else:
                # For joints without defined limits, use the normalized value directly
                joint_positions[joint_idx] = clipped_action[joint_idx]
        
        return joint_positions

class RobotPositioningRevampedEnv(gym.Env):
    """
    Revamped environment for robot end-effector positioning task with sophisticated
    reward engineering, alternative action representation, and better state encoding.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 gui=True, 
                 gui_delay=0.0, 
                 workspace_size=0.7, 
                 clean_viz=False, 
                 viz_speed=0.0, 
                 verbose=False, 
                 parallel_viz=False, 
                 rank=0, 
                 offset_x=0.0,
                 training_mode=True):
        """
        Initialize the robot positioning environment with improvements.
        
        Args:
            gui: Whether to use GUI visualization
            gui_delay: Delay in seconds between steps (for visualization)
            workspace_size: Size of the workspace (in meters)
            clean_viz: Whether to clean up visualization artifacts
            viz_speed: Speed of visualization (seconds between steps)
            verbose: Whether to print verbose output
            parallel_viz: Whether this is used in parallel visualization mode
            rank: The rank of this robot in parallel training
            offset_x: X-axis offset for parallel robots
            training_mode: Whether the environment is used for training (affects exploration)
        """
        super().__init__()
        
        # Store parameters
        self.gui = gui
        self.gui_delay = gui_delay
        self.workspace_size = workspace_size
        self.clean_viz = clean_viz
        self.viz_speed = viz_speed
        self.verbose = verbose
        self.parallel_viz = parallel_viz
        self.rank = rank
        self.offset_x = offset_x
        self.training_mode = training_mode
        
        # Initialize PyBullet client
        self.client_id = get_shared_pybullet_client(render=gui)
        
        # Create the robot environment
        self.robot = FANUCRobotEnv(render=gui, verbose=verbose, client=self.client_id)
        
        # Apply offset if needed (for parallel robots)
        self.robot_offset = np.array([offset_x, 0.0, 0.0])
        self._apply_robot_offset()
        
        # Get robot's degrees of freedom
        self.dof = self.robot.dof
        
        # Initialize home position (robot's shoulder/base)
        self.home_position = np.array([0.0, 0.0, 0.0]) + self.robot_offset
        
        # Setup successful target parameters
        self.accuracy_threshold = 0.015  # 15mm accuracy (tighter than before)
        self.timeout_steps = 150  # Shorter episode length to encourage efficiency
        
        # Curriculum learning parameters
        self.curriculum_level = 0
        self.max_target_distance = 0.3  # Start with easier targets
        self.target_expansion_increment = 0.05  # Increment target distance by 5cm at a time
        self.consecutive_successful_episodes = 0
        self.last_episode_successful = False
        
        # Determine reachable workspace
        max_reach, workspace_bounds = determine_reachable_workspace(
            self.robot, 
            self.home_position, 
            num_samples=1000, 
            verbose=verbose
        )
        
        # Calculate workspace center from bounds
        workspace_center = [
            (workspace_bounds['x'][0] + workspace_bounds['x'][1]) / 2.0,
            (workspace_bounds['y'][0] + workspace_bounds['y'][1]) / 2.0,
            (workspace_bounds['z'][0] + workspace_bounds['z'][1]) / 2.0
        ]
        
        self.workspace_bounds = workspace_bounds
        self.workspace_center = workspace_center
        self.robot_id = self.robot.robot_id
        
        # Initialize target properties
        self.target_position = None
        self.target_visual_id = None
        self.last_target_randomization_time = time.time()  # For target updates
        
        # Initialize state variables
        self.steps = 0
        self.previous_action = np.zeros(self.dof)  # Previous action
        self.previous_distance = None
        self.initial_distance_to_target = None
        self.best_distance_in_episode = None
        self.best_position_in_episode = None
        
        # Reward parameters
        self.total_reward_in_episode = 0.0
        
        # Visualization tracking
        self.ee_markers = []
        self.trajectory_markers = []
        self.marker_creation_steps = {}
        
        # Initialize observation history buffer for temporal information (5 timesteps)
        self.observation_history_length = 5
        empty_observation = np.zeros(self.dof + 7)  # joint positions + ee position + dist to target
        self.observation_history = [empty_observation.copy() for _ in range(self.observation_history_length)]
        
        # Define action space: Use the new JointLimitedBox space instead of regular Box
        # This ensures actions inherently respect joint limits
        self.action_space = JointLimitedBox(self.robot, shape=(self.dof,), dtype=np.float32)
        
        # Define observation space: 
        # - Current normalized joint angles (5)
        # - Current end effector position (3)
        # - Target position (3)
        # - Distance to target (1)
        # - Normalized direction to target (3)
        # - Previous action (5)
        # - Joint position history (5 * history_length)
        # - End effector position history (3 * history_length)
        # Total: 5 + 3 + 3 + 1 + 3 + 5 + (5+3)*5 = 60
        
        # Maximum values for each component
        max_joints = np.ones(self.dof)  # Normalized joint angles (0-1)
        max_position = np.ones(3) * (workspace_size * 2.0)  # Position (meters)
        max_target = np.ones(3) * (workspace_size * 2.0)  # Target position (meters)
        max_distance = np.array([workspace_size * 2.0])  # Distance (meters)
        max_direction = np.ones(3)  # Normalized direction (-1 to 1)
        max_action = np.ones(self.dof) * 1.0  # Assuming max action range is [-1, 1]
        
        # Combine all max values
        max_obs = np.concatenate([
            max_joints,  # 5
            max_position,  # 3
            max_target,  # 3
            max_distance,  # 1
            max_direction,  # 3
            max_action,  # 5
            np.tile(np.concatenate([max_joints, max_position]), self.observation_history_length)  # (5+3)*history_length
        ])
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-max_obs, 
            high=max_obs, 
            dtype=np.float32
        )
        
        # Reset the environment to initialize everything
        self.reset()
        
    def _apply_robot_offset(self):
        """Apply offset to the robot for parallel training"""
        if np.linalg.norm(self.robot_offset) > 0.0:
            # Get the base position
            base_pos, base_orn = p.getBasePositionAndOrientation(
                self.robot.robot_id, 
                physicsClientId=self.client_id
            )
            
            # Apply offset
            new_base_pos = np.array(base_pos) + self.robot_offset
            
            # Set the new base position
            p.resetBasePositionAndOrientation(
                self.robot.robot_id,
                new_base_pos,
                base_orn,
                physicsClientId=self.client_id
            )
    
    def _sample_target(self):
        """Sample a target position within the reachable workspace"""
        # Define limits for x, y, z
        ws_bounds = self.workspace_bounds
        shoulder_position = self.home_position
        
        # We'll pick from different strategies for target sampling
        sampling_strategies = ["uniform", "outer_region", "precision_region", "vertical_stack"]
        strategy_weights = [0.3, 0.4, 0.2, 0.1]  # Weight toward outer regions for more challenge
        
        # Adjust weights based on curriculum level
        if self.curriculum_level < 3:
            # Early levels: more inner workspace targets for easier learning
            strategy_weights = [0.5, 0.2, 0.3, 0.0]
        elif self.curriculum_level < 6:
            # Mid levels: balance between inner and outer workspace
            strategy_weights = [0.4, 0.3, 0.2, 0.1]
        
        # Try different sampling strategies until we get a valid target
        max_attempts = 50
        for _ in range(max_attempts):
            # Choose a sampling strategy
            sampling_strategy = np.random.choice(sampling_strategies, p=strategy_weights)
            
            if sampling_strategy == "uniform":
                # Simple uniform sampling within the workspace bounds
                x = np.random.uniform(ws_bounds['x'][0], ws_bounds['x'][1])
                y = np.random.uniform(ws_bounds['y'][0], ws_bounds['y'][1])
                z = np.random.uniform(ws_bounds['z'][0], ws_bounds['z'][1])
                
                # Limit maximum distance based on curriculum level for easier early targets
                target_position = np.array([x, y, z]) + self.robot_offset
                distance_to_shoulder = np.linalg.norm(target_position - shoulder_position)
                
                if distance_to_shoulder > self.max_target_distance:
                    # Target is too far for current curriculum level, try again
                    continue
                
            elif sampling_strategy == "outer_region":
                # Sample in the outer regions of the workspace (challenging positions)
                # Start with random direction from center
                radius = np.random.uniform(0.7, 1.0) * self.max_target_distance
                phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
                theta = np.random.uniform(0, np.pi)  # Polar angle
                
                # Convert spherical to cartesian
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                
                # Offset from center of workspace
                target_position = np.array([x, y, z]) + shoulder_position
                
            elif sampling_strategy == "precision_region":
                # Sample in a small region around a previous best position
                # This encourages fine-tuning in areas where the robot was almost successful
                if self.best_position_in_episode is not None:
                    # Sample around the best position from the previous episode
                    # with a small radius to encourage precise positioning
                    radius = np.random.uniform(0.05, 0.15)
                    phi = np.random.uniform(0, 2 * np.pi)
                    theta = np.random.uniform(0, np.pi)
                    
                    # Convert spherical to cartesian
                    x = radius * np.sin(theta) * np.cos(phi)
                    y = radius * np.sin(theta) * np.sin(phi)
                    z = radius * np.cos(theta)
                    
                    # Offset from best position
                    target_position = np.array([x, y, z]) + self.best_position_in_episode
                else:
                    # Fall back to uniform if no best position
                    x = np.random.uniform(ws_bounds['x'][0], ws_bounds['x'][1])
                    y = np.random.uniform(ws_bounds['y'][0], ws_bounds['y'][1])
                    z = np.random.uniform(ws_bounds['z'][0], ws_bounds['z'][1])
                    target_position = np.array([x, y, z]) + self.robot_offset
            
            elif sampling_strategy == "vertical_stack":
                # Sample in a vertical stack configuration
                # This is useful for learning to reach up/down
                x = np.random.uniform(-0.2, 0.2) + shoulder_position[0]
                y = np.random.uniform(-0.2, 0.2) + shoulder_position[1]
                z = np.random.uniform(ws_bounds['z'][0], ws_bounds['z'][1])
                target_position = np.array([x, y, z])
            
            # Check if the target is valid (above the ground and not in the robot base)
            is_above_ground = target_position[2] > 0.05  # 5cm above ground
            
            # Check if target is not inside the robot base
            base_radius = 0.2  # Approximate robot base radius
            xy_distance_from_shoulder = np.linalg.norm(target_position[:2] - shoulder_position[:2])
            is_not_in_base = (xy_distance_from_shoulder > base_radius) or (target_position[2] > 0.3)
            
            if is_above_ground and is_not_in_base:
                if self.verbose:
                    print(f"Valid target position sampled at {target_position} using {sampling_strategy} strategy")
                return target_position
        
        # Fallback strategy if all else fails
        random_angle = np.random.uniform(0, 2 * np.pi)
        fallback_position = shoulder_position + np.array([
            0.4 * np.cos(random_angle),
            0.4 * np.sin(random_angle),
            0.2
        ])
        
        if self.verbose:
            print(f"Using fallback target position at {fallback_position}")
        
        return fallback_position
    
    def _get_observation(self):
        """
        Get the current observation state with improved features for learning.
        """
        # Get current state from robot environment
        state = self.robot._get_state()
        
        # Extract joint positions
        joint_positions = state[:self.robot.dof*2:2]  # Extract joint positions
        
        # Normalize joint positions to 0-1 range
        normalized_joint_positions = []
        for i, pos in enumerate(joint_positions):
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                # Calculate normalized position with full precision
                norm_pos = (pos - limit_low) / (limit_high - limit_low)
                # Cap to [0, 1] range to prevent numerical issues
                norm_pos = max(0.0, min(1.0, norm_pos))
                normalized_joint_positions.append(norm_pos)
            else:
                # Default for joints without limits
                normalized_joint_positions.append(0.5)
        
        # Convert to numpy array
        normalized_joint_positions = np.array(normalized_joint_positions, dtype=np.float32)
        
        # Extract end effector position
        ee_position = state[12:15]
        
        # Calculate relative position vector (target - ee)
        relative_position = self.target_position - ee_position
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(relative_position)
        
        # Calculate normalized direction vector to target
        if distance_to_target > 1e-6:
            direction_to_target = relative_position / distance_to_target
        else:
            direction_to_target = np.zeros(3)
        
        # Create current basic observation
        current_basic_observation = np.concatenate([
            normalized_joint_positions,   # Current joint positions (normalized 0-1)
            ee_position,                  # Current end effector position
        ])
        
        # Update observation history
        self.observation_history.pop(0)  # Remove oldest observation
        self.observation_history.append(current_basic_observation)  # Add newest observation
        
        # Create the full observation including history
        observation = np.concatenate([
            normalized_joint_positions,        # Current normalized joint angles
            ee_position,                       # Current end effector position
            self.target_position,              # Target position
            [distance_to_target],              # Distance to target
            direction_to_target,               # Normalized direction to target
            self.previous_action,              # Previous action
            np.concatenate(self.observation_history)  # History of observations
        ]).astype(np.float32)
        
        return observation
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the robot
        self.robot.reset()
        
        # Always sample a new target position on reset
        self.target_position = self._sample_target()
        
        if self.verbose:
            print(f"Robot {self.rank} received a new target at {self.target_position}")
        
        # Visualize the target if rendering is enabled
        if self.gui:
            try:
                # Remove previous target visualization if it exists
                if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
                    try:
                        p.removeBody(self.target_visual_id, physicsClientId=self.client_id)
                    except:
                        pass
                
                # Create new target visualization
                self.target_visual_id = visualize_target(self.target_position, self.client_id)
                
            except Exception as e:
                print(f"Warning: Could not visualize target: {e}")
        
        # Reset step counter
        self.steps = 0
        
        # Reset total reward for this episode
        self.total_reward_in_episode = 0.0
        
        # Get current state
        state = self.robot._get_state()
        ee_position = state[12:15]
        
        # Calculate and store initial distance to target
        self.initial_distance_to_target = np.linalg.norm(ee_position - self.target_position)
        
        # Reset best position tracking
        self.best_distance_in_episode = self.initial_distance_to_target
        self.best_position_in_episode = ee_position.copy()
        
        # Set previous distance to current distance for first reward calculation
        self.previous_distance = self.initial_distance_to_target
        
        # Reset previous action
        self.previous_action = np.zeros(self.dof)
        
        # Reset observation history
        empty_observation = np.zeros(self.dof + 3)  # joint positions + ee position
        self.observation_history = [empty_observation.copy() for _ in range(self.observation_history_length)]
        
        # Initialize observation history with current state
        current_basic_observation = np.concatenate([
            np.zeros(self.dof),  # Placeholder normalized joint positions
            ee_position,         # Current end effector position
        ])
        
        # Fill observation history with current state
        for i in range(self.observation_history_length):
            self.observation_history[i] = current_basic_observation.copy()
        
        # Get observation
        observation = self._get_observation()
        
        if self.verbose:
            print(f"Robot {self.rank}: Initial distance to target: {self.initial_distance_to_target*100:.2f}cm")
        
        # Return observation and info dict (Gymnasium API)
        return observation, {'initial_distance': self.initial_distance_to_target}
    
    def step(self, action):
        """
        Apply action to the robot using direct joint position control with built-in joint limit enforcement.
        
        Args:
            action: Normalized actions in [-1, 1] for each joint
        """
        # The action is now in the range [-1, 1] for each joint
        # Convert to actual joint positions within the joint limits
        joint_positions = self.action_space.unnormalize_action(action)
        
        # Add small random noise to joint positions (exploration during training)
        if self.training_mode:
            # Add tiny noise to help explore the space more thoroughly
            noise_scale = 0.001  # Very small noise
            joint_positions += np.random.normal(0, noise_scale, size=len(joint_positions))
        
        # Check if positions would exceed limits (should never happen with JointLimitedBox)
        # This is purely for monitoring and diagnostics
        positions_within_limits = True
        for i, pos in enumerate(joint_positions):
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                if pos < limit_low or pos > limit_high:
                    positions_within_limits = False
                    if self.verbose:
                        print(f"Warning: Action would exceed joint {i} limits: {pos:.4f} not in [{limit_low:.4f}, {limit_high:.4f}]")
                    # Force within limits (should be unnecessary with JointLimitedBox)
                    joint_positions[i] = np.clip(pos, limit_low, limit_high)
        
        # Apply the joint positions with zero velocities
        zero_velocities = [0.0] * len(joint_positions)
        next_state = self.robot.step((joint_positions, zero_velocities))
        
        # Store the current action for next observation
        self.previous_action = np.array(action)
        
        # Increment step counter
        self.steps += 1
        
        # Get current end effector position
        state = self.robot._get_state()
        current_ee_pos = state[12:15]
        
        # Calculate distance to target
        distance = np.linalg.norm(current_ee_pos - self.target_position)
        
        # Create info dictionary for debugging and monitoring
        info = {
            "distance_cm": distance * 100,
            "joint_positions": joint_positions,
            "positions_within_limits": positions_within_limits
        }
        
        # Determine if target is reached
        target_reached = distance <= self.accuracy_threshold
        info["target_reached"] = target_reached
        
        # Calculate reward
        reward = self._calculate_reward(distance, current_ee_pos, action, positions_within_limits)
        
        # Update previous distance for next step's calculation
        self.previous_distance = distance
        
        # Check episode termination conditions
        if target_reached:
            done = True
            # Add large bonus reward for reaching the target
            success_bonus = 10.0
            reward += success_bonus
            info["success"] = True
            
            # Check for curriculum advancement
            episode_successful = True
            if episode_successful:
                if self.last_episode_successful:
                    self.consecutive_successful_episodes += 1
                    
                    # Expand target boundary if not at maximum
                    if self.max_target_distance < self.workspace_size:
                        self.max_target_distance = min(
                            self.max_target_distance + self.target_expansion_increment,
                            self.workspace_size
                        )
                        self.curriculum_level += 1
                        
                        if self.verbose:
                            print(f"\n{'='*60}")
                            print(f"CURRICULUM LEVEL UP! Now at level {self.curriculum_level}")
                            print(f"Target boundary expanded to: {self.max_target_distance:.2f}m")
                            print(f"{'='*60}\n")
                
                self.last_episode_successful = True
            else:
                self.last_episode_successful = False
                
        else:
            # Update best distance if this is closer than before
            if distance < self.best_distance_in_episode:
                self.best_distance_in_episode = distance
                self.best_position_in_episode = current_ee_pos.copy()
            
            # Check timeout
            if self.steps >= self.timeout_steps:
                done = True
                info["timeout"] = True
            else:
                done = False
        
        # Store detailed info for analysis
        info["reward"] = reward
        info["distance"] = distance
        info["best_distance"] = self.best_distance_in_episode
        
        # Update total reward
        self.total_reward_in_episode += reward
        info["total_reward"] = self.total_reward_in_episode
        
        # Get observation for next step
        observation = self._get_observation()
        
        # Return according to Gymnasium API
        return observation, reward, done, False, info
    
    def _calculate_reward(self, distance, current_ee_pos, action, positions_within_limits=True):
        """Calculate reward based on distance to target, progress, and action penalties"""
        # 1. Distance-based reward
        distance_reward = self._distance_reward(distance)
        
        # 2. Progress reward - reward for movement that reduces distance to target
        progress_reward = 0.0
        if hasattr(self, 'last_distance'):
            # Calculate improvement in distance
            improvement = self.last_distance - distance
            # Scale improvement to create a smooth reward
            progress_reward = improvement * 20.0  # Scale to make it significant
            # Clip to avoid extreme values
            progress_reward = np.clip(progress_reward, -1.0, 1.0)
        
        # 3. Action penalty - small penalty for unnecessary movement
        # Encourages finding efficient paths to the target
        action_penalty = -np.sum(np.abs(action)) * 0.01
        
        # Combine all reward components with simplified weights
        reward = distance_reward + progress_reward + action_penalty
        
        return reward
    
    def close(self):
        """Close the environment"""
        # Clean up visualization if needed
        if self.gui and self.clean_viz:
            if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
                try:
                    p.removeBody(self.target_visual_id, physicsClientId=self.client_id)
                except:
                    pass
            
            for marker_id in self.trajectory_markers:
                try:
                    p.removeBody(marker_id, physicsClientId=self.client_id)
                except:
                    pass

# Custom neural network architectures for better learning
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Simplified feature extractor with a more coherent architecture.
    Processes observations in a more direct way with better parameter sharing.
    """
    def __init__(self, observation_space: spaces.Box):
        # Use a smaller feature dimension for efficiency
        features_dim = 256
        super().__init__(observation_space, features_dim=features_dim)
        
        # Determine observation dimensions
        obs_dim = observation_space.shape[0]
        print(f"Observation space dimension: {obs_dim}")
        
        # Number of joints (DOF) in the environment (5 for FANUC robot)
        self.dof = 5
        
        # Simplified architecture: 2-layer MLP with layer normalization
        self.shared_network = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process the observations with the simplified network
        """
        # Forward pass through the shared network
        features = self.shared_network(observations)
        return features

class CustomActorNetwork(nn.Module):
    """
    Custom actor network with improved architecture for precise robot control.
    Uses residual connections and deeper layers for better gradient flow.
    The network's output is explicitly designed to respect joint limits inherently.
    """
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        
        # Main network with residual connections
        self.fc1 = nn.Linear(feature_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        
        # Output layer for mean
        self.mean_out = nn.Linear(128, action_dim)
        
        # Log standard deviation layer with learned parameters
        # Starting with a lower initial value for more precise actions and
        # to reduce sampling outside of valid regions
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        
        # Additional layer to scale the exploration as we get closer to limits
        # This helps the model learn to avoid sampling actions near the boundaries
        self.adaptive_std_scale = nn.Linear(feature_dim, action_dim)
    
    def forward(self, features):
        # First layer
        x = F.relu(self.ln1(self.fc1(features)))
        
        # Residual block 1
        residual = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.ln3(self.fc3(x))
        x = F.relu(x + residual)  # Residual connection
        
        # Final layers
        x = F.relu(self.ln4(self.fc4(x)))
        
        # Output mean with tanh to keep values in [-1, 1] range
        # We scale in the env for actual action range
        mean = torch.tanh(self.mean_out(x))
        
        # Apply adaptive standard deviation scaling
        # This allows the model to learn to reduce exploration near limits
        std_scaling = torch.sigmoid(self.adaptive_std_scale(features))
        
        # Base log standard deviation, scaled by the adaptive factor
        # This helps prevent sampling outside valid regions
        log_std = self.log_std * (0.5 + 0.5 * std_scaling)
        
        # Clamp log_std for numerical stability while allowing
        # sufficient exploration in the middle of the joint range
        log_std = torch.clamp(log_std, -20.0, 0.0)
        
        return mean, log_std

class CustomCriticNetwork(nn.Module):
    """
    Custom critic network with improved architecture for better value estimation.
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # Main network with residual connections
        self.fc1 = nn.Linear(feature_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        
        # Output layer
        self.value_out = nn.Linear(128, 1)
    
    def forward(self, features):
        # First layer
        x = F.relu(self.ln1(self.fc1(features)))
        
        # Residual block 1
        residual = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.ln3(self.fc3(x))
        x = F.relu(x + residual)  # Residual connection
        
        # Final layers
        x = F.relu(self.ln4(self.fc4(x)))
        
        # Output value
        value = self.value_out(x)
        
        return value

# Custom policy for PPO with the new architecture
class CustomActorCriticPolicy(nn.Module):
    """
    Custom actor-critic policy for PPO with our improved architectures.
    Combines the feature extractor, actor, and critic networks.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Feature extractor
        self.features_extractor = CustomFeatureExtractor(observation_space)
        feature_dim = self.features_extractor.features_dim
        
        # Actor network
        self.actor = CustomActorNetwork(feature_dim, action_space.shape[0])
        
        # Critic network
        self.critic = CustomCriticNetwork(feature_dim)
    
    def forward(self, obs):
        # Extract features
        features = self.features_extractor(obs)
        
        # Get action distribution parameters
        mean, log_std = self.actor(features)
        
        # Get value estimate
        value = self.critic(features)
        
        return mean, log_std, value

# Callback for saving models during training
class SaveModelCallback(BaseCallback):
    """
    Callback for saving models during training.
    """
    def __init__(self, save_freq=None, check_freq=None, save_path=None, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        # Support both save_freq (legacy) and check_freq (new)
        self.save_freq = save_freq if save_freq is not None else check_freq
        self.save_path = save_path
    
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            save_path = f"{self.save_path}/model_{self.n_calls}_steps"
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Model saved to {save_path}")
        return True

# Callback for training monitoring and visualization
class TrainingMonitorCallback(BaseCallback):
    """
    Callback for monitoring training progress and creating visualizations.
    """
    def __init__(self, log_interval=100, verbose=1, plot_interval=10000, plot_dir=None, model_dir=None):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        
        # Support custom plot and model directories
        self.plot_dir = plot_dir if plot_dir is not None else "./plots"
        self.model_dir = model_dir if model_dir is not None else "./models"
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.timesteps = []
        self.rewards = []
        self.distances = []
        self.success_rates = []
        self.episode_lengths = []
        
        # Metrics for the last N episodes
        self.last_n_episodes = 100
        self.recent_rewards = []
        self.recent_distances = []
        self.recent_successes = []
        self.recent_lengths = []
    
    def _on_step(self):
        # Get info from the most recent episode
        if len(self.model.ep_info_buffer) > 0:
            # Process all new episodes since last check
            while len(self.recent_rewards) < len(self.model.ep_info_buffer):
                ep_idx = len(self.recent_rewards)
                if ep_idx < len(self.model.ep_info_buffer):
                    ep_info = self.model.ep_info_buffer[ep_idx]
                    
                    # Extract metrics
                    reward = ep_info.get('r', 0.0)
                    length = ep_info.get('l', 0)
                    success = ep_info.get('success', False)
                    distance = ep_info.get('final_distance', float('inf'))
                    
                    # Store metrics
                    self.recent_rewards.append(reward)
                    self.recent_lengths.append(length)
                    self.recent_successes.append(1.0 if success else 0.0)
                    self.recent_distances.append(distance)
                    
                    # Keep only last N episodes
                    if len(self.recent_rewards) > self.last_n_episodes:
                        self.recent_rewards.pop(0)
                        self.recent_lengths.pop(0)
                        self.recent_successes.pop(0)
                        self.recent_distances.pop(0)
        
        # Log at regular intervals
        if self.n_calls % self.log_interval == 0:
            if len(self.recent_rewards) > 0:
                avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                avg_length = sum(self.recent_lengths) / len(self.recent_lengths)
                success_rate = sum(self.recent_successes) / len(self.recent_successes)
                avg_distance = sum(self.recent_distances) / len(self.recent_distances)
                
                # Store for plotting
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(avg_reward)
                self.distances.append(avg_distance)
                self.success_rates.append(success_rate)
                self.episode_lengths.append(avg_length)
                
                # Log to console
                print(f"Steps: {self.num_timesteps}, "
                      f"Avg reward: {avg_reward:.2f}, "
                      f"Success rate: {success_rate*100:.1f}%, "
                      f"Avg distance: {avg_distance*100:.2f}cm, "
                      f"Avg length: {avg_length:.1f}")
        
        # Plot at regular intervals
        if self.n_calls % self.plot_interval == 0 and len(self.timesteps) > 1:
            self.plot_training_progress(f"{self.plot_dir}/progress_{self.num_timesteps}.png")
            
            # Save checkpoint too
            save_path = f"{self.model_dir}/checkpoint_{self.num_timesteps}"
            self.model.save(save_path)
            print(f"Checkpoint saved to {save_path}")
        
        return True
    
    def plot_training_progress(self, save_path):
        """Plot training progress metrics"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot rewards
            axs[0, 0].plot(self.timesteps, self.rewards)
            axs[0, 0].set_title('Average Reward')
            axs[0, 0].set_xlabel('Timesteps')
            axs[0, 0].set_ylabel('Reward')
            axs[0, 0].grid(True)
            
            # Plot distances
            axs[0, 1].plot(self.timesteps, [d*100 for d in self.distances])  # Convert to cm
            axs[0, 1].set_title('Average Distance to Target (cm)')
            axs[0, 1].set_xlabel('Timesteps')
            axs[0, 1].set_ylabel('Distance (cm)')
            axs[0, 1].grid(True)
            
            # Plot success rate
            axs[1, 0].plot(self.timesteps, [s*100 for s in self.success_rates])  # Convert to %
            axs[1, 0].set_title('Success Rate (%)')
            axs[1, 0].set_xlabel('Timesteps')
            axs[1, 0].set_ylabel('Success Rate (%)')
            axs[1, 0].grid(True)
            
            # Plot episode lengths
            axs[1, 1].plot(self.timesteps, self.episode_lengths)
            axs[1, 1].set_title('Average Episode Length')
            axs[1, 1].set_xlabel('Timesteps')
            axs[1, 1].set_ylabel('Steps')
            axs[1, 1].grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Training progress plot saved to {save_path}")
        except Exception as e:
            print(f"Error plotting training progress: {e}")

# Custom PPO algorithm wrapper that uses our custom policy
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

# Function to create environments for training
def create_revamped_envs(num_envs=1, viz_speed=0.0, parallel_viz=False, training_mode=True):
    """
    Create multiple instances of the revamped robot positioning environment.
    
    Args:
        num_envs: Number of environments to create
        viz_speed: Speed of visualization (seconds between steps)
        parallel_viz: Whether to use parallel visualization mode
        training_mode: Whether the environments are used for training (affects exploration)
    
    Returns:
        List of environment instances
    """
    # Create environments based on parameters
    envs = []
    
    if parallel_viz:
        # Calculate offsets for multiple robots in same visualization
        client_id = get_shared_pybullet_client(render=True)
        for i in range(num_envs):
            offset_x = i * 1.0  # 1m spacing between robots
            env = RobotPositioningRevampedEnv(
                gui=True,
                viz_speed=viz_speed,
                verbose=False,
                parallel_viz=True,
                rank=i,
                offset_x=offset_x,
                training_mode=training_mode
            )
            
            
                
            envs.append(env)
        
        # Adjust camera to see all robots
        adjust_camera_for_robots(client_id, num_envs)
    else:
        # Create separate environments
        for i in range(num_envs):
            # First env is GUI if visualization is enabled
            is_gui = (i == 0 and viz_speed > 0.0)
            env = RobotPositioningRevampedEnv(
                gui=is_gui,
                viz_speed=viz_speed if is_gui else 0.0,
                verbose=False,
                rank=i,
                training_mode=training_mode
            )
            
            
                
            envs.append(env)
    
    return envs

def train_revamped_robot(args):
    """
    Train a robot with a consolidated, hardware-agnostic approach.
    Standardized hyperparameters and memory-efficient implementations.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create environments based on available hardware
        envs = create_revamped_envs(
            num_envs=args.parallel,
            viz_speed=args.viz_speed if args.gui else 0.0,
        parallel_viz=args.parallel_viz and args.gui,
            training_mode=True
        )
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    
    # Normalize observations and rewards with standardized parameters
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    )
    
    # Standard policy configuration that works well across hardware platforms
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "activation_fn": nn.ReLU,
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),  # Smaller networks for better efficiency
        "ortho_init": True,  # Use orthogonal initialization for better training stability
    }
    
    # Choose device based on hardware availability
    if args.use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device for training")
    else:
        device = "cpu"
        print(f"Using CPU for training")
    
    # Create directories for models and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./models/ppo_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    plot_dir = f"./plots/ppo_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load existing model or create a new one with standard hyperparameters
    if args.load:
        print(f"Loading model from {args.load}")
        model = PPO.load(args.load, env=vec_env)
    else:
        print("Creating new PPO model with standardized hyperparameters")
        
        # Standard PPO parameters that work well across different hardware
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,  # Standard learning rate for PPO
            n_steps=2048,        # Standard rollout length for PPO
            batch_size=64,       # Smaller batch size for better memory efficiency
            n_epochs=10,         # Standard number of epochs
            gamma=0.99,          # Standard discount factor
            gae_lambda=0.95,     # Standard GAE lambda
            clip_range=0.2,      # Standard PPO clip range
            normalize_advantage=True,
            ent_coef=0.01,       # Slightly higher entropy for better exploration
            vf_coef=0.5,         # Standard value function coefficient
            max_grad_norm=0.5,   # Standard gradient clipping
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,
            tensorboard_log=f"{model_dir}/tensorboard"
        )
    
    # Create callbacks for model saving and monitoring
    save_callback = SaveModelCallback(
        check_freq=10000,
        save_path=model_dir,
        verbose=1
    )
    
    monitor_callback = TrainingMonitorCallback(
        log_interval=1000,
        plot_interval=10000,
        plot_dir=plot_dir,
        model_dir=model_dir
    )
    
    # Create a callback for monitoring joint limits
    joint_limit_monitor = JointLimitMonitorCallback(
        log_interval=5000,
        verbose=1
    )
    
    # Enable deterministic garbage collection for better memory management
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive GC thresholds
    
    print(f"Starting training with {args.steps} total timesteps")
    print(f"Logs and models will be saved to {model_dir}")
    print(f"Plot files will be saved to {plot_dir}")
    
    # Start training with all callbacks
    model.learn(
        total_timesteps=args.steps,
        callback=[save_callback, monitor_callback, joint_limit_monitor]
    )
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save normalization statistics
    vec_normalize_path = f"{model_dir}/vec_normalize_stats"
    vec_env.save(vec_normalize_path)
    print(f"Normalization statistics saved to {vec_normalize_path}")
    
    # Force garbage collection before closing environments
    gc.collect()
    
    # Close environment
    vec_env.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a robot arm for precise end effector positioning (Revamped)')
    parser.add_argument('--steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--load', type=str, default=None, help='Load a pre-trained model to continue training')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation on a pre-trained model')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--demo', action='store_true', help='Run a demonstration sequence with the model')
    parser.add_argument('--save-video', action='store_true', help='Save a video of the evaluation')
    parser.add_argument('--gui', action='store_true', default=True, help='Enable GUI visualization')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI visualization')
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--parallel-viz', action='store_true', help='Enable parallel visualization')
    parser.add_argument('--viz-speed', type=float, default=0.0, help='Control visualization speed (delay in seconds)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use CUDA for training if available')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--algorithm', choices=['ppo'], default='ppo', help='RL algorithm to use')
    # Add missing training parameters
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps for each update')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs when optimizing')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip-range', type=float, default=0.2, help='Clipping parameter for PPO')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--save-freq', type=int, default=50000, help='Frequency to save model')
    parser.add_argument('--eval-after-training', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Handle gui/no-gui conflict
    if args.no_gui:
        args.gui = False
    
    # Determine the device to use (CUDA or CPU)
    if args.use_cuda and torch.cuda.is_available():
        args.device = "cuda"
        print("Using CUDA for training (GPU acceleration)")
    else:
        args.device = "cpu"
        print("Using CPU for training (default setting, can be overridden with command line arguments)")
    
    return args

def main():
    """
    Entry point for the revamped robot training script.
    Parses command line arguments and either trains a new model or evaluates an existing one.
    """
    args = parse_args()
    
    # Seed if requested
    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Check if we're just evaluating
    if args.eval_only:
        if args.load:
            try:
                if args.demo:
                    # Run evaluation sequence for demo
                    run_evaluation_sequence(
                        model_path=args.load,
                        viz_speed=args.viz_speed if args.viz_speed > 0 else 0.02,
                        save_video=args.save_video
                    )
                else:
                    # Evaluate model with standard evaluation
                    evaluate_model_wrapper(
                        model_path=args.load,
                        num_episodes=args.eval_episodes,
                        visualize=args.gui,
                        verbose=args.verbose
                    )
            except Exception as e:
                import traceback
                error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
                print(f"Error occurred during evaluation: {e}")
        else:
            print("Error: --eval-only requires a model path specified with --load")
            return  # Exit after evaluation
    
    # Training mode from here on
    # Create environments
    env = create_revamped_envs(
        num_envs=args.parallel,
        viz_speed=args.viz_speed if args.gui else 0.0,
        parallel_viz=args.parallel_viz and args.gui,
        training_mode=True
    )
    
    # Normalization wrapper for observations
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create policy
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "activation_fn": torch.nn.ReLU,
        "net_arch": [dict(pi=[256, 128, 64], vf=[256, 128, 64])],
    }
    
    # Compute appropriate batch size based on n_steps and parallel environments
    # This ensures we use full trajectories
    n_steps_per_env = args.n_steps // args.parallel
    batch_size = min(args.batch_size, n_steps_per_env * args.parallel)
    
    # Ensure batch size is compatible with n_steps
    if n_steps_per_env * args.parallel % batch_size != 0:
        # Adjust to nearest compatible batch size
        old_batch_size = batch_size
        batch_size = n_steps_per_env * args.parallel // (n_steps_per_env * args.parallel // batch_size)
        print(f"Adjusted batch size from {old_batch_size} to {batch_size} for compatibility with n_steps")
    
    # Print training parameters
    print("\nTraining Parameters:")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Timesteps: {args.steps}")
    print(f"n_steps: {args.n_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"Parallel Environments: {args.parallel}")
    print(f"Gamma: {args.gamma}")
    print(f"GAE Lambda: {args.gae_lambda}")
    print(f"Clip Range: {args.clip_range}")
    print(f"VF Coefficient: {args.vf_coef}")
    print(f"Save Frequency: {args.save_freq}")
    print("Architecture:", policy_kwargs["net_arch"])
    print()
    
    # Create the model timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./models/revamped_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create callbacks for saving and monitoring
    callbacks = [
        SaveModelCallback(
            save_freq=args.save_freq // args.parallel,  # Save every n steps, adjusted for parallel envs
            save_path=model_dir,
            verbose=1,
        ),
        TrainingMonitorCallback(
            log_interval=100,
            verbose=args.verbose,
            plot_interval=args.save_freq,
            plot_dir=f"{model_dir}/plots",
            model_dir=model_dir
        ),
        JointLimitMonitorCallback(
            log_interval=5000, 
            verbose=args.verbose
        )
    ]
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=n_steps_per_env,  # Steps per environment
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        verbose=args.verbose,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        tensorboard_log=f"{model_dir}/tb_logs"
    )
    
    # Load pretrained model if specified
    if args.load:
        try:
            print(f"Loading model from {args.load}")
            # Handle both paths with and without .zip extension
            load_path = args.load
            if not os.path.exists(load_path) and os.path.exists(load_path + ".zip"):
                load_path = load_path + ".zip"
            
            # Load the model
            model = PPO.load(
                load_path,
                env=env, 
                device=args.device,
                custom_objects={
                    "learning_rate": args.learning_rate,
                    "n_steps": n_steps_per_env,  
                    "batch_size": batch_size,
                    "n_epochs": args.n_epochs,
                    "clip_range": args.clip_range,
                    "ent_coef": 0.0,
                    "vf_coef": args.vf_coef,
                    "max_grad_norm": 0.5,
                }
            )
            print("Model loaded successfully!")
            
            # Also check for normalization stats
            norm_path = args.load.replace("final_model", "vec_normalize_stats")
            if os.path.exists(norm_path):
                print(f"Loading normalization stats from {norm_path}")
                env = VecNormalize.load(norm_path, env)
                # Don't update stats during training, we want to preserve the loaded stats
                env.training = True  # Set to true as we are continuing training
                print("Normalization stats loaded successfully!")
    except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a fresh model...")
    
    # Train the model
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        log_interval=1,  # Print stats every n updates (but our callback handles actual logging)
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    
    # Save normalization statistics
    norm_path = os.path.join(model_dir, "vec_normalize_stats")
    env.save(norm_path)
    
    print(f"\nTraining completed. Final model saved to {final_model_path}")
    
    # Run a final evaluation
    if args.eval_after_training:
        print("\nRunning final evaluation...")
        evaluate_model_wrapper(model_path=final_model_path, num_episodes=20, visualize=args.gui and args.viz_speed > 0, verbose=args.verbose)

def evaluate_model(env, policy, rollout_steps, num_episodes=10, render=False, verbose=True, 
                   model_file=None, render_cb=None, directml=False, info_prefix=""):
    """
    Evaluate a model using a specified policy.
    
    Args:
        env: Environment to evaluate in
        policy: Policy to use for evaluation
        rollout_steps: Maximum number of steps per episode
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Whether to print verbose information
        model_file: Path to model file (for tracking purposes)
        render_cb: Optional callback for custom rendering
        directml: Whether this is a DirectML model
        info_prefix: Prefix for info messages
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Information to track during evaluation
    total_rewards = []
    episode_lengths = []
    success_count = 0
    avg_distance = 0
    best_distance = float('inf')
    worst_distance = 0
    all_distances = []
    all_rewards = []
    all_joint_limits = []
    
    # For tracking
    episode_data = []
    
    # Check if render is supported
    render_supported = render and hasattr(env, 'render')
    
    # For each episode
    for i in range(num_episodes):
        if verbose:
            print(f"{info_prefix}Episode {i+1}/{num_episodes}...")
            
        # Reset the environment
        observation, _ = env.reset()
        
        # Start at zero reward
        total_reward = 0
        
        # For tracking the episode
        episode_rewards = []
        episode_obs = []
        episode_actions = []
        episode_dones = []
        episode_infos = []
        
        # Whether the episode was a success
        episode_success = False
        episode_distance = float('inf')
        within_limits = True
        
        # Run a single episode
        for t in range(rollout_steps):
            # Store observation
            episode_obs.append(observation)
            
            # If we should render, do so
            if render_supported:
                try:
                    env.render()
                except (NotImplementedError, Exception) as e:
                    print(f"Warning: Rendering not supported or failed: {e}")
                    render_supported = False  # Disable rendering for future steps
            
            # Use the policy to predict an action
            if directml:
                action, _ = policy.predict(observation, deterministic=True)
    else:
                action, _ = policy.predict(observation)
                
            # Store action
            episode_actions.append(action)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store reward and done
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_infos.append(info)
            
            # Add to total reward
            total_reward += reward
            
            # Custom rendering callback
            if render_cb:
                render_cb(env, t, info)
                
            # If done, break
            if done:
                break
                
        # Store the final observation
        episode_obs.append(observation)
        
        # Track success
        if 'success' in info and info['success']:
            episode_success = True
            success_count += 1
            
        # Track distance
        if 'distance' in info:
            episode_distance = info['distance']
            avg_distance += episode_distance
            all_distances.append(episode_distance)
            best_distance = min(best_distance, episode_distance)
            worst_distance = max(worst_distance, episode_distance)
            
        # Track joint limits
        if 'positions_within_limits' in info:
            within_limits = info['positions_within_limits']
            all_joint_limits.append(within_limits)
            
        # Add the episode information
        episode_data.append({
            'obs': episode_obs,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'dones': episode_dones,
            'infos': episode_infos,
            'total_reward': total_reward,
            'success': episode_success,
            'distance': episode_distance,
            'within_limits': within_limits,
            'length': len(episode_rewards)
        })
        
        # Add to metrics
        total_rewards.append(total_reward)
        episode_lengths.append(len(episode_rewards))
        all_rewards.extend(episode_rewards)
        
        # Print verbose information
        if verbose:
            success_str = "Success" if episode_success else "Failure"
            distance_str = f"{episode_distance:.2f} cm" if episode_distance != float('inf') else "N/A"
            limits_str = "Within limits" if within_limits else "Exceeded limits"
            print(f"{info_prefix}  {success_str}, Distance: {distance_str}, Steps: {len(episode_rewards)}, Reward: {total_reward:.2f}, {limits_str}")
    
    # Calculate metrics
    success_rate = success_count / num_episodes
    avg_reward = sum(total_rewards) / num_episodes
    avg_episode_length = sum(episode_lengths) / num_episodes
    avg_distance = avg_distance / num_episodes if num_episodes > 0 else float('inf')
    
    # Calculate statistics on rewards
    reward_mean = np.mean(all_rewards) if len(all_rewards) > 0 else 0
    reward_std = np.std(all_rewards) if len(all_rewards) > 0 else 0
    reward_min = min(all_rewards) if len(all_rewards) > 0 else 0
    reward_max = max(all_rewards) if len(all_rewards) > 0 else 0
    
    # Print overall results
    if verbose:
        print("\n" + "="*50)
        print(f"{info_prefix}Evaluation Results:")
        print(f"{info_prefix}  Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
        print(f"{info_prefix}  Average Reward: {avg_reward:.2f}")
        print(f"{info_prefix}  Average Episode Length: {avg_episode_length:.1f} steps")
        if all_distances:
            print(f"{info_prefix}  Average Distance: {avg_distance:.2f} cm")
            print(f"{info_prefix}  Best Distance: {best_distance:.2f} cm")
            print(f"{info_prefix}  Worst Distance: {worst_distance:.2f} cm")
        if all_joint_limits:
            limits_rate = sum(all_joint_limits) / len(all_joint_limits)
            print(f"{info_prefix}  Joint Limits Respected: {limits_rate:.2%}")
        print(f"{info_prefix}  Reward Statistics: mean={reward_mean:.2f}, std={reward_std:.2f}, min={reward_min:.2f}, max={reward_max:.2f}")
        print("="*50)
    
    # Add DirectML specific visualizations if needed
    if directml and len(episode_data) > 0:
        try:
            generate_directml_visualizations(episode_data)
        except Exception as e:
            print(f"Error generating DirectML visualizations: {e}")
    
    # Return the metrics
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_episode_length': avg_episode_length,
        'avg_distance': avg_distance,
        'best_distance': best_distance,
        'worst_distance': worst_distance,
        'episode_data': episode_data,
        'model_file': model_file
    }

def generate_directml_visualizations(episode_data):
    """
    Generate visualizations for DirectML model evaluation data
    
    Args:
        episode_data: List of dictionaries containing episode data
        
    Returns:
        Path to the visualization directory
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        import numpy as np
        
        # Create visualization directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = f"./visualizations/directml_{timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate episode-specific visualizations
        for i, ep_data in enumerate(episode_data):
            ep_dir = f"{viz_dir}/episode_{i+1}"
            os.makedirs(ep_dir, exist_ok=True)
            
            # Convert lists to numpy arrays for visualization
            rewards = np.array(ep_data['rewards']) if 'rewards' in ep_data and ep_data['rewards'] else np.array([])
            actions = np.array(ep_data['actions']) if 'actions' in ep_data and ep_data['actions'] else np.array([])
            
            # Extract distances from info dictionaries
            distances = []
            if 'infos' in ep_data and ep_data['infos']:
                for info in ep_data['infos']:
                    if 'distance' in info:
                        distances.append(info['distance'])
            distances = np.array(distances)
            
            # Convert observations
            observations = None
            if 'obs' in ep_data and ep_data['obs']:
                # Convert list of observations to numpy array
                # Skip the last observation which is after the episode ends
                if len(ep_data['obs']) > 1:
                    try:
                        # First try to convert directly
                        observations = np.array(ep_data['obs'][:-1])
                    except:
                        # If that fails, try to handle different shapes
                        obs_list = []
                        for obs in ep_data['obs'][:-1]:
                            # Handle tuple observations
                            if isinstance(obs, tuple) and len(obs) > 0:
                                obs = obs[0]
                            # Convert to numpy if possible
                            if isinstance(obs, np.ndarray):
                                obs_list.append(obs)
                            elif isinstance(obs, list):
                                obs_list.append(np.array(obs))
                        
                        if obs_list:
                            # Take first dimension if observations have multiple dimensions
                            if len(obs_list[0].shape) > 1:
                                observations = np.array([o.flatten() for o in obs_list])
                            else:
                                observations = np.array(obs_list)
            
            # Plot reward over time
            if len(rewards) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(rewards)
                plt.title(f"Episode {i+1} Rewards")
                plt.xlabel("Step")
                plt.ylabel("Reward")
                plt.grid(True)
                plt.savefig(f"{ep_dir}/rewards.png")
                plt.close()
            
            # Plot distance over time
            if len(distances) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(distances)
                plt.title(f"Episode {i+1} Distance to Target")
                plt.xlabel("Step")
                plt.ylabel("Distance (cm)")
                plt.grid(True)
                plt.savefig(f"{ep_dir}/distances.png")
                plt.close()
            
            # Plot actions over time
            if len(actions) > 0 and actions.size > 0:
                plt.figure(figsize=(10, 6))
                if len(actions.shape) > 1:
                    for j in range(actions.shape[1]):
                        plt.plot(actions[:, j], label=f"Action {j+1}")
                    plt.legend()
        else:
                    plt.plot(actions, label="Action")
                plt.title(f"Episode {i+1} Actions")
                plt.xlabel("Step")
                plt.ylabel("Action Value")
                plt.grid(True)
                plt.savefig(f"{ep_dir}/actions.png")
                plt.close()
            
            # Visualization of observations
            if observations is not None and observations.size > 0:
                # Only visualize if observations are not too high-dimensional
                if len(observations.shape) > 1 and observations.shape[1] <= 30:
                    plt.figure(figsize=(12, 8))
                    # Plot first 10 dimensions or all if less than 10
                    for j in range(min(10, observations.shape[1])):
                        plt.plot(observations[:, j], label=f"Obs {j+1}")
                    plt.title(f"Episode {i+1} Observations")
                    plt.xlabel("Step")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{ep_dir}/observations.png")
                    plt.close()
                
                # Create heatmap of observations if not too large
                if len(observations.shape) > 1 and observations.shape[1] <= 100:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(observations.T, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Value')
                    plt.title(f"Episode {i+1} Observation Heatmap")
                    plt.xlabel("Step")
                    plt.ylabel("Observation Dimension")
                    plt.tight_layout()
                    plt.savefig(f"{ep_dir}/observation_heatmap.png")
                    plt.close()
        
        # Generate summary visualizations
        plt.figure(figsize=(10, 6))
        total_rewards = [ep['total_reward'] for ep in episode_data if 'total_reward' in ep]
        if total_rewards:
            plt.bar(range(1, len(total_rewards) + 1), total_rewards)
            plt.title("Total Rewards by Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(axis='y')
            plt.savefig(f"{viz_dir}/total_rewards.png")
            plt.close()
        
        plt.figure(figsize=(10, 6))
        distances = [ep['distance'] for ep in episode_data if 'distance' in ep]
        if distances:
            plt.bar(range(1, len(distances) + 1), distances)
            plt.title("Final Distances by Episode")
            plt.xlabel("Episode")
            plt.ylabel("Final Distance (cm)")
            plt.grid(axis='y')
            plt.savefig(f"{viz_dir}/final_distances.png")
            plt.close()
        
        plt.figure(figsize=(10, 6))
        success_count = sum(1 for ep in episode_data if 'success' in ep and ep['success'])
        if episode_data:
            success_rate = (success_count / len(episode_data)) * 100
            plt.bar(['Success', 'Failure'], [success_rate, 100 - success_rate])
            plt.title("Episode Outcomes")
            plt.ylabel("Percentage")
            plt.grid(axis='y')
            plt.savefig(f"{viz_dir}/success_rate.png")
            plt.close()
        
        # Create a README file
        with open(f"{viz_dir}/README.md", "w") as f:
            f.write("# DirectML Model Evaluation Visualizations\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total episodes: {len(episode_data)}\n")
            if episode_data:
                success_rate = (success_count / len(episode_data)) * 100
                f.write(f"- Success rate: {success_rate:.1f}%\n")
                if total_rewards:
                    f.write(f"- Average reward: {np.mean(total_rewards):.2f}\n")
                if distances:
                    f.write(f"- Average final distance: {np.mean(distances):.2f} cm\n")
            f.write(f"\n## Visualization Guide\n\n")
            f.write("Each episode directory contains:\n\n")
            f.write("- `rewards.png`: Reward at each step\n")
            f.write("- `distances.png`: Distance to target at each step\n")
            f.write("- `actions.png`: Actions taken at each step\n")
            f.write("- `observations.png`: Observation values over time\n")
            f.write("- `observation_heatmap.png`: Heatmap of observation dimensions\n\n")
            f.write("Summary visualizations in the root directory:\n\n")
            f.write("- `total_rewards.png`: Total reward for each episode\n")
            f.write("- `final_distances.png`: Final distance for each episode\n")
            f.write("- `success_rate.png`: Success vs failure rate\n")
        
        print(f"DirectML visualizations generated in: {viz_dir}")
        return viz_dir
        
    except Exception as e:
        import traceback
        print(f"Error generating visualizations: {e}")
        print(traceback.format_exc())
        return None

def run_evaluation_sequence(model_path, viz_speed=0.02, save_video=False):
    """
    Run a predefined evaluation sequence to showcase model capabilities.
    
    Args:
        model_path: Path to the saved model
        viz_speed: Speed of visualization (seconds between steps)
        save_video: Whether to save a video of the evaluation
    """
    print(f"Running evaluation sequence for model: {model_path}")
    
    # Load workspace data
    load_workspace_data(verbose=True)
    
    # Create environment
    env = RobotPositioningRevampedEnv(
        gui=True,
        viz_speed=viz_speed,
        verbose=True
    )
    
    # Wrap in VecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Check for normalization statistics
    vec_normalize_path = model_path.replace("final_model", "vec_normalize_stats")
    if os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=vec_env)
    
    # Setup video recording if requested
    if save_video:
        try:
        import imageio
        frames = []
        
        # Set higher resolution for video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=120,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.3]
        )
        except ImportError:
            print("Warning: imageio not found, video recording will be disabled")
            save_video = False
    
    # Run demo sequence (5 episodes)
    for ep in range(5):
        print(f"\nDemo Episode {ep+1}/5")
        
        # Reset environment with API compatibility handling
        try:
            # Try new gymnasium API (returns obs, info)
            obs, info = vec_env.reset()
            initial_distance = info[0].get('initial_distance', 0.0)
        except ValueError:
            # Fall back to older gym API (returns only obs)
            obs = vec_env.reset()
            # Estimate initial distance as best we can
            state = env.robot._get_state()
            ee_position = state[12:15]
            initial_distance = np.linalg.norm(ee_position - env.target_position)
        
        print(f"Initial distance to target: {initial_distance*100:.2f}cm")
        
        done = False
        step = 0
        
        # Pre-episode pause for better visualization
        if save_video:
            for _ in range(30):  # Capture 30 frames of initial state
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
        else:
            time.sleep(1.0)
        
        # Run episode
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Actions are now inherently limited by the JointLimitedBox space
            # No need for additional limit enforcement
            
            # Step environment with the action
            obs, reward, done, info = vec_env.step(action)
            
            # Capture frame for video or delay for visualization
            if save_video:
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
            else:
                time.sleep(viz_speed)
            
            step += 1
        
        # Post-episode pause
        if save_video:
            for _ in range(30):  # Capture 30 frames of final state
                img = p.getCameraImage(1200, 800, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                frames.append(img[2])
        else:
            time.sleep(1.0)
        
        # Print episode results
        try:
            success = info[0].get('target_reached', False)
            distance = info[0].get('distance', 0.0)
        except (IndexError, TypeError):
            success = False
            distance = 0.0
            
        print(f"Episode {ep+1} - {'SUCCESS' if success else 'FAILURE'}")
        print(f"  Final distance: {distance*100:.2f}cm")
        print(f"  Steps: {step}")
    
    # Save video if requested
    if save_video:
        try:
        from datetime import datetime
        video_path = f"./evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    # Close environment
    vec_env.close()

class JointLimitMonitorCallback(BaseCallback):
    """
    Callback to monitor whether actions respect joint limits during training.
    This is essential for ensuring the learning process respects physical constraints.
    """
    def __init__(self, log_interval=5000, verbose=0):
        super(JointLimitMonitorCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.violations = 0
        self.total_checked = 0
        self.last_log_time = time.time()
        
    def _init_callback(self):
        # Initialize counters
        self.violations = 0
        self.total_checked = 0
        self.last_log_time = time.time()
    
    def _on_step(self):
        """Check if latest actions respect joint limits."""
        if self.n_calls % self.log_interval == 0 and hasattr(self.model, 'last_obs'):
            # Get current actions from the policy
            actions, _ = self.model.predict(self.model.last_obs, deterministic=False)
            
            # Check if actions would result in joint positions within limits
            env_violations = 0
            
            for env_idx, action in enumerate(actions):
                # Get the environment and check if the unnormalized action respects limits
                env = self.training_env.envs[env_idx]
                if hasattr(env, 'action_space') and isinstance(env.action_space, JointLimitedBox):
                    joint_positions = env.action_space.unnormalize_action(action)
                    
                    # Check each joint against its limits
                    for joint_idx, pos in enumerate(joint_positions):
                        if joint_idx in env.action_space.joint_limits:
                            limit_low, limit_high = env.action_space.joint_limits[joint_idx]
                            if pos < limit_low or pos > limit_high:
                                env_violations += 1
                                if self.verbose > 1:  # Only show detailed violations at higher verbosity
                                    print(f"Joint limit violation: Joint {joint_idx} position {pos:.4f} outside limits [{limit_low:.4f}, {limit_high:.4f}]")
            
            self.violations += env_violations
            self.total_checked += len(actions)
            
            # Log at regular intervals
            if self.verbose > 0 and self.total_checked > 0:
                violation_rate = (self.violations / self.total_checked) * 100
                elapsed_time = time.time() - self.last_log_time
                print(f"Joint limit check: {violation_rate:.2f}% violations detected ({self.violations}/{self.total_checked}) in {elapsed_time:.1f}s")
                self.last_log_time = time.time()
        
        return True

class CustomDirectMLModel:
    """
    Advanced implementation of a DirectML model that dynamically adapts to 
    the architecture of the trained model and provides compatible predict method.
    Features:
    - Automatic dimension adaptation
    - Flexible parameter loading
    - Architecture inference from state dict
    """
    
    def __init__(self, observation_space, action_space, device="cpu"):
        """Initialize the model with the right observation and action spaces"""
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.shape[0]
        
        # Initialize networks with placeholder structure
        # (will be refined after loading state dict)
        self.architecture = {}
        self.networks = {}
        self.constants = {}
        
        # We'll initialize the actual networks after analyzing the state dict
        # to ensure the dimensions exactly match
    
    def predict(self, observation, deterministic=True):
        """
        Get action predictions from the model, compatible with SB3 interface.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Action to take
            _: Placeholder for state (None)
        """
        with torch.no_grad():
            # Handle different observation types
            if isinstance(observation, tuple):
                # For tuple observations, take the first element
                print(f"Tuple observation detected with {len(observation)} elements")
                observation = observation[0]
            elif isinstance(observation, dict):
                # For dict observations, extract the main observation
                if 'obs' in observation:
                    observation = observation['obs']
                else:
                    # Try to find the largest tensor or array as the main observation
                    main_obs = None
                    max_size = 0
                    for key, value in observation.items():
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            size = np.prod(value.shape) if hasattr(value, 'shape') else 1
                            if size > max_size:
                                max_size = size
                                main_obs = value
                    if main_obs is not None:
                        observation = main_obs
                    else:
                        raise ValueError(f"Could not determine main observation from dict: {observation.keys()}")
            
            # Convert to tensor if it's a numpy array
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation).to(self.device)
                if observation.dim() == 1:
                    observation = observation.unsqueeze(0)
            
            # Check if model is initialized
            if not hasattr(self, 'initialized') or not self.initialized:
                raise RuntimeError("Model not initialized. Call load() first!")
            
            # Forward pass through feature extractor
            features = self.feature_extractor(observation)
            
            # Forward pass through shared trunk (components applied individually)
            shared_features = features
            for module_name in sorted([k for k in self.networks.keys() if k.startswith('shared_trunk')]):
                module = self.networks[module_name]
                shared_features = module(shared_features)
                # Apply activation after linear/norm layers if needed
                if module_name.endswith('linear') and self.architecture.get('trunk_activation') == 'tanh':
                    shared_features = torch.tanh(shared_features)
                elif module_name.endswith('linear') and self.architecture.get('trunk_activation') == 'relu':
                    shared_features = torch.relu(shared_features)
            
            # Get action mean
            action_mean = self.networks['action_mean'](shared_features)
            
            if deterministic:
                action = action_mean
            else:
                action_log_std = self.networks['action_log_std'](shared_features)
                action_std = torch.exp(action_log_std)
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()
            
            # Convert to numpy
            action_np = action.cpu().numpy().flatten()
            return action_np, None
    
    def get_features(self, observation):
        """
        Extract features from an observation using the model's feature extractor
        
        Args:
            observation: Environment observation
            
        Returns:
            Feature vector
        """
        with torch.no_grad():
            # Convert to tensor if it's a numpy array
            if isinstance(observation, np.ndarray):
                observation = torch.FloatTensor(observation).to(self.device)
                if observation.dim() == 1:
                    observation = observation.unsqueeze(0)
            
            # Extract features
            features = self.feature_extractor(observation)
            return features.cpu().numpy()
    
    def _infer_architecture(self, state_dict):
        """
        Analyze state dict to infer the architecture of the original model
        
        Args:
            state_dict: Model state dict with parameter tensors
        
        Returns:
            Dict with architecture specifications
        """
        import torch.nn as nn
        architecture = {}
        
        # Collect all feature extractor layer dimensions
        fe_dims = []
        fe_weights = sorted([k for k in state_dict.keys() if k.startswith('feature_extractor') and k.endswith('weight')])
        for i, key in enumerate(fe_weights):
            if i == 0:  # Input layer
                try:
                    # Check tensor shape and dimensionality
                    tensor = state_dict[key]
                    if len(tensor.shape) > 1:
                        in_dim = tensor.shape[1]  # Input dimension
                        out_dim = tensor.shape[0]  # Output dimension
                        fe_dims.append((in_dim, out_dim))
                    else:
                        # Skip 1D tensors (like normalization weights)
                        print(f"Skipping 1D tensor at {key} with shape {tensor.shape}")
                except (IndexError, AttributeError) as e:
                    print(f"Warning: Could not extract dimensions from {key} with shape {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'unknown'}")
                    # Use default dimensions for the input layer
                    fe_dims.append((self.input_dim, 128))
            else:
                # Check for existing layers
                if key in state_dict:
                    try:
                        # Check tensor shape and dimensionality
                        tensor = state_dict[key]
                        if len(tensor.shape) > 1:
                            in_dim = tensor.shape[1]  # Input dimension
                            out_dim = tensor.shape[0]  # Output dimension
                            fe_dims.append((in_dim, out_dim))
                        else:
                            # Skip 1D tensors (like normalization weights)
                            print(f"Skipping 1D tensor at {key} with shape {tensor.shape}")
                    except (IndexError, AttributeError) as e:
                        print(f"Warning: Could not extract dimensions from {key} with shape {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'unknown'}")
                        # Use default dimensions
                        if len(fe_dims) > 0:
                            fe_dims.append((fe_dims[-1][1], fe_dims[-1][1]))
                        else:
                            fe_dims.append((self.input_dim, 128))
        
        # If no FE dimensions were extracted, use defaults
        if not fe_dims:
            print("No feature extractor dimensions found. Using defaults.")
            fe_dims = [(self.input_dim, 128), (128, 128), (128, 128)]
            
        # Collect shared trunk dimensions
        trunk_dims = []
        trunk_weights = sorted([k for k in state_dict.keys() if k.startswith('shared_trunk') and k.endswith('weight')])
        for i, key in enumerate(trunk_weights):
            if key in state_dict:
                try:
                    tensor = state_dict[key]
                    if len(tensor.shape) > 1:
                        in_dim = tensor.shape[1]  # Input dimension
                        out_dim = tensor.shape[0]  # Output dimension
                        trunk_dims.append((in_dim, out_dim))
                    else:
                        # For 1D tensors, use the same dim for in and out
                        dim = tensor.shape[0]
                        trunk_dims.append((dim, dim))
                except (IndexError, AttributeError) as e:
                    print(f"Warning: Could not extract dimensions from {key} with shape {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'unknown'}")
                    # Use default dimensions based on feature extractor output
                    if i == 0 and fe_dims:
                        trunk_dims.append((fe_dims[-1][1], 256))
                    elif trunk_dims:
                        trunk_dims.append((trunk_dims[-1][1], 256))
                    else:
                        trunk_dims.append((128, 256))
        
        # If no trunk dimensions were extracted, use defaults
        if not trunk_dims and fe_dims:
            print("No shared trunk dimensions found. Using defaults.")
            trunk_dims = [(fe_dims[-1][1], 256)]
        elif not trunk_dims:
            trunk_dims = [(128, 256)]
            
        # Determine output dimensions from action networks
        action_mean_key = 'action_mean.weight'
        action_output_dim = self.output_dim
        action_input_dim = trunk_dims[-1][1] if trunk_dims else 256
        
        try:
            if action_mean_key in state_dict:
                tensor = state_dict[action_mean_key]
                if len(tensor.shape) > 1:
                    action_output_dim = tensor.shape[0]
                    action_input_dim = tensor.shape[1]
        except (IndexError, AttributeError) as e:
            print(f"Warning: Could not extract dimensions from {action_mean_key}")
            # Keep defaults
        
        # Determine value head dimensions
        value_dim = None
        value_keys = [k for k in state_dict.keys() if k.startswith('value_head') and k.endswith('weight')]
        if value_keys:
            try:
                tensor = state_dict[value_keys[0]]
                if len(tensor.shape) > 1:
                    value_dim = tensor.shape[1]
            except (IndexError, AttributeError) as e:
                print(f"Warning: Could not extract value head dimensions")
                value_dim = 128  # Default
        
        # Check for activation clues in parameter names
        has_tanh = any('tanh' in k.lower() for k in state_dict.keys())
        has_relu = any('relu' in k.lower() for k in state_dict.keys())
        
        # Select likely activation function - default to tanh for DirectML models
        activation = 'tanh'
        if has_relu:
            activation = 'relu'
        elif has_tanh:
            activation = 'tanh'
        
        # Store architecture details
        architecture['input_dim'] = self.input_dim
        architecture['output_dim'] = self.output_dim
        architecture['fe_dims'] = fe_dims
        architecture['trunk_dims'] = trunk_dims
        architecture['action_input_dim'] = action_input_dim
        architecture['action_output_dim'] = action_output_dim
        architecture['value_dim'] = value_dim or 128  # Default if not found
        architecture['activation'] = activation
        architecture['trunk_activation'] = activation
        
        # Identify if model uses layer normalization (typical for DirectML models)
        uses_layernorm = any('layernorm' in k.lower() or '.ln' in k.lower() for k in state_dict.keys())
        architecture['uses_layernorm'] = uses_layernorm
        
        # Analyze state dict keys to identify if it's a custom architecture
        has_feature_extractor = any('feature_extractor' in k for k in state_dict.keys())
        has_shared_trunk = any('shared_trunk' in k for k in state_dict.keys())
        
        architecture['is_custom_architecture'] = has_feature_extractor and has_shared_trunk
        
        return architecture
    
    def _build_networks(self):
        """
        Build the model networks based on the inferred architecture
        """
        import torch.nn as nn
        
        # Create feature extractor
        fe_layers = []
        fe_in_dim = self.architecture['input_dim']
        
        # Get feature extractor dimensions
        fe_dims = self.architecture.get('fe_dims', [(fe_in_dim, 128), (128, 128), (128, 128)])
        
        # Create feature extractor layers
        for i, (in_dim, out_dim) in enumerate(fe_dims):
            # Add linear layer
            fe_layers.append(nn.Linear(in_dim, out_dim))
            
            # Add normalization if the model uses it
            if self.architecture.get('uses_layernorm', True):
                fe_layers.append(nn.LayerNorm(out_dim))
            
            # Add activation except after the last layer
            if i < len(fe_dims) - 1:
                if self.architecture.get('activation') == 'tanh':
                    fe_layers.append(nn.Tanh())
                else:
                    fe_layers.append(nn.ReLU())
        
        # Create the feature extractor as a sequential model
        self.feature_extractor = nn.Sequential(*fe_layers)
        
        # Get trunk dimensions - ensure we have at least the right input/output
        trunk_dims = self.architecture.get('trunk_dims', [(fe_dims[-1][1], 128), (128, 256)])
        if not trunk_dims:
            trunk_dims = [(fe_dims[-1][1], 256)]
        
        # Create shared trunk components as separate modules for more flexibility
        last_dim = fe_dims[-1][1]  # Output dim of feature extractor
        
        # Need to track exact order of modules to apply them correctly in predict()
        for i, (in_dim, out_dim) in enumerate(trunk_dims):
            # Check if this is a linear or normalization layer
            if in_dim != out_dim:  # Likely a linear layer
                # Add linear layer
                self.networks[f'shared_trunk_{i}_linear'] = nn.Linear(in_dim, out_dim)
                last_dim = out_dim
            else:  # Likely a normalization layer
                if self.architecture.get('uses_layernorm', True):
                    self.networks[f'shared_trunk_{i}_norm'] = nn.LayerNorm(in_dim)
        
        # Action networks - input dim is the output of the last trunk layer
        action_input_dim = self.architecture.get('action_input_dim', last_dim)
        action_output_dim = self.architecture.get('action_output_dim', self.output_dim)
        
        self.networks['action_mean'] = nn.Linear(action_input_dim, action_output_dim)
        self.networks['action_log_std'] = nn.Linear(action_input_dim, action_output_dim)
        
        # Value head - if needed
        if self.architecture.get('value_dim'):
            value_layers = []
            value_input_dim = action_input_dim
            value_hidden_dim = self.architecture.get('value_dim', 128)
            
            value_layers.append(nn.Linear(value_input_dim, value_hidden_dim))
            
            # Add activation
            if self.architecture.get('activation') == 'tanh':
                value_layers.append(nn.Tanh())
            else:
                value_layers.append(nn.ReLU())
                
            value_layers.append(nn.Linear(value_hidden_dim, 1))
            
            self.networks['value_head'] = nn.Sequential(*value_layers)
        
        # Add constants
        self.constants['action_scale'] = torch.tensor(1.0)
        self.constants['log_2pi'] = torch.log(torch.tensor(2.0 * np.pi))
        self.constants['sqrt_2'] = torch.sqrt(torch.tensor(2.0))
        self.constants['eps'] = torch.tensor(1e-8)
        
        # Move to device
        self.to(self.device)
        
        # Mark as initialized
        self.initialized = True
    
    def to(self, device):
        """Move the model components to the specified device"""
        # Convert string to device if needed
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        
        # Move feature extractor if it exists
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor = self.feature_extractor.to(device)
        
        # Move all network components
        for name, network in self.networks.items():
            self.networks[name] = network.to(device)
        
        # Move constants
        for name, constant in self.constants.items():
            if isinstance(constant, torch.Tensor):
                self.constants[name] = constant.to(device)
        
        return self
    
    def load(self, path):
        """
        Load and adapt to model parameters from a saved model file
        
        Args:
            path: Path to the model file
        """
        print(f"Loading model from {path}...")
        try:
            # Use 'cpu' string to avoid DirectML issues
            checkpoint = torch.load(path, map_location='cpu')
            
            # Load policy parameters
            if 'policy_state_dict' in checkpoint:
                state_dict = checkpoint['policy_state_dict']
                
                print("Found policy state dict with keys:", list(state_dict.keys()))
                
                # Infer the architecture from the state dict
                self.architecture = self._infer_architecture(state_dict)
                print(f"Inferred architecture: {self.architecture}")
                
                # Now build the networks based on the inferred architecture
                self._build_networks()
                
                # Load constants first if they exist
                for name in ['action_scale', 'log_2pi', 'sqrt_2', 'eps']:
                    if name in state_dict and isinstance(state_dict[name], torch.Tensor):
                        self.constants[name] = state_dict[name].to(self.device)
                
                # Create a mapping from state dict keys to model parameters
                param_mapping = self._create_parameter_mapping(state_dict)
                
                # Load parameters based on the mapping
                loaded_params = 0
                mismatched_params = 0
                
                for state_dict_key, model_param in param_mapping.items():
                    if state_dict_key in state_dict:
                        # Get the parameter from state dict
                        dict_param = state_dict[state_dict_key]
                        
                        # Check if shapes match
                        if dict_param.shape == model_param.shape:
                            model_param.data.copy_(dict_param)
                            loaded_params += 1
                        else:
                            print(f"Shape mismatch for {state_dict_key}: expected {model_param.shape}, got {dict_param.shape}")
                            mismatched_params += 1
                    else:
                        # This is less concerning - some keys might not be present in simpler models
                        if self.architecture.get('is_custom_architecture', True):
                            print(f"Parameter {state_dict_key} not found in state dict")
                
                print(f"Successfully loaded {loaded_params} parameters, with {mismatched_params} mismatches")
                if loaded_params > 0:
                    print("Model loaded successfully!")
                    return True
                else:
                    print("Warning: No parameters were loaded. Model may not function correctly.")
                    return False
                
            else:
                print("ERROR: No policy state dict found in checkpoint")
                return False
                
        except Exception as e:
            import traceback
            print(f"Error loading model parameters: {e}")
            with open("error_log.txt", "w") as f:
                f.write(f"Exception: {str(e)}\n")
                f.write(traceback.format_exc())
            raise e
    
    def _create_parameter_mapping(self, state_dict):
        """
        Create a mapping from state dict keys to model parameters
        
        Args:
            state_dict: Dictionary of parameter tensors
        
        Returns:
            Dictionary mapping state dict keys to model parameters
        """
        param_mapping = {}
        
        # Map feature extractor parameters
        fe_layer_idx = 0
        for i in range(len(self.feature_extractor)):
            if hasattr(self.feature_extractor[i], 'weight'):
                # Try different naming patterns
                possible_keys = [
                    f'feature_extractor.{i}.weight',
                    f'feature_extractor.{fe_layer_idx}.weight',
                    f'features.{i}.weight',
                    f'features.{fe_layer_idx}.weight'
                ]
                
                # Find first matching key
                key = next((k for k in possible_keys if k in state_dict), None)
                
                if key:
                    param_mapping[key] = self.feature_extractor[i].weight
                    
                    # Also map the bias if it exists
                    bias_key = key.replace('weight', 'bias')
                    if bias_key in state_dict and hasattr(self.feature_extractor[i], 'bias'):
                        param_mapping[bias_key] = self.feature_extractor[i].bias
                
                fe_layer_idx += 1
        
        # Map shared trunk parameters
        for name, module in self.networks.items():
            if name.startswith('shared_trunk'):
                # Extract layer index from name
                parts = name.split('_')
                idx = int(parts[2]) if len(parts) > 2 else 0
                
                # Different naming patterns for trunk layers
                if 'linear' in name and hasattr(module, 'weight'):
                    # Try different naming patterns
                    possible_keys = [
                        f'shared_trunk.{idx*2}.weight',
                        f'shared_trunk.{idx}.weight',
                        f'trunk.{idx*2}.weight',
                        f'trunk.{idx}.weight'
                    ]
                    
                    # Find first matching key
                    key = next((k for k in possible_keys if k in state_dict), None)
                    
                    if key:
                        param_mapping[key] = module.weight
                        
                        # Also map the bias if it exists
                        bias_key = key.replace('weight', 'bias')
                        if bias_key in state_dict and hasattr(module, 'bias'):
                            param_mapping[bias_key] = module.bias
                
                elif 'norm' in name and hasattr(module, 'weight'):
                    # Try different naming patterns for normalization layers
                    possible_keys = [
                        f'shared_trunk.{idx*2+1}.weight',
                        f'shared_trunk.{idx+1}.weight',
                        f'trunk.{idx*2+1}.weight',
                        f'trunk.{idx+1}.weight'
                    ]
                    
                    # Find first matching key
                    key = next((k for k in possible_keys if k in state_dict), None)
                    
                    if key:
                        param_mapping[key] = module.weight
                        
                        # Also map the bias if it exists
                        bias_key = key.replace('weight', 'bias')
                        if bias_key in state_dict and hasattr(module, 'bias'):
                            param_mapping[bias_key] = module.bias
        
        # Map action output networks
        for name in ['action_mean', 'action_log_std']:
            if name in self.networks:
                # Try different naming patterns
                possible_keys = [
                    f'{name}.weight',
                    f'{name.replace("action_", "")}.weight',
                    f'pi_out.weight' if name == 'action_mean' else 'log_std.weight'
                ]
                
                # Find first matching key
                key = next((k for k in possible_keys if k in state_dict), None)
                
                if key:
                    param_mapping[key] = self.networks[name].weight
                    
                    # Also map the bias if it exists
                    bias_key = key.replace('weight', 'bias')
                    if bias_key in state_dict:
                        param_mapping[bias_key] = self.networks[name].bias
        
        # Map value head parameters
        if 'value_head' in self.networks:
            value_head = self.networks['value_head']
            
            # Map each layer of the value head
            for i in range(len(value_head)):
                if hasattr(value_head[i], 'weight'):
                    # Try different naming patterns
                    possible_keys = [
                        f'value_head.{i}.weight',
                        f'value_head.{i*2}.weight',
                        f'vf.{i}.weight',
                        f'vf.{i*2}.weight'
                    ]
                    
                    # Find first matching key
                    key = next((k for k in possible_keys if k in state_dict), None)
                    
                    if key:
                        param_mapping[key] = value_head[i].weight
                        
                        # Also map the bias if it exists
                        bias_key = key.replace('weight', 'bias')
                        if bias_key in state_dict and hasattr(value_head[i], 'bias'):
                            param_mapping[bias_key] = value_head[i].bias
        
        return param_mapping

def evaluate_model_wrapper(model_path, num_episodes=10, visualize=True, verbose=True):
    """
    Wrapper function that bridges the different evaluate_model interfaces
    
    Args:
        model_path: Path to model file to load
        num_episodes: Number of episodes to evaluate
        visualize: Whether to visualize the evaluation
        verbose: Whether to print progress
    """
    # Create environment for evaluation
    env = create_revamped_envs(
        num_envs=1,
        viz_speed=0.02 if visualize else 0.0,
        parallel_viz=False,
        training_mode=False
    )[0]  # Get the unwrapped environment
    
    # Check if this is a DirectML model based on path
    directml = "directml" in model_path.lower()
    
    # Load the model
    policy = None
    
    if directml:
        try:
            print(f"Loading DirectML model from {model_path}...")
            # Create a custom model for DirectML
            directml_model = CustomDirectMLModel(env.observation_space, env.action_space, device='cpu')
            
            # Load the model parameters
            if directml_model.load(model_path):
                # Successfully loaded DirectML model
                policy = directml_model
                print("DirectML model loaded successfully!")
            else:
                raise ValueError("Failed to load DirectML model")
        except Exception as e:
            import traceback
            error_msg = f"Error loading DirectML model: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open("error_log.txt", "w") as f:
                f.write(error_msg)
            return None
    else:
        try:
            # Load a standard model
            print(f"Loading standard model from {model_path}...")
            from stable_baselines3 import PPO
            policy = PPO.load(model_path)
            print("Standard model loaded successfully!")
        except Exception as e:
            import traceback
            error_msg = f"Error loading standard model: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open("error_log.txt", "w") as f:
                f.write(error_msg)
            return None
    
    # Set up evaluation parameters
    rollout_steps = 150  # Standard episode length
    
    # Call the actual evaluation function
    return evaluate_model(
        env=env,
        policy=policy,
        rollout_steps=rollout_steps,
        num_episodes=num_episodes,
        render=visualize,
        verbose=verbose,
        model_file=model_path,
        directml=directml,
        info_prefix="[Eval] "
    )

if __name__ == "__main__":
    main() 
