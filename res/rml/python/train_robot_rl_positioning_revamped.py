#!/usr/bin/env python3
# train_robot_rl_positioning_revamped.py
# A completely revamped approach to training a robot for accurate end-effector positioning

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
from train_robot_rl_positioning import (
    get_shared_pybullet_client, 
    FANUCRobotEnv, 
    load_workspace_data,
    determine_reachable_workspace,
    adjust_camera_for_robots
)

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
    Environment wrapper that strictly enforces joint limits from the URDF model.
    """
    def __init__(self, env):
        super().__init__(env)
        self.robot = env.robot
        print("Joint limits will be strictly enforced according to URDF specifications")
    
    def step(self, action):
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
                 offset_x=0.0):
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
        
        # Define action space: Delta joint positions (continuous)
        # Each action is a delta to the current joint position, limited to a reasonable range
        max_delta = 0.1  # 5.7 degrees maximum change per step for more precise control
        self.action_space = spaces.Box(
            low=-max_delta, 
            high=max_delta, 
            shape=(self.dof,),
            dtype=np.float32
        )
        
        # Define observation space: 
        # - Current normalized joint angles (6)
        # - Current end effector position (3)
        # - Target position (3)
        # - Distance to target (1)
        # - Normalized direction to target (3)
        # - Previous action (6)
        # - Joint position history (6 * history_length)
        # - End effector position history (3 * history_length)
        # Total: 6 + 3 + 3 + 1 + 3 + 6 + (6+3)*5 = 73
        
        # Maximum values for each component
        max_joints = np.ones(self.dof)  # Normalized joint angles (0-1)
        max_position = np.ones(3) * (workspace_size * 2.0)  # Position (meters)
        max_target = np.ones(3) * (workspace_size * 2.0)  # Target position (meters)
        max_distance = np.array([workspace_size * 2.0])  # Distance (meters)
        max_direction = np.ones(3)  # Normalized direction (-1 to 1)
        max_action = np.ones(self.dof) * max_delta  # Previous action
        
        # Combine all max values
        max_obs = np.concatenate([
            max_joints,  # 6
            max_position,  # 3
            max_target,  # 3
            max_distance,  # 1
            max_direction,  # 3
            max_action,  # 6
            np.tile(np.concatenate([max_joints, max_position]), self.observation_history_length)  # (6+3)*history_length
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
        Apply action to the robot using direct joint position control
        
        Args:
            action: Delta joint positions (in radians)
        """
        # Get current joint positions
        state = self.robot._get_state()
        current_joint_positions = state[:self.robot.dof*2:2]
        
        # Calculate new joint positions by adding the delta
        new_joint_positions = []
        
        for i, delta in enumerate(action):
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                # Current position
                current_pos = current_joint_positions[i]
                
                # Calculate new position with delta
                new_pos = current_pos + delta
                
                # Enforce joint limits
                new_pos = max(limit_low, min(limit_high, new_pos))
                
                new_joint_positions.append(new_pos)
            else:
                # For joints without limits, just add the delta
                new_joint_positions.append(current_joint_positions[i] + delta)
        
        # Apply the new joint positions
        # Fix by passing both positions and velocities (zero velocities)
        zero_velocities = [0.0] * len(new_joint_positions)
        next_state = self.robot.step((new_joint_positions, zero_velocities))
        
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
            "joint_positions": current_joint_positions,
        }
        
        # Determine if target is reached
        target_reached = distance <= self.accuracy_threshold
        info["target_reached"] = target_reached
        
        # Calculate reward
        reward = self._calculate_reward(distance, current_ee_pos, action)
        
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
    
    def _calculate_reward(self, distance, current_ee_pos, action):
        """
        Calculate reward with sophisticated reward engineering
        
        The reward has these components:
        1. Distance component: reward for reducing distance to target
        2. Progress component: reward based on improvement from previous step
        3. Movement efficiency: penalty for excessive movement
        4. Smooth motion: penalty for jerky movements
        5. Joint limit avoidance: penalty for approaching joint limits
        """
        # 1. Distance component - negative exponential of distance
        # This gives higher gradients close to the target (encouraging precision)
        distance_factor = 0.5  # Controls how quickly the reward falls off with distance
        distance_reward = np.exp(-distance_factor * distance) - 0.5
        
        # 2. Progress component - reward for getting closer to the target
        if self.previous_distance is not None:
            # Raw progress in meters (positive = getting closer)
            progress = self.previous_distance - distance
            
            # Scale by previous distance to normalize (percent improvement)
            if self.previous_distance > 1e-6:
                relative_progress = progress / self.previous_distance
                progress_reward = relative_progress * 2.0
            else:
                progress_reward = 0.0
                
            # Cap the progress reward to avoid large spikes
            progress_reward = np.clip(progress_reward, -0.5, 0.5)
        else:
            progress_reward = 0.0
        
        # 3. Movement efficiency - small penalty for large actions
        # Encourages using minimum necessary movement
        efficiency_penalty = -0.02 * np.sum(np.square(action))
        
        # 4. Smooth motion - penalty for jerky movements
        # Compare this action to the previous action to discourage rapid changes
        if np.any(self.previous_action):  # Check if previous action is not all zeros
            jerk = np.sum(np.square(action - self.previous_action))
            smoothness_penalty = -0.02 * jerk
        else:
            smoothness_penalty = 0.0
        
        # 5. Joint limit avoidance - penalty for getting close to joint limits
        # Get current joint states
        state = self.robot._get_state()
        current_joint_positions = state[:self.robot.dof*2:2]
        
        joint_limit_penalty = 0.0
        for i, pos in enumerate(current_joint_positions):
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                
                # Calculate proximity to limits (0 = at limit, 1 = at middle)
                range_size = limit_high - limit_low
                mid_point = (limit_high + limit_low) / 2.0
                normalized_pos = 2.0 * abs(pos - mid_point) / range_size
                
                # Apply penalty that increases as we get closer to limits
                # No penalty in the middle 50% of the range, then increasing
                if normalized_pos > 0.5:
                    # Map 0.5-1.0 to 0.0-1.0 for penalty calculation
                    limit_proximity = (normalized_pos - 0.5) * 2.0
                    joint_limit_penalty -= 0.1 * (limit_proximity ** 2)
        
        # Combine all reward components
        reward = (
            0.5 * distance_reward +      # Base distance reward
            0.3 * progress_reward +      # Progress reward
            0.1 * efficiency_penalty +   # Efficiency penalty (small influence)
            0.05 * smoothness_penalty +  # Smoothness penalty (smaller influence)
            0.05 * joint_limit_penalty   # Joint limit penalty (smaller influence)
        )
        
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
    Custom feature extractor that processes observation components separately
    and combines them in a structured way for better learning.
    """
    def __init__(self, observation_space: spaces.Box):
        # Extract features to a feature dimension that will then be fed to the policy/value networks
        super().__init__(observation_space, features_dim=512)
        
        # Determine observation dimensions
        obs_dim = observation_space.shape[0]
        
        # Define the encoder for different parts of the observation
        # This structured approach helps the network better understand spatial relationships
        
        # Joint positions and previous action encoders (12 values: 6 joints + 6 previous actions)
        self.joint_encoder = nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Position encoder (9 values: 3 ee position + 3 target position + 3 direction)
        self.position_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Distance encoder (1 value)
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # History encoder (processes the time series information)
        # This captures the motion dynamics over time
        history_dim = 9 * 5  # 9 values per timestep (6 joints + 3 ee pos) * 5 timesteps
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Final combined encoder
        self.final_encoder = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 128, 256),  # Combine all encoders
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),  # Output features
            nn.LayerNorm(512),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process the observations using our structured approach
        """
        batch_size = observations.shape[0]
        
        # Extract different components from observation
        # Note: these indices must match the observation creation in the environment
        joint_pos = observations[:, :6]  # First 6 are joint positions
        ee_pos = observations[:, 6:9]  # Next 3 are ee position
        target_pos = observations[:, 9:12]  # Next 3 are target position
        distance = observations[:, 12:13]  # Next 1 is distance
        direction = observations[:, 13:16]  # Next 3 are direction
        prev_action = observations[:, 16:22]  # Next 6 are previous action
        history = observations[:, 22:]  # Rest is history
        
        # Process different components
        joint_features = self.joint_encoder(torch.cat([joint_pos, prev_action], dim=1))
        position_features = self.position_encoder(torch.cat([ee_pos, target_pos, direction], dim=1))
        distance_features = self.distance_encoder(distance)
        history_features = self.history_encoder(history)
        
        # Combine features
        combined_features = torch.cat([
            joint_features, 
            position_features, 
            distance_features, 
            history_features
        ], dim=1)
        
        # Final encoding
        features = self.final_encoder(combined_features)
        
        return features

class CustomActorNetwork(nn.Module):
    """
    Custom actor network with improved architecture for precise robot control.
    Uses residual connections and deeper layers for better gradient flow.
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
        # Starting with a low log std for more precise initial actions
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
    
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
        
        # Return mean and log std
        log_std = self.log_std.expand_as(mean)
        
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
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
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
    def __init__(self, log_interval=100, verbose=1, plot_interval=10000):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.plot_dir = "./plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
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
            save_path = f"./models/checkpoint_{self.num_timesteps}"
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
    Custom PPO implementation with our improved policy architecture.
    """
    def __init__(self, policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 clip_range_vf=None, normalize_advantage=True, ent_coef=0.0,
                 vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1,
                 target_kl=None, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=None, device='auto',
                 _init_setup_model=True):
        
        # Initialize standard PPO
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
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

# Function to create environments for training
def create_revamped_envs(num_envs=1, viz_speed=0.0, parallel_viz=False, strict_limits=False):
    """
    Create multiple instances of the revamped robot positioning environment.
    
    Args:
        num_envs: Number of environments to create
        viz_speed: Speed of visualization (seconds between steps)
        parallel_viz: Whether to use parallel visualization mode
        strict_limits: Whether to strictly enforce joint limits
    
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
                offset_x=offset_x
            )
            
            # Wrap in JointLimitEnforcingEnv if requested
            if strict_limits:
                env = JointLimitEnforcingEnv(env)
                
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
                rank=i
            )
            
            # Wrap in JointLimitEnforcingEnv if requested
            if strict_limits:
                env = JointLimitEnforcingEnv(env)
                
            envs.append(env)
    
    return envs

def train_revamped_robot(args):
    """
    Train a robot with the revamped approach.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create environments
    if args.parallel_viz and args.gui:
        print(f"Using {args.parallel} robots in the same environment with visualization")
        envs = create_revamped_envs(
            num_envs=args.parallel,
            viz_speed=args.viz_speed,
            parallel_viz=True,
            strict_limits=getattr(args, 'strict_limits', False)
        )
    else:
        envs = create_revamped_envs(
            num_envs=args.parallel,
            viz_speed=args.viz_speed if args.gui else 0.0,
            parallel_viz=False,
            strict_limits=getattr(args, 'strict_limits', False)
        )
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    
    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    )
    
    # Define policy kwargs
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        "activation_fn": nn.ReLU
    }
    
    # Create or load model
    if args.load:
        print(f"Loading model from {args.load}")
        model = PPO.load(args.load, env=vec_env)
    else:
        print("Creating new PPO model with custom architecture")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=512,  # Shorter rollout buffer for faster updates
            batch_size=64,
            n_epochs=10,  # More epochs per update for better learning
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            normalize_advantage=True,
            ent_coef=0.01,  # Small entropy coefficient for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        )
    
    # Evaluation mode
    if args.eval_only:
        from stable_baselines3.common.evaluation import evaluate_policy
        
        print(f"Evaluating model for {args.eval_episodes} episodes")
        mean_reward, std_reward = evaluate_policy(
            model,
            vec_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        vec_env.close()
        return
    
    # Prepare directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./models/revamped_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create callbacks
    save_callback = SaveModelCallback(
        save_freq=50000,  # Save every 50k steps
        save_path=model_dir,
        verbose=1
    )
    
    monitor_callback = TrainingMonitorCallback(
        log_interval=100,  # Log every 100 steps
        verbose=1,
        plot_interval=10000  # Plot every 10k steps
    )
    
    # Print training information
    print("\n" + "="*80)
    print(f"Starting training with revamped approach")
    print(f"Using {args.parallel} parallel environments")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Training for {args.steps} timesteps")
    print(f"Models will be saved to {model_dir}")
    print("="*80 + "\n")
    
    # Start training
    model.learn(
        total_timesteps=args.steps,
        callback=[save_callback, monitor_callback]
    )
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save final VecNormalize statistics
    vec_normalize_path = f"{model_dir}/vec_normalize_stats"
    vec_env.save(vec_normalize_path)
    print(f"Normalization statistics saved to {vec_normalize_path}")
    
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
    parser.add_argument('--strict-limits', action='store_true', help='Strictly enforce joint limits from URDF model')
    
    args = parser.parse_args()
    
    # Handle gui/no-gui conflict
    if args.no_gui:
        args.gui = False
    
    return args

def main():
    """Main function."""
    try:
        # Save sys.argv and reset it temporarily to prevent conflicts between demo and main
        original_argv = sys.argv
        
        # Parse arguments
        args = parse_args()
        
        # Load workspace data
        load_workspace_data(verbose=args.verbose)
        
        # Check for evaluation mode
        if args.eval_only:
            if args.load is None:
                print("Error: Must provide a model path with --load for evaluation")
                return
            
            # Run evaluation
            evaluate_model(
                model_path=args.load,
                num_episodes=args.eval_episodes,
                visualize=args.gui,
                verbose=args.verbose,
                strict_limits=args.strict_limits
            )
            return
        
        # Check for demo mode
        if args.demo:
            if args.load is None:
                print("Error: Must provide a model path with --load for demonstration")
                return
            
            # Run demonstration sequence
            run_evaluation_sequence(
                model_path=args.load,
                viz_speed=args.viz_speed if args.viz_speed > 0 else 0.02,
                save_video=args.save_video,
                strict_limits=args.strict_limits
            )
            return
        
        # Train with revamped approach
        train_revamped_robot(args)
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(f"Exception: {str(e)}\n")
            f.write(traceback.format_exc())
        print(f"Error occurred. Check error_log.txt for details.")

def evaluate_model(model_path, num_episodes=20, visualize=True, verbose=True, strict_limits=False):
    """
    Evaluate a trained model on the robot positioning task.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to evaluate
        visualize: Whether to visualize the evaluation
        verbose: Whether to print verbose output
        strict_limits: Whether to strictly enforce joint limits
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"\nEvaluating model: {model_path}")
    
    # Load workspace data if not already loaded
    if _WORKSPACE_POSITIONS is None:
        load_workspace_data(verbose=verbose)
    
    # Create environment for evaluation
    env = RobotPositioningRevampedEnv(
        gui=visualize,
        viz_speed=0.01 if visualize else 0.0,
        verbose=verbose
    )
    
    # Wrap with joint limit enforcer if requested
    if strict_limits:
        env = JointLimitEnforcingEnv(env)
    
    # Wrap in VecEnv as required by Stable-Baselines3
    vec_env = DummyVecEnv([lambda: env])
    
    # Check if VecNormalize statistics are available
    vec_normalize_path = model_path.replace("final_model", "vec_normalize_stats")
    if os.path.exists(vec_normalize_path):
        # Load with normalization
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Don't update normalization statistics during evaluation
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        print("Loaded normalization statistics")
    else:
        print("No normalization statistics found, evaluating without normalization")
    
    # Load the model
    model = PPO.load(model_path, env=vec_env)
    print(f"Model loaded from {model_path}")
    
    # Initialize metrics
    metrics = {
        "success_rate": 0.0,
        "avg_distance": 0.0,
        "avg_steps": 0.0,
        "avg_reward": 0.0,
        "min_distance": float('inf'),
        "max_distance": 0.0,
        "success_episodes": [],
        "failure_episodes": [],
    }
    
    # Run evaluation episodes
    total_reward = 0.0
    total_steps = 0
    total_distance = 0.0
    successes = 0
    
    for ep in range(num_episodes):
        if verbose:
            print(f"\nEvaluation Episode {ep+1}/{num_episodes}")
        
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
        
        if verbose:
            print(f"Initial distance to target: {initial_distance*100:.2f}cm")
        
        # Initialize episode variables
        done = False
        ep_reward = 0.0
        ep_steps = 0
        best_distance = initial_distance
        
        # Run episode
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Enforce joint limits before stepping if strict limits enabled
            if strict_limits:
                # Get current joint positions
                state = env.robot._get_state()
                current_joint_positions = state[:env.robot.dof*2:2]
                
                # Calculate new joint positions from action deltas
                new_joint_positions = []
                for i, delta in enumerate(action[0]):  # action is wrapped in array due to vectorized env
                    new_pos = current_joint_positions[i] + delta
                    new_joint_positions.append(new_pos)
                
                # Enforce limits
                limited_positions = ensure_joint_limits(env.robot, new_joint_positions)
                
                # If action was limited, create a new action array with deltas that respect limits
                if not np.array_equal(new_joint_positions, limited_positions):
                    limited_deltas = []
                    for i, (pos, limited_pos) in enumerate(zip(new_joint_positions, limited_positions)):
                        if pos != limited_pos:
                            # Calculate the allowed delta that respects the limit
                            allowed_delta = limited_pos - current_joint_positions[i]
                            limited_deltas.append(allowed_delta)
                        else:
                            limited_deltas.append(action[0][i])
                    
                    # Replace action with limited version
                    action = np.array([limited_deltas])
            
            # Step environment with API compatibility handling
            try:
                # Try new gymnasium API
                obs, reward, terminated, truncated, info = vec_env.step(action)
                done = terminated[0] or truncated[0]
            except ValueError:
                # Fall back to older gym API
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
            
            # Update metrics
            ep_reward += reward[0]
            ep_steps += 1
            
            try:
                current_distance = info[0].get('distance', 0.0)
            except (IndexError, TypeError):
                # Fallback if info is not structured as expected
                current_distance = best_distance
            
            # Update best distance
            best_distance = min(best_distance, current_distance)
            
            # Delay for visualization
            if visualize:
                time.sleep(0.01)
        
        # Calculate success
        try:
            success = info[0].get('target_reached', False)
        except (IndexError, TypeError):
            success = False
        
        # Update metrics
        total_reward += ep_reward
        total_steps += ep_steps
        total_distance += best_distance
        
        if success:
            successes += 1
            metrics["success_episodes"].append(ep)
        else:
            metrics["failure_episodes"].append(ep)
        
        # Update min/max distances
        metrics["min_distance"] = min(metrics["min_distance"], best_distance)
        metrics["max_distance"] = max(metrics["max_distance"], best_distance)
        
        # Print episode results
        if verbose:
            print(f"Episode {ep+1} - {'SUCCESS' if success else 'FAILURE'}")
            print(f"  Best distance: {best_distance*100:.2f}cm")
            print(f"  Steps: {ep_steps}")
            print(f"  Reward: {ep_reward:.2f}")
    
    # Calculate final metrics
    metrics["success_rate"] = successes / num_episodes
    metrics["avg_distance"] = total_distance / num_episodes
    metrics["avg_steps"] = total_steps / num_episodes
    metrics["avg_reward"] = total_reward / num_episodes
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average Distance: {metrics['avg_distance']*100:.2f}cm")
    print(f"Average Steps: {metrics['avg_steps']:.1f}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Best Distance: {metrics['min_distance']*100:.2f}cm")
    
    # Close environment
    vec_env.close()
    
    return metrics

def run_evaluation_sequence(model_path, viz_speed=0.02, save_video=False, strict_limits=False):
    """
    Run a sequence of evaluation episodes for demonstration purposes
    with visualization and optional video saving.
    
    Args:
        model_path: Path to the trained model
        viz_speed: Visualization speed (delay in seconds)
        save_video: Whether to save a video of the evaluation
        strict_limits: Whether to strictly enforce joint limits
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
    
    # Wrap with joint limit enforcer if requested
    if strict_limits:
        env = JointLimitEnforcingEnv(env)
    
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
            
            # Enforce joint limits before stepping if strict limits enabled
            if strict_limits:
                # Get current joint positions
                state = env.robot._get_state()
                current_joint_positions = state[:env.robot.dof*2:2]
                
                # Calculate new joint positions from action deltas
                new_joint_positions = []
                for i, delta in enumerate(action[0]):  # action is wrapped in array due to vectorized env
                    new_pos = current_joint_positions[i] + delta
                    new_joint_positions.append(new_pos)
                
                # Enforce limits
                limited_positions = ensure_joint_limits(env.robot, new_joint_positions)
                
                # If action was limited, create a new action array with deltas that respect limits
                if not np.array_equal(new_joint_positions, limited_positions):
                    limited_deltas = []
                    for i, (pos, limited_pos) in enumerate(zip(new_joint_positions, limited_positions)):
                        if pos != limited_pos:
                            # Calculate the allowed delta that respects the limit
                            allowed_delta = limited_pos - current_joint_positions[i]
                            limited_deltas.append(allowed_delta)
                        else:
                            limited_deltas.append(action[0][i])
                    
                    # Replace action with limited version
                    action = np.array([limited_deltas])
            
            # Step environment with API compatibility handling
            try:
                # Try new gymnasium API
                obs, reward, terminated, truncated, info = vec_env.step(action)
                done = terminated[0] or truncated[0]
            except ValueError:
                # Fall back to older gym API
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
            
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
        from datetime import datetime
        video_path = f"./evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to {video_path}")
    
    # Close environment
    vec_env.close()

if __name__ == "__main__":
    main() 