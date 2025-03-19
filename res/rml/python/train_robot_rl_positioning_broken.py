#!/usr/bin/env python3
# train_robot_rl_positioning.py
# A new approach for real-time end effector positioning using reinforcement learning
# Designed for real-time end effector positioning using reinforcement learning

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import multiprocessing
import sys
import traceback
import json
import math
import random  # Add this import for random.choice()
from typing import Dict, List, Optional, Any, Tuple, Union
# Ignore the linter error for pybullet - it's installed but the linter can't find it
import pybullet as p  # type: ignore
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from datetime import datetime
import psutil  # type: ignore
import warnings
import logging
import pybullet_data  # Add this import
# We're defining DomainRandomizedEnv in this file, so we don't need to import it
# from domain_randomisation import DomainRandomizedEnv

# Import utility functions from pybullet_utils
from pybullet_utils import (
    configure_visualization,
    visualize_target,
    visualize_ee_position,
    visualize_target_line
)

# Ensure output directories exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./plots", exist_ok=True)  # Add directory for plots

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a robot arm for precise end effector positioning')
parser.add_argument('--steps', type=int, default=1000000, help='Total number of training steps')
parser.add_argument('--target-accuracy', type=float, default=1.0, help='Target accuracy in cm')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
parser.add_argument('--load', type=str, default='', help='Load a pre-trained model to continue training')
parser.add_argument('--eval-only', action='store_true', help='Only run evaluation on a pre-trained model')
parser.add_argument('--gui-delay', type=float, default=0.01, help='Delay between steps for better visualization (seconds)')
parser.add_argument('--parallel', type=int, default=2, help='Number of parallel environments (default: 2)')
parser.add_argument('--parallel-viz', action='store_true', default=True, help='Enable parallel visualization (multiple robots in same view), default: True')
parser.add_argument('--workspace-size', type=float, default=0.7, help='Size of the workspace for target positions')
parser.add_argument('--algorithm', type=str, default='sac', choices=['sac'], help='RL algorithm to use (default: sac)')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size for updates (default: 256, balanced for learning)')
parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size for SAC (default: 1M, larger for better stability)')
parser.add_argument('--train-freq', type=int, default=1, help='Update frequency for SAC (default: 1, update every step)')
parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps per update for SAC (default: 1, balanced learning)')
parser.add_argument('--eval-freq', type=int, default=2000, help='Evaluation frequency in steps')
parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
parser.add_argument('--save-freq', type=int, default=5000, help='Model saving frequency in steps')
parser.add_argument('--clean-viz', action='store_true', help='Enable clean visualization with zoomed-in view of the robot')
parser.add_argument('--eval-viz', action='store_true', help='Enable visualization for evaluation only')
parser.add_argument('--high-lr', action='store_true', help='Use a higher learning rate (1e-3) for faster learning')
parser.add_argument('--viz-speed', type=float, default=0.1, help='Control visualization speed (delay in seconds, higher = slower, default: 0.1)')
parser.add_argument('--optimize-training', action='store_true', help='Enable optimized training settings for faster learning', default=True)
parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
parser.add_argument('--exploration', type=float, default=0.2, help='Initial exploration rate (higher values = more exploration)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--disable-bounds-collision', action='store_true', help='Disable workspace boundary collisions')
args = parser.parse_args()

# Set up device based on args
device = torch.device("cpu")  # Default to CPU, the actual device will be set in main()
print("Using CPU for training (default setting, can be overridden with command line arguments)")

# Apply high learning rate if requested
if args.high_lr:
    args.learning_rate = 1e-3
    print(f"Using high learning rate mode: {args.learning_rate}")
else:
    print(f"Using learning rate: {args.learning_rate}")

# Set GUI delay based on viz-speed parameter
if args.viz_speed > 0:
    args.gui_delay = args.viz_speed
    print(f"Using visualization speed: {args.viz_speed}s delay (slow motion)")
else:
    args.gui_delay = 0.0

# Optimize CPU performance
cpu_count = multiprocessing.cpu_count()
torch.set_num_threads(cpu_count)
print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")

# Set environment variables for better CPU performance
os.environ['MKL_NUM_THREADS'] = str(cpu_count)
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

# Enable faster math operations if available
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Check available memory
try:
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024 ** 3)
    available_gb = memory.available / (1024 ** 3)
    print(f"System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")

    # Warn if memory is low
    if available_gb < 4:
        warnings.warn(f"Low memory detected ({available_gb:.1f} GB available). Training may be slow or crash.")
except:
    pass

# Reduce parallel environments to avoid overloading CPU
if args.parallel <= 0:
    n_parallel_envs = max(1, cpu_count - 2)  # Leave only 2 cores for system processes
    print(f"Maximizing parallelization: using {n_parallel_envs} parallel environments (out of {cpu_count} CPU cores)")

# Insert missing definitions for training parameters
debug_mode = args.debug
target_accuracy_cm = args.target_accuracy
target_accuracy_m = target_accuracy_cm / 100.0  # Convert to meters
gui_delay = args.gui_delay  # Delay for better visualization

# Create timestamp for unique run identification
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.algorithm}_{timestamp}"
log_dir = f"./logs/{run_name}"
model_dir = f"./models/{run_name}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Add a global variable to track the shared PyBullet client
_SHARED_PYBULLET_CLIENT: int | None = None

# Function to get a shared PyBullet client
def get_shared_pybullet_client(render=True):
    """
    Get a shared PyBullet client to ensure all environments use the same visualization window.
    
    Args:
        render: Whether to use GUI mode (True) or DIRECT mode (False)
        
    Returns:
        PyBullet client ID
    """
    global _SHARED_PYBULLET_CLIENT
    
    # If we already have a client, return it
    if _SHARED_PYBULLET_CLIENT is not None:
        return _SHARED_PYBULLET_CLIENT
    
    # Create a new client
    if render:
        try:
            client_id = p.connect(p.GUI)
            print(f"Created shared GUI connection with client ID: {client_id}")
            
            # Configure the visualization for a wide view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)
            
            # We'll set the camera later when we know how many robots we have
            
            # Enable shadows for better depth perception
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
            
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client_id)
            
        except Exception as e:
            print(f"Warning: Could not create GUI connection: {e}")
            print("Falling back to DIRECT mode")
            client_id = p.connect(p.DIRECT)
    else:
        client_id = p.connect(p.DIRECT)
    
    # Store the client ID
    _SHARED_PYBULLET_CLIENT = client_id
    
    return client_id

# Add a function to adjust the camera based on the number of robots
def adjust_camera_for_robots(client_id, num_robots):
    """
    Adjust the camera to ensure all robots are visible.
    
    Args:
        client_id: PyBullet client ID
        num_robots: Number of robots in the scene
    """
    if client_id is None or p.getConnectionInfo(client_id)['connectionMethod'] != p.GUI:
        return
    
    # Calculate the grid dimensions based on the number of robots
    grid_size = int(np.ceil(np.sqrt(num_robots)))
    
    # Calculate camera distance based on grid size
    # We need to see the entire grid, so distance increases with grid size
    camera_distance = max(3.5, grid_size * 1.5)
    
    # Set camera to a wide view to see all robots
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,  # Adjusted based on number of robots
        cameraYaw=45,                    # Angled view
        cameraPitch=-35,                 # From above
        cameraTargetPosition=[0.0, 0.0, 0.3],  # Center of the scene
        physicsClientId=client_id
    )
    
    print(f"Camera adjusted for {num_robots} robots with distance {camera_distance}")

# Modify the FANUCRobotEnv class to accept a client parameter
class FANUCRobotEnv:
    def __init__(self, render=True, verbose=False, client=None):
        # Store verbose flag
        self.verbose = verbose
        
        # Store render mode
        self.render_mode = render
        
        # Connect to the physics server using the provided client or create a new one
        if client is not None:
            self.client = client
        else:
            # Connect to the physics server using the shared client function
            self.client = get_shared_pybullet_client(render=render)
            
        if self.verbose:
            print(f"Connected to PyBullet physics server with client ID: {self.client}")
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane - keeping it visible as requested
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set collision filter for the plane to interact with all robots
        # The plane is in group 0 and can collide with all groups
        p.setCollisionFilterGroupMask(
            self.plane_id,
            -1,  # Base link of the plane
            0,   # Group 0
            0xFFFF,  # Mask: collide with all groups
            physicsClientId=self.client
        )
        
        # Robot parameters from the documentation
        self.dof = 6  # 6 degrees of freedom
        self.max_force = 100  # Maximum force for joint motors
        self.position_gain = 0.3
        self.velocity_gain = 1.0
        
        # Load the robot URDF (you'll need to create this based on the manual specs)
        # For now, we'll use a placeholder
        self.robot_id = self._load_robot()
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        
        # Joint limits from manual
        self.joint_limits = {
            0: [-720, 720],  # J1 axis - physical limit (multiple rotations allowed)
            1: [-360, 360],  # J2 axis - physical limit
            2: [-360, 360],  # J3 axis - physical limit
            3: [-720, 720],  # J4 axis - physical limit (multiple rotations allowed)
            4: [-360, 360],  # J5 axis - physical limit
            5: [-1080, 1080]  # J6 axis - physical limit (multiple rotations allowed)
        }
        
        # Convert to radians
        for joint, limits in self.joint_limits.items():
            self.joint_limits[joint] = [np.deg2rad(limits[0]), np.deg2rad(limits[1])]
        
        # Set up collision detection: disable self-collisions for all links except end effector
        self._configure_collision_detection()
            
        # Initial configuration
        self.reset()
    
    def _configure_collision_detection(self):
        """
        Configure collision detection for the robot:
        1. Disable self-collisions between robot links
        2. Keep only end effector collision enabled
        """
        if self.verbose:
            print("Configuring robot collision detection: end effector only")
            
        # Define collision groups
        ROBOT_BODY_GROUP = 1  # Group for the robot body
        EE_GROUP = 2          # Group for the end effector
        
        # End effector is the last link in the chain
        ee_link_id = self.dof - 1
        
        # Set all links except the end effector to not collide with each other
        for i in range(self.num_joints):
            if i != ee_link_id:
                # Set this link to the robot body group
                p.setCollisionFilterGroupMask(
                    self.robot_id,
                    i,
                    ROBOT_BODY_GROUP,  # This link is in the robot body group
                    0,  # This link doesn't collide with any other link
                    physicsClientId=self.client
                )
            else:
                # Set the end effector to its own group that can collide
                p.setCollisionFilterGroupMask(
                    self.robot_id,
                    i,
                    EE_GROUP,  # End effector is in its own group
                    0xFFFF,    # End effector can collide with everything
                    physicsClientId=self.client
                )
        
        # Also set the base link (link_id = -1)
        p.setCollisionFilterGroupMask(
            self.robot_id,
            -1,
            ROBOT_BODY_GROUP,
            0,  # Base doesn't collide with any robot parts
            physicsClientId=self.client
        )
        
        # Store for reference
        self.ee_link_id = ee_link_id
        self.ROBOT_BODY_GROUP = ROBOT_BODY_GROUP
        self.EE_GROUP = EE_GROUP
    
    def _load_robot(self):
        # Load the URDF for the FANUC LR Mate 200iC
        
        # Get the absolute path to the workspace directory
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        
        # Define the absolute path to the URDF file
        urdf_path = os.path.join(workspace_dir, "res", "fanuc_lrmate_200ic.urdf")
        
        # Check if the URDF file exists
        if os.path.exists(urdf_path):
            if self.verbose:
                print(f"Loading FANUC LR Mate 200iC URDF from: {urdf_path}")
            return p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        
        # If we couldn't find the URDF, print a warning and fall back to a simple robot
        print("WARNING: Could not find FANUC LR Mate 200iC URDF file. Falling back to default robot.")
        print("Expected URDF path:", urdf_path)
        print("Current working directory:", os.getcwd())
        
        # Fallback to a simple robot for testing
        return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
    
    def reset(self):
        # Get the current joint states if they exist (for subsequent resets)
        current_joint_states = []
        for i in range(self.dof):
            try:
                state = p.getJointState(self.robot_id, i)
                current_joint_states.append(state[0])  # Joint position
            except:
                # If we can't get the joint state, use home position
                current_joint_states = None
                break
        
        # On first reset, use home position
        if current_joint_states is None or not hasattr(self, 'first_reset_done'):
            # Reset to home position only on the first reset
            home_position = [0, 0, 0, 0, 0, 0]  # All joints at 0 position
            for i, pos in enumerate(home_position):
                p.resetJointState(self.robot_id, i, pos)
            self.first_reset_done = True
        else:
            # On subsequent resets, keep the current position
            for i, pos in enumerate(current_joint_states):
                p.resetJointState(self.robot_id, i, pos)
        
        # Get current state
        state = self._get_state()
        return state
        
    def step(self, action):
        """
        Apply action to the robot
        
        Args:
            action: A list of 6 values representing power and direction for each motor
                   -100 = full power in one direction
                   0 = not moving at all
                   100 = full power in the other direction
        """
        # Check if the global target randomization time has been updated since this robot's last reset
        # If so, we need to sample a new target for this robot
        if get_target_randomization_time() > self.last_target_randomization_time:
            # Update this robot's last target randomization time
            self.last_target_randomization_time = get_target_randomization_time()
            
            # Sample a new target position
            self.target_position = self._sample_target()
            
            if self.verbose:
                print(f"Robot {self.rank} received a new target at {self.target_position}")
            
            # Visualize the new target if rendering is enabled
            if self.gui:
                try:
                    # Remove previous target visualization if it exists
                    if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
                        try:
                            p.removeBody(self.target_visual_id, physicsClientId=self.robot.client)
                        except:
                            pass
                    
                    # Create new target visualization
                    self.target_visual_id = visualize_target(self.target_position, self.robot.client)
                except Exception as e:
                    print(f"Warning: Could not visualize target: {e}")
        
        # Get current joint states for limit checking
        current_state = self.robot._get_state()
        current_joint_positions = current_state[:self.robot.dof*2:2]  # Extract joint positions
        current_joint_velocities = current_state[1:self.robot.dof*2:2]  # Extract joint velocities
        
        # Get the current end effector position
        current_ee_pos = current_state[12:15]
        
        # Calculate distance to target (in meters) for velocity scaling
        distance_to_target = np.linalg.norm(current_ee_pos - self.target_position)
        
        # Calculate distance factor for adaptive velocity scaling
        # Closer to target = more precise/slower movements
        distance_factor = np.clip(distance_to_target / (self.workspace_size * 0.5), 0.1, 1.0)
        
        # Apply the action directly as motor powers/velocities
        # Convert the power/direction values (-100 to 100) to appropriate joint velocities
        joint_velocities = []
        
        for i, power in enumerate(action):
            # Convert the -100 to 100 power value to actual velocity
            # Maximum velocity is scaled based on joint limits and distance to target
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                
                # Calculate current position with full precision
                current_pos = float(current_joint_positions[i])
                current_vel = float(current_joint_velocities[i])
                
                # Get the distance to each limit with higher precision
                distance_to_low = current_pos - limit_low
                distance_to_high = limit_high - current_pos
                
                # Calculate joint range for velocity scaling
                joint_range = limit_high - limit_low
                
                # Calculate adaptive maximum velocity 
                # - Scale based on distance to target (closer = more precise)
                # - Scale based on joint range (larger range = higher max velocity)
                # - Include inertia factor to prevent abrupt changes
                base_max_vel = joint_range * 0.12 * distance_factor  # 12% of range per step
                
                # Add inertia factor - limit acceleration/deceleration
                # This helps prevent abrupt direction changes that can cause instability
                if i < len(self.previous_action):
                    prev_power = self.previous_action[i]
                    # Limit power change to 30% per step for smoother motion
                    max_power_change = 30.0
                    power = prev_power + np.clip(power - prev_power, -max_power_change, max_power_change)
                
                # Calculate target velocity with adaptive scaling
                velocity = (power / 100.0) * base_max_vel
                
                # Enforce limits when approaching boundaries using sigmoid function
                # for smoother transition (smoother gradient)
                limit_threshold = joint_range * 0.05  # 5% of range
                
                if distance_to_low < limit_threshold and velocity < 0:
                    # Close to lower limit and trying to move further down
                    # Use sigmoid scaling for smoother transition
                    sigmoid_factor = 1.0 / (1.0 + np.exp(-(10.0 * distance_to_low / limit_threshold - 5.0)))
                    velocity = velocity * sigmoid_factor
                    
                    if self.verbose and sigmoid_factor < 0.5:
                        print(f"Joint {i} approaching lower limit: {current_pos:.4f} rad (limit: {limit_low:.4f} rad)")
                
                elif distance_to_high < limit_threshold and velocity > 0:
                    # Close to upper limit and trying to move further up
                    # Use sigmoid scaling for smoother transition
                    sigmoid_factor = 1.0 / (1.0 + np.exp(-(10.0 * distance_to_high / limit_threshold - 5.0)))
                    velocity = velocity * sigmoid_factor
                    
                    if self.verbose and sigmoid_factor < 0.5:
                        print(f"Joint {i} approaching upper limit: {current_pos:.4f} rad (limit: {limit_high:.4f} rad)")
                
                # Add clamped velocity to the list (use float32 for compatibility with physics engine)
                joint_velocities.append(np.float32(velocity))
            else:
                # Default velocity for joints without limits
                # Scale by distance factor for adaptive control
                joint_velocities.append(np.float32(power * 0.1 * distance_factor))
        
        # Pass the action to the FANUCRobotEnv by passing None for positions (velocity control only)
        next_state = self.robot.step((None, joint_velocities))
        
        # Store the current action for next observation
        self.previous_action = np.array(action)
        
        # Increment step counter
        self.steps += 1
        
        # Use the already calculated distance, no need to recalculate
        # Just ensure we're using the same state as for other calculations
        
        # Create an info dictionary to store additional information
        info = {}
        info["distance_cm"] = distance_to_target * 100  # Keep this in cm for easier reading
        
        # Check for collisions - ground and end effector self-collisions
        ground_collision, ee_self_collision, collision_info = self._detect_collisions()
        info["collision_info"] = collision_info
        
        # Improved reward system with more components
        # Base reward is 0
        reward = 0
        
        # Check if we've reached the target
        if distance_to_target <= self.accuracy_threshold:
            # Target reached! Maximum reward
            reward = 50.0  # Increased from 20.0 to 50.0 for stronger signal
            done = True
            info["target_reached"] = True
        else:
            # Not at target yet
            
            # Calculate progress compared to previous step with higher precision
            if hasattr(self, 'prev_distance'):
                # Progress is positive if we got closer to the target
                progress = self.prev_distance - distance_to_target
                
                # Add a linear progress reward component with adaptive scaling
                # Scale factor is inversely proportional to current distance
                # This rewards small improvements more when already close to target
                progress_scale = 15.0 * min(1.0, self.accuracy_threshold / max(distance_to_target, 1e-6))
                progress_reward = progress * progress_scale
                
                # Only reward positive progress to avoid exploits
                if progress > 0:
                    reward += progress_reward
                    info["progress_reward"] = progress_reward
                    
                    # Add extra bonus for continuous improvement
                    if hasattr(self, 'prev_progress') and progress > self.prev_progress:
                        continuous_improvement_bonus = progress * 5.0
                        reward += continuous_improvement_bonus
                        info["continuous_improvement_bonus"] = continuous_improvement_bonus
                
                # Store current progress for next step
                self.prev_progress = progress
            
            # Store current distance for next step with full precision
            self.prev_distance = distance_to_target
            
            # Linear distance reward instead of exponential
            # The closer to target, the higher the reward
            # Use softmax-style normalization for better numerical stability
            normalized_distance = np.exp(-distance_to_target / (self.workspace_size * 0.25))
            
            # Linear scale with diminishing returns at extreme distances
            distance_reward = 3.0 * normalized_distance
            
            # Apply the distance reward
            reward += distance_reward
            info["distance_reward"] = distance_reward
            
            # Calculate robot compactness (how curled up the robot is)
            compactness = self._calculate_robot_compactness()
            
            # Track the minimum compactness (most extended configuration)
            if compactness < self.min_compactness_in_episode:
                self.min_compactness_in_episode = compactness
                info["min_compactness"] = self.min_compactness_in_episode
            
            # Create a compactness penalty that's inverse to the distance reward
            # Use a smoother function to avoid discontinuities in the gradient
            compactness_penalty_factor = 3.0  # Increased from 2.0 for stronger signal
            compactness_penalty = compactness_penalty_factor * compactness * (1.0 - normalized_distance)
            
            # Apply the compactness penalty
            reward -= compactness_penalty
            info["compactness"] = compactness
            info["compactness_penalty"] = compactness_penalty
            
            if self.verbose and compactness > 0.7 and self.steps % 20 == 0:
                print(f"Robot {self.rank}: High compactness detected ({compactness:.2f}), penalty: {compactness_penalty:.2f}")
            
            # Add a penalty for staying near the center of the workspace
            # Calculate distance from home position (center of workspace)
            home_distance = np.linalg.norm(current_ee_pos - self.home_position)
            
            # Use sigmoid normalization for smoother gradients
            normalized_home_distance = 2.0 / (1.0 + np.exp(-5.0 * home_distance / self.workspace_size)) - 1.0
            
            # Penalize being close to the center (higher penalty when closer to center)
            center_penalty = 0.8 * (1.0 - normalized_home_distance)
            
            # Only apply center penalty when we're not very close to the target
            if distance_to_target > self.accuracy_threshold * 3:
                reward -= center_penalty
                info["center_penalty"] = center_penalty
            
            # Simple fixed penalty for ground collision
            if ground_collision:
                reward -= self.ground_collision_penalty  # Use the class variable for consistency
                info["ground_collision"] = True
            
            # Increased penalty for end effector self-collision to strongly discourage this behavior
            if ee_self_collision:
                # More severe penalty for end effector collisions (2x the ground collision penalty)
                ee_collision_penalty = self.ground_collision_penalty * 2.0
                reward -= ee_collision_penalty
                info["ee_self_collision"] = True
                info["ee_collision_penalty"] = ee_collision_penalty
                
                if self.verbose and not hasattr(self, 'collision_notified'):
                    print(f"\nRobot {self.rank}: End effector collision detected with link {collision_info['ee_collision_links']}")
                    print(f"Applied penalty: {ee_collision_penalty:.2f}\n")
                    self.collision_notified = True
                    
                    # Visualize the collision if there are colliding links
                    if collision_info['ee_collision_links']:
                        # Take the first collision pair
                        ee_link, other_link = collision_info['ee_collision_links'][0]
                        self._visualize_collision(ee_link, other_link)
            
            # Check timeout
            if self.steps >= self.timeout_steps:
                done = True
                info["timeout"] = True
            else:
                done = False
        
        # Store reward and distance info
        info["reward"] = reward
        
        # Track best position achieved during the episode
        if distance_to_target < self.best_distance_in_episode:
            self.best_distance_in_episode = distance_to_target
            self.best_position_in_episode = current_ee_pos.copy()
            info["best_distance"] = self.best_distance_in_episode * 100  # Convert to cm
            info["best_position"] = self.best_position_in_episode
            
            # Log when the robot achieves a distance within the curriculum threshold
            if distance_to_target <= self.target_expansion_threshold and self.verbose:
                print(f"\nRobot {self.rank}: Achieved curriculum threshold distance: {distance_to_target*100:.2f}cm!")
                print(f"Current curriculum level: {self.curriculum_level}, max target distance: {self.max_target_distance:.2f}m\n")
        
        # Check if the robot's best position is better than its initial position
        # This will be used to determine if model updates should occur
        position_improved = self.best_distance_in_episode < self.initial_distance_to_target
        improvement_amount = self.initial_distance_to_target - self.best_distance_in_episode
        
        # Add improvement info to the info dictionary
        info["position_improved"] = position_improved
        info["improvement_amount"] = improvement_amount * 100  # Convert to cm
        info["initial_distance"] = self.initial_distance_to_target * 100  # Convert to cm
        info["min_compactness"] = self.min_compactness_in_episode
        info["current_compactness"] = compactness
        
        # If the episode is ending, log the improvement information
        if done and self.verbose:
            if position_improved:
                print(f"\nRobot {self.rank}: Position IMPROVED by {improvement_amount*100:.2f}cm")
                print(f"  Initial distance: {self.initial_distance_to_target*100:.2f}cm")
                print(f"  Best distance: {self.best_distance_in_episode*100:.2f}cm")
                print(f"  Min compactness: {self.min_compactness_in_episode:.2f} (lower is more extended)")
                print(f"  Model weights WILL be updated\n")
            else:
                print(f"\nRobot {self.rank}: Position did NOT improve")
                print(f"  Initial distance: {self.initial_distance_to_target*100:.2f}cm")
                print(f"  Best distance: {self.best_distance_in_episode*100:.2f}cm")
                print(f"  Min compactness: {self.min_compactness_in_episode:.2f} (lower is more extended)")
                print(f"  Model weights will NOT be updated\n")
        
        # Get observation for next step
        observation = self._get_observation()
        
        return observation, reward, done, False, info
    
    def _get_state(self):
        # Get joint states
        joint_states = []
        for i in range(self.dof):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])  # Joint position
            joint_states.append(state[1])  # Joint velocity
        
        # Get end-effector position and orientation
        ee_link_state = p.getLinkState(self.robot_id, self.dof-1)
        ee_position = ee_link_state[0]
        ee_orientation = ee_link_state[1]
        
        # Combine for full state
        state = np.array(joint_states + list(ee_position) + list(ee_orientation))
        return state
    
    def close(self):
        # We don't disconnect the client here since it's shared
        pass

    def _apply_robot_offset(self):
        """Apply the robot offset to the robot's position"""
        # Get the robot's position
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot.robot_id, physicsClientId=self.robot.client)
        
        # Apply the offset
        new_pos = (robot_pos[0] + self.offset_x, robot_pos[1], robot_pos[2])
        
        # Set the new position
        p.resetBasePositionAndOrientation(self.robot.robot_id, new_pos, robot_orn, physicsClientId=self.robot.client)
        
        if self.verbose:
            print(f"Applied offset of {self.offset_x} to Robot {self.rank}")
            print(f"Robot {self.rank} positioned at [{new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}]")
    
    def _visualize_reachable_workspace(self):
        """
        Visualize the reachable workspace of the robot.
        Shows only the maximum reach distance as a sphere.
        """
        # Get the robot's base position
        robot_base_pos, _ = p.getBasePositionAndOrientation(
            self.robot.robot_id, 
            physicsClientId=self.robot.client
        )
        
        # Add a vertical offset to center at the robot's shoulder height
        # The FANUC LR Mate 200iC has its first joint (shoulder) about 0.33m above the base
        shoulder_height = 0.33
        shoulder_position = np.array([
            robot_base_pos[0],
            robot_base_pos[1],
            robot_base_pos[2] + shoulder_height
        ])
        
        # Define the range of distances that are reachable
        min_reach = 0.15  # Minimum reach distance (kept for reference but not visualized)
        # Use the workspace_size for the maximum reach to match collision detection
        max_reach = self.workspace_size
        
        # Define the robot base radius (kept for reference but not visualized)
        robot_base_radius = 0.15  # Approximate radius of the robot's base
        
        # Define ground clearance
        ground_clearance = 0.1  # 10cm above the ground
        
        # Remove any previous visualizations
        if hasattr(self, 'viz_ids') and self.viz_ids:
            for viz_id in self.viz_ids:
                p.removeBody(viz_id, physicsClientId=self.robot.client)
        
        # Initialize visualization IDs list
        self.viz_ids = []
        
        # Create a visual sphere for the maximum reach
        max_sphere_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=max_reach,
            rgbaColor=[0, 0, 1, 0.3],  # Blue, semi-transparent
            physicsClientId=self.robot.client
        )
        max_sphere_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=max_sphere_id,
            basePosition=shoulder_position,
            physicsClientId=self.robot.client
        )
        self.viz_ids.append(max_sphere_body)
        
        # Create invisible ground clearance plane for functionality (completely transparent)
        clearance_plane_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[2.0, 2.0, 0.001],  # Very thin box
            rgbaColor=[0, 0, 0, 0],  # Completely transparent
            physicsClientId=self.robot.client
        )
        clearance_plane_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=clearance_plane_id,
            basePosition=[robot_base_pos[0], robot_base_pos[1], self.ground_plane_height + ground_clearance],
            physicsClientId=self.robot.client
        )
        self.viz_ids.append(clearance_plane_body)
        
        if self.verbose:
            print(f"Visualized reachable workspace:")
            print(f"  - Maximum reach: {max_reach:.2f}m (blue sphere)")

    def _detect_collisions(self):
        """
        Enhanced collision detection:
        1. Ground plane collisions (as before)
        2. End effector collision with other robot parts
        
        Returns:
            tuple: (ground_collision, ee_self_collision, collision_info)
        """
        # Get current robot state
        state = self.robot._get_state()
        
        # Get end effector position
        ee_position = state[12:15]
        
        # Check for ground plane collisions - only check end effector and last two links
        ground_collision = False
        ground_collision_links = []
        
        # Only check the last 3 links (including end effector)
        critical_links = [self.robot.num_joints-3, self.robot.num_joints-2, self.robot.num_joints-1]
        
        for i in critical_links:
            if i >= 0 and i < self.robot.num_joints:  # Ensure valid index
                link_state = p.getLinkState(self.robot.robot_id, i, physicsClientId=self.client_id)
                link_pos = link_state[0]  # Link position
                
                # Check if link is too close to or below the ground plane
                if link_pos[2] <= self.ground_plane_height + self.ground_collision_threshold:
                    ground_collision = True
                    ground_collision_links.append(i)
                    break  # Exit early once a collision is found
        
        # Check for end effector self-collisions
        ee_self_collision = False
        ee_collision_links = []
        
        # Make sure the robot has the ee_link_id attribute (added in _configure_collision_detection)
        if hasattr(self.robot, 'ee_link_id'):
            robot_id = self.robot.robot_id
            ee_link = self.robot.ee_link_id
            
            # Get all contact points for the end effector
            contact_points = p.getContactPoints(
                bodyA=robot_id,
                linkIndexA=ee_link,
                physicsClientId=self.client_id
            )
            
            # Check if any contact involves the end effector and another part of the robot
            for point in contact_points:
                if point[1] == robot_id:  # If bodyB is also our robot
                    link_b = point[3]
                    # Check if this is another part of the robot, not the end effector itself
                    # Also exclude the link immediately before end effector which might be in continuous contact
                    if link_b != ee_link and link_b != ee_link-1 and link_b != -1:  
                        ee_self_collision = True
                        ee_collision_links.append((ee_link, link_b))
                        break
        
        # Calculate distance from home position (center of workspace)
        distance_from_home = np.linalg.norm(ee_position - self.home_position)
        
        # Prepare collision info dictionary
        collision_info = {
            "ground_collision": ground_collision,
            "ground_collision_links": ground_collision_links,
            "ee_self_collision": ee_self_collision,
            "ee_collision_links": ee_collision_links,
            "ee_position": ee_position,
            "distance_from_home": distance_from_home
        }
        
        # Return with both ground and end effector collision information
        return ground_collision, ee_self_collision, collision_info

    def _cleanup_visualization(self):
        """Clean up all visualization resources to prevent memory leaks"""
        # Clean up end effector trajectory markers
        if hasattr(self, 'ee_markers') and self.ee_markers:
            for marker_id in self.ee_markers:
                try:
                    p.removeBody(marker_id, physicsClientId=self.robot.client)
                except Exception:
                    pass  # Silently ignore errors
            self.ee_markers = []
            
        # Clean up trajectory markers
        if hasattr(self, 'trajectory_markers') and self.trajectory_markers:
            for marker_id in self.trajectory_markers:
                try:
                    p.removeBody(marker_id, physicsClientId=self.robot.client)
                except Exception:
                    pass  # Silently ignore errors
            self.trajectory_markers = []
            
        # Clean up collision visualization markers
        if hasattr(self, 'collision_markers') and self.collision_markers:
            for marker_id in self.collision_markers:
                try:
                    p.removeBody(marker_id, physicsClientId=self.robot.client)
                except Exception:
                    pass  # Silently ignore errors
            self.collision_markers = []
            
        # Clean up collision line visualization
        if hasattr(self, 'collision_line') and self.collision_line is not None:
            try:
                p.removeUserDebugItem(self.collision_line, physicsClientId=self.robot.client)
            except Exception:
                pass  # Silently ignore errors
            self.collision_line = None
            
        # Clear the marker creation steps tracking
        if hasattr(self, 'marker_creation_steps'):
            self.marker_creation_steps = {}
        
        # Clean up target visualization
        if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.robot.client)
                self.target_visual_id = None
            except Exception:
                pass  # Silently ignore errors
                
        # Clean up target line
        if hasattr(self, 'target_line_id') and self.target_line_id is not None:
            try:
                p.removeUserDebugItem(self.target_line_id, physicsClientId=self.robot.client)
                self.target_line_id = None
            except Exception:
                pass  # Silently ignore errors
        
        # Clean up workspace visualization
        if hasattr(self, 'viz_ids') and self.viz_ids:
            for viz_id in self.viz_ids:
                try:
                    p.removeBody(viz_id, physicsClientId=self.robot.client)
                except Exception:
                    pass  # Silently ignore errors
            self.viz_ids = []

    def _sample_target(self):
        """
        Sample a target position for the robot to reach.
        Uses the curriculum-adjusted workspace of the robot, with distances relative to the shoulder position.
        Ensures targets don't spawn in contact with the robot's base or the ground.
        
        Returns:
            numpy.ndarray: Target position [x, y, z]
        """
        # Get the robot's base position
        robot_base_pos, _ = p.getBasePositionAndOrientation(
            self.robot.robot_id, 
            physicsClientId=self.robot.client
        )
        
        # Add a vertical offset to center at the robot's shoulder height
        # The FANUC LR Mate 200iC has its first joint (shoulder) about 0.33m above the base
        shoulder_height = 0.33
        shoulder_position = np.array([
            robot_base_pos[0],
            robot_base_pos[1],
            robot_base_pos[2] + shoulder_height
        ])
        
        # Define the range of distances that are reachable
        # Use the curriculum-adjusted target distance
        min_reach = 0.25  # Minimum reach to avoid targets too close to the center
        max_reach = self.max_target_distance  # Use curriculum-adjusted maximum target distance
        
        if self.verbose:
            print(f"Sampling target with curriculum level {self.curriculum_level}, max distance: {max_reach:.2f}m")
        
        # Define the robot base radius to avoid spawning targets inside or too close to the base
        robot_base_radius = 0.15  # Approximate radius of the robot's base
        
        # Define ground clearance to ensure targets are not in contact with or below the ground
        ground_clearance = 0.1  # 10cm above the ground
        
        # Use the ground plane height for ground level
        
        # Maximum attempts to find a valid target
        max_attempts = 100
        
        # Choose a sampling strategy - bias towards outer regions of workspace
        sampling_strategy = np.random.choice(['uniform', 'outer_bias', 'structured'], 
                                            p=[0.2, 0.5, 0.3])
        
        for _ in range(max_attempts):
            # Different sampling strategies to create more variety
            if sampling_strategy == 'uniform':
                # Uniform sampling in spherical coordinates
                theta = np.random.uniform(0, np.pi)  # Polar angle from Z axis (0 to pi)
                phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle (0 to 2pi)
                distance = np.random.uniform(min_reach, max_reach)
                
            elif sampling_strategy == 'outer_bias':
                # Bias towards outer regions of workspace using beta distribution
                theta = np.random.uniform(0, np.pi)  # Polar angle from Z axis (0 to pi)
                phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle (0 to 2pi)
                # Beta distribution with alpha=1, beta=0.7 biases towards higher values (outer radius)
                normalized_distance = np.random.beta(1.0, 0.7)  
                distance = min_reach + normalized_distance * (max_reach - min_reach)
                
            else:  # 'structured'
                # Generate targets at specific segments of the workspace
                # This ensures coverage of different areas in a more structured way
                segment = np.random.randint(0, 8)  # Divide workspace into 8 segments
                
                # Each segment corresponds to a specific region
                phi_segment = (segment % 4) * (np.pi/2) + np.random.uniform(-np.pi/4, np.pi/4)
                theta_segment = np.pi/3 if segment < 4 else 2*np.pi/3 + np.random.uniform(-np.pi/6, np.pi/6)
                
                # Use maximum reach for structured targets to encourage reaching
                distance = np.random.uniform(0.6 * max_reach, max_reach)
                
                theta = theta_segment
                phi = phi_segment
            
            # Convert spherical coordinates to Cartesian
            x = distance * np.sin(theta) * np.cos(phi)
            y = distance * np.sin(theta) * np.sin(phi)
            z = distance * np.cos(theta)
            
            # Add to shoulder position - this ensures the target is relative to the robot's actual position
            target_position = shoulder_position + np.array([x, y, z])
            
            # Calculate horizontal distance from base to target (ignoring height)
            horizontal_distance = np.sqrt((target_position[0] - robot_base_pos[0])**2 + 
                                         (target_position[1] - robot_base_pos[1])**2)
            
            # Check if the target is above the ground with sufficient clearance
            is_above_ground = target_position[2] >= (self.ground_plane_height + ground_clearance)
            
            # Ensure the target is not too close to the robot's base horizontally
            is_not_in_base = horizontal_distance > robot_base_radius
            
            # If the target is valid, return it
            if is_above_ground and is_not_in_base:
                if self.verbose:
                    print(f"Valid target position sampled at {target_position} using {sampling_strategy} strategy")
                return target_position
        
        # If we couldn't find a valid target after max_attempts, use a fallback method
        # Generate a point in a more distant location to encourage movement
        # Pick a random direction instead of always in front
        random_angle = np.random.uniform(0, 2 * np.pi)
        fallback_position = shoulder_position + np.array([
            0.5 * np.cos(random_angle),  # Random direction
            0.5 * np.sin(random_angle),  # Random direction
            0.2
        ])
        
        if self.verbose:
            print(f"Using fallback target position at {fallback_position}")
        
        return fallback_position
        
    def _get_observation(self):
        """Get the current observation according to the new requirements"""
        # Get current state from robot environment
        state = self.robot._get_state()
        
        # Extract joint positions
        joint_positions = state[:self.robot.dof*2:2]  # Extract joint positions
        
        # Normalize joint positions using full floating-point precision
        # Instead of clipping to 0-100 range, use the full range of floating point values
        normalized_joint_positions = []
        for i, pos in enumerate(joint_positions):
            if i in self.robot.joint_limits:
                limit_low, limit_high = self.robot.joint_limits[i]
                
                # Calculate the full range of the joint
                joint_range = limit_high - limit_low
                
                # Normalize relative to midpoint for more numerical stability
                # This centers the values around 0, which is better for neural networks
                midpoint = (limit_high + limit_low) / 2.0
                normalized_value = (pos - midpoint) / (joint_range / 2.0)
                
                # Scale to -100 to 100 range for consistency with actions
                # No capping is applied to preserve the full signal
                norm_pos = normalized_value * 100.0
                normalized_joint_positions.append(norm_pos)
            else:
                # For joints without limits, use a default normalized position
                normalized_joint_positions.append(0.0)  # Center point (changed from 50.0)
        
        # Convert to numpy array with float32 precision (adequate for neural networks while being efficient)
        normalized_joint_positions = np.array(normalized_joint_positions, dtype=np.float32)
        
        # Extract end effector position
        ee_position = state[12:15]
        
        # Calculate relative position (target - ee)
        relative_position = self.target_position - ee_position
        
        # Calculate distance to target (vector length between end effector and target in mm)
        # Preserving full floating point precision
        distance_to_target = np.linalg.norm(relative_position) * 1000.0  # Convert to mm
        
        # Add normalized distance component (0.0-1.0) to provide better learning signal
        normalized_distance = np.clip(distance_to_target / (self.workspace_size * 1000.0), 0.0, 1.0)
        
        # Combine all observations into a single array according to new requirements
        observation = np.concatenate([
            self.previous_action,         # Previous outputs (-100 to 100 for each motor)
            normalized_joint_positions,   # Current joint angles (-100 to 100 for each joint)
            [distance_to_target],         # Current vector length to target (in mm)
            [normalized_distance]         # Normalized distance (0-1 range)
        ])
        
        return observation

    def _visualize_collision(self, link_a, link_b):
        """
        Visualize the collision between two links by temporarily highlighting them.
        
        Args:
            link_a: First link index involved in collision
            link_b: Second link index involved in collision
        """
        if not self.robot.render:
            return  # Skip visualization if rendering is disabled
            
        # Colors for highlighting the colliding links
        COLLISION_COLOR_A = [1, 0, 0, 0.7]  # Red with alpha
        COLLISION_COLOR_B = [1, 0.5, 0, 0.7]  # Orange with alpha
        
        # Save the original visual states of the links
        link_a_visual_state = p.getVisualShapeData(self.robot.robot_id, link_a, physicsClientId=self.client_id)
        link_b_visual_state = p.getVisualShapeData(self.robot.robot_id, link_b, physicsClientId=self.client_id)
        
        # Get the positions of the colliding links
        link_a_state = p.getLinkState(self.robot.robot_id, link_a, physicsClientId=self.client_id)
        link_b_state = p.getLinkState(self.robot.robot_id, link_b, physicsClientId=self.client_id)
        link_a_pos = link_a_state[0]
        link_b_pos = link_b_state[0]
        
        # Create temporary visual markers to highlight collision
        marker_a = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=COLLISION_COLOR_A,
            physicsClientId=self.client_id
        )
        
        marker_b = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=COLLISION_COLOR_B,
            physicsClientId=self.client_id
        )
        
        # Create bodies for the markers (no physics)
        marker_body_a = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_a,
            basePosition=link_a_pos,
            physicsClientId=self.client_id
        )
        
        marker_body_b = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_b,
            basePosition=link_b_pos,
            physicsClientId=self.client_id
        )
        
        # Draw a line between the collision points
        line_id = p.addUserDebugLine(
            link_a_pos,
            link_b_pos,
            lineColorRGB=[1, 0, 0],
            lineWidth=3.0,
            lifeTime=0.5,  # 0.5 seconds
            physicsClientId=self.client_id
        )
        
        # Store markers for cleanup
        self.collision_markers = [marker_body_a, marker_body_b]
        self.collision_line = line_id
        
        # Log details about the collision
        if self.verbose:
            print(f"Visualizing collision between links {link_a} and {link_b}")
            print(f"Link {link_a} position: {link_a_pos}")
            print(f"Link {link_b} position: {link_b_pos}")
            print(f"Distance between links: {np.linalg.norm(np.array(link_a_pos) - np.array(link_b_pos)):.4f}m")

    def _calculate_robot_compactness(self):
        """
        Calculate a measure of the robot's configuration compactness with high precision.
        Higher value = more curled up/compact configuration.
        Lower value = more extended configuration.
        
        Returns:
            float: Compactness score (higher = more compact/curled up)
        """
        # Get positions of all links with full floating-point precision
        link_positions = []
        link_orientations = []
        
        # Skip the base link (index -1) and start from 0
        for i in range(self.robot.num_joints):
            # Get full state including position and orientation
            link_state = p.getLinkState(self.robot.robot_id, i, computeLinkVelocity=0, computeForwardKinematics=1, physicsClientId=self.client_id)
            link_pos = np.array(link_state[0], dtype=np.float64)  # Link position (x, y, z) with higher precision
            link_orn = np.array(link_state[1], dtype=np.float64)  # Link orientation as quaternion
            link_positions.append(link_pos)
            link_orientations.append(link_orn)
            
        # Calculate weighted pairwise distances between links
        # Focus more on distant links (higher weight for non-adjacent links)
        total_weighted_distance = 0.0
        total_weight = 0.0
        
        # Reference is fully extended robot (maximum distances)
        # We'll normalize relative to this theoretical maximum
        max_robot_length = self.workspace_size * 0.8  # Theoretical maximum length if fully extended
        
        # Calculate center of mass of the robot
        if len(link_positions) > 0:
            center_of_mass = np.mean(link_positions, axis=0)
        else:
            center_of_mass = np.zeros(3)
            
        # Calculate average distance from center of mass
        total_distance_from_com = 0.0
        
        for i in range(len(link_positions)):
            # Distance from center of mass
            pos_i = link_positions[i]
            dist_from_com = np.linalg.norm(pos_i - center_of_mass)
            total_distance_from_com += dist_from_com
            
            # Pairwise distances between links
            for j in range(i + 1, len(link_positions)):
                # Calculate importance weight based on link indices
                # More distant links in the chain have higher weights
                link_distance_in_chain = abs(i - j)
                
                # Skip adjacent links (which are naturally close)
                if link_distance_in_chain > 1:
                    pos_j = link_positions[j]
                    
                    # Calculate Euclidean distance with high precision
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Weight is higher for links that are farther apart in the chain
                    # This emphasizes the overall extension of the robot
                    weight = float(link_distance_in_chain) ** 1.5  # Non-linear weighting
                    
                    total_weighted_distance += distance * weight
                    total_weight += weight
        
        # Avoid division by zero
        if total_weight == 0.0 or len(link_positions) == 0:
            return 0.0
            
        # Calculate average weighted distance
        average_weighted_distance = total_weighted_distance / total_weight
        
        # Calculate average distance from center of mass
        avg_distance_from_com = total_distance_from_com / len(link_positions) if len(link_positions) > 0 else 0.0
        
        # Combine both metrics with weights:
        # - 70% based on pairwise distances (overall extension)
        # - 30% based on distance from center of mass (spread)
        pairwise_compactness = 1.0 - min(1.0, average_weighted_distance / max_robot_length)
        com_compactness = 1.0 - min(1.0, avg_distance_from_com / (max_robot_length * 0.5))
        
        # Weighted combination for final compactness score (0.0 to 1.0)
        compactness = 0.7 * pairwise_compactness + 0.3 * com_compactness
        
        return compactness

# Custom neural network architecture for the policy with the new requirements
class CustomActorCriticNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(CustomActorCriticNetwork, self).__init__()
        
        # Input dimension: 
        # - Previous outputs (6 values, -100 to 100)
        # - Current joint angles (6 values, -100 to 100)
        # - Current vector length to target (1 value, in mm)
        # - Normalized distance to target (1 value, 0-1)
        input_dim = 14  # 6 (previous outputs) + 6 (joint angles) + 1 (distance) + 1 (normalized distance)
        
        # Massively larger shared feature extractor with more layers and width
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Increased from 512 to 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),  # Increased from 512 to 2048
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1536),  # Increased from 384 to 1536
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 1024),  # Increased from 256 to 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 768),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Larger actor network (policy) with more layers
        self.actor_net = nn.Sequential(
            nn.Linear(feature_dim, 1024),  # Increased from 256 to 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 768),  # Increased from 192 to 768
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 384),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, 256),  # Increased from 128 to 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),  # Keep the final layer at 64 to match output layers
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Larger critic network (value function) with more layers
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, 1024),  # Increased from 256 to 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 768),  # Increased from 192 to 768
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 384),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, 256),  # Increased from 128 to 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),  # Added new layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),  # Keep the final layer at 64 to match output layers
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Output layer: power and direction of each motor (-100 to 100)
        # -100 = full power in one direction
        # 0 = not moving at all
        # 100 = full power in the other direction
        self.mean_layer = nn.Sequential(
            nn.Linear(64, 6),  # 6 outputs (one per motor)
            nn.Tanh(),         # Outputs between -1 and 1
            ScaleLayer(-100.0, 100.0)  # Scale to -100 to 100 range
        )
        
        # Log standard deviation layer (for stochastic policy)
        self.log_std_layer = nn.Linear(64, 6)  # 6 outputs to match mean layer
        
        # Value layer
        self.value_layer = nn.Sequential(
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.shared_net(x)
        
        # Actor
        actor_features = self.actor_net(features)
        mean = self.mean_layer(actor_features)  # Outputs 6 values between -100 and 100
        log_std = self.log_std_layer(actor_features)  # Outputs 6 log std values
        
        # Critic
        critic_features = self.critic_net(features)
        value = self.value_layer(critic_features)
        
        return mean, log_std, value

# Custom scaling layer to transform values from one range to another
class ScaleLayer(nn.Module):
    def __init__(self, min_val, max_val, eps=1e-7):
        super(ScaleLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.mid_val = (max_val + min_val) / 2.0
        self.eps = eps  # Small epsilon to prevent numerical issues
        
    def forward(self, x):
        # Input is assumed to be between -1 and 1 (from tanh)
        # Scale to min_val to max_val using full floating-point precision
        
        # First clip the input to the valid range to avoid extreme values
        # This doesn't reduce precision within the valid range
        x_clipped = torch.clamp(x, min=-1.0 + self.eps, max=1.0 - self.eps)
        
        # Scale using the pre-calculated values for better numerical stability
        scaled = x_clipped * (self.range / 2.0) + self.mid_val
        
        # Convert to float32 for compatibility with other layers
        # This doesn't lose precision as the result is already in a reasonable range
        return scaled

# Custom features extractor for the policy
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Input size from observation space (now includes normalized distance)
        n_input = int(np.prod(observation_space.shape))
        
        # Much larger network architecture with more layers and width
        self.net = nn.Sequential(
            nn.Linear(n_input, 1024),  # Increased from 256 to 1024
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),  # Added new layer with increased width
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1536),  # Wider middle layers
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, features_dim),  # Output dimension unchanged
            nn.ReLU(),
            nn.BatchNorm1d(features_dim)
        )
    
    def forward(self, observations):
        # Ensure observations are in float32 for consistent hardware utilization
        if observations.dtype != torch.float32:
            observations = observations.to(torch.float32)
        return self.net(observations)

# Callback for saving models and logging
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
    
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/model_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        return True

# Add this class to monitor and report training progress
class TrainingMonitorCallback(BaseCallback):
    """
    Callback for monitoring training progress and logging metrics.
    All metrics are standardized to be within the 0 to 1 range.
    """
    
    def __init__(self, log_interval=250, verbose=1):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.log_interval = log_interval
        
        # Initialize lists to store metrics
        self.timesteps = []
        self.mean_rewards = []
        self.mean_lengths = []
        self.success_rates = []
        self.mean_distances = []
        self.normalized_distances = []  # Add normalized distances
        self.normalized_rewards = []    # Add normalized rewards
        
        # Initialize episode counter and metrics
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.episode_norm_distances = []  # Add normalized distances
        self.episode_successes = []
        
        # Initialize the last log time
        self.last_log_time = time.time()
    
    def _init_callback(self) -> None:
        # Initialize the timesteps list with 0 and mean_rewards with initial value
        # This ensures we have at least one data point for plotting
        self.timesteps.append(0)
        self.mean_rewards.append(0.0)
        self.mean_lengths.append(0.0)
        self.success_rates.append(0.0)
        self.mean_distances.append(1.0)  # Start at maximum distance (worst case)
        self.normalized_distances.append(1.0)  # Start at 1.0 (worst case)
        self.normalized_rewards.append(0.0)    # Start at 0.0 (worst case)
    
    def _on_step(self) -> bool:
        # Get the current environment
        env = self.training_env.envs[0]
        
        # Check if an episode has ended
        if hasattr(env, 'dones') and any(env.dones):
            # Increment episode counter
            self.episode_count += 1
            
            # Get the episode reward
            episode_reward = env.rewards[0]
            self.episode_rewards.append(episode_reward)
            
            # Get the episode length
            episode_length = env.episode_lengths[0]
            self.episode_lengths.append(episode_length)
            
            # Get the final distance to target (if available)
            if hasattr(env, 'infos') and env.infos[0] is not None:
                if 'distance_cm' in env.infos[0]:
                    distance = env.infos[0]['distance_cm'] / 100.0  # Convert back to meters
                    self.episode_distances.append(distance)
                
                # Get normalized distance if available
                if 'normalized_distance' in env.infos[0]:
                    norm_distance = env.infos[0]['normalized_distance']
                    self.episode_norm_distances.append(norm_distance)
                
                # Check if the episode was successful
                if 'target_reached' in env.infos[0] and env.infos[0]['target_reached']:
                    self.episode_successes.append(1.0)
                else:
                    self.episode_successes.append(0.0)
            
            # Log metrics at the specified interval
            if self.episode_count % self.log_interval == 0:
                # Calculate mean metrics
                mean_reward = np.mean(self.episode_rewards[-self.log_interval:])
                mean_length = np.mean(self.episode_lengths[-self.log_interval:])
                success_rate = np.mean(self.episode_successes[-self.log_interval:])
                
                # Calculate mean distance if available
                mean_distance = 1.0
                if len(self.episode_distances) > 0:
                    mean_distance = np.mean(self.episode_distances[-self.log_interval:])
                
                # Calculate mean normalized distance if available
                mean_norm_distance = 1.0
                if len(self.episode_norm_distances) > 0:
                    mean_norm_distance = np.mean(self.episode_norm_distances[-self.log_interval:])
                
                # Normalize mean reward to 0-1 range
                # Assuming rewards are already normalized in the reward function
                normalized_reward = np.clip(mean_reward, 0.0, 1.0)
                
                # Store metrics for plotting
                self.timesteps.append(self.num_timesteps)
                self.mean_rewards.append(mean_reward)
                self.mean_lengths.append(mean_length / 100.0)  # Normalize to 0-1 range (assuming max 100 steps)
                self.success_rates.append(success_rate)
                self.mean_distances.append(mean_distance)
                self.normalized_distances.append(mean_norm_distance)
                self.normalized_rewards.append(normalized_reward)
                
                # Calculate time elapsed since last log
                current_time = time.time()
                time_elapsed = current_time - self.last_log_time
                self.last_log_time = current_time
                
                # Log metrics
                if self.verbose > 0:
                    print(f"Timestep: {self.num_timesteps}")
                    print(f"Episodes: {self.episode_count}")
                    print(f"Mean reward: {mean_reward:.4f} (normalized: {normalized_reward:.4f})")
                    print(f"Mean episode length: {mean_length:.2f} steps (normalized: {mean_length/100.0:.4f})")
                    print(f"Success rate: {success_rate*100:.2f}%")
                    print(f"Mean distance to target: {mean_distance*100:.2f}cm (normalized: {mean_norm_distance:.4f})")
                    print(f"Time elapsed: {time_elapsed:.2f}s")
                    print("-----------------------------------")
        
        return True
    
    def plot_training_progress(self, save_path):
        """Plot training progress metrics, all normalized to 0-1 range."""
        if len(self.timesteps) <= 1:
            print("Not enough data to plot training progress")
            return
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot normalized rewards
        axs[0, 0].plot(self.timesteps, self.normalized_rewards)
        axs[0, 0].set_title('Normalized Rewards (0-1)')
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Plot success rate
        axs[0, 1].plot(self.timesteps, self.success_rates)
        axs[0, 1].set_title('Success Rate (0-1)')
        axs[0, 1].set_xlabel('Timesteps')
        axs[0, 1].set_ylabel('Success Rate')
        axs[0, 1].grid(True)
        
        # Plot normalized episode lengths
        axs[1, 0].plot(self.timesteps, [length/100.0 for length in self.mean_lengths])
        axs[1, 0].set_title('Normalized Episode Length (0-1)')
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Length')
        axs[1, 0].grid(True)
        
        # Plot normalized distances
        axs[1, 1].plot(self.timesteps, self.normalized_distances)
        axs[1, 1].set_title('Normalized Distance to Target (0-1)')
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Distance')
        axs[1, 1].grid(True)
        
        # Plot all metrics together (normalized)
        axs[2, 0].plot(self.timesteps, self.normalized_rewards, label='Reward')
        axs[2, 0].plot(self.timesteps, self.success_rates, label='Success')
        axs[2, 0].plot(self.timesteps, [length/100.0 for length in self.mean_lengths], label='Length')
        axs[2, 0].plot(self.timesteps, self.normalized_distances, label='Distance')
        axs[2, 0].set_title('All Metrics (Normalized 0-1)')
        axs[2, 0].set_xlabel('Timesteps')
        axs[2, 0].set_ylabel('Value')
        axs[2, 0].legend()
        axs[2, 0].grid(True)
        
        # Plot raw rewards (for reference)
        axs[2, 1].plot(self.timesteps, self.mean_rewards)
        axs[2, 1].set_title('Raw Rewards')
        axs[2, 1].set_xlabel('Timesteps')
        axs[2, 1].set_ylabel('Reward')
        axs[2, 1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# Function to plot training metrics
def plot_training_metrics(model, save_path):
    """
    Plot training metrics from the model's episode info buffer.
    
    Args:
        model: The trained model with episode info buffer
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Extract episode rewards and lengths
        ep_rewards = [ep_info["r"] for ep_info in model.ep_info_buffer]
        ep_lengths = [ep_info["l"] for ep_info in model.ep_info_buffer]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot episode rewards
        ax1.plot(ep_rewards, label='Episode Reward')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(ep_lengths, label='Episode Length', color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length (steps)')
        ax2.legend()
        ax2.grid(True)
        
        # Add a rolling average to both plots
        if len(ep_rewards) > 10:
            window_size = min(10, len(ep_rewards) // 5)
            rewards_avg = np.convolve(ep_rewards, np.ones(window_size)/window_size, mode='valid')
            lengths_avg = np.convolve(ep_lengths, np.ones(window_size)/window_size, mode='valid')
            
            # Pad the beginning of the rolling average to match the original data length
            padding = len(ep_rewards) - len(rewards_avg)
            rewards_avg = np.pad(rewards_avg, (padding, 0), 'edge')
            lengths_avg = np.pad(lengths_avg, (padding, 0), 'edge')
            
            ax1.plot(rewards_avg, label=f'{window_size}-Episode Avg', linestyle='--', color='red')
            ax2.plot(lengths_avg, label=f'{window_size}-Episode Avg', linestyle='--', color='red')
            ax1.legend()
            ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
    except Exception as e:
        print(f"Error generating plot: {e}")

# Define a wrapper class for adding delay to GUI rendering
class DelayedGUIEnv(gym.Wrapper):
    """
    A wrapper that adds a delay after each step to slow down visualization.
    """
    def __init__(self, env, delay=0.01):
        super(DelayedGUIEnv, self).__init__(env)
        self.delay = delay
        
    def step(self, action):
        result = self.env.step(action)
        if self.delay > 0:
            time.sleep(self.delay)
        
        # Return the result as is - don't modify the API format
        return result

def create_envs(num_envs, viz_speed=0.0, parallel_viz=False, eval_env=False, disable_bounds_collision=False):
    """
    Create multiple environments for parallel training.
    
    Args:
        num_envs: Number of environments to create
        viz_speed: Visualization speed (0.0 for no visualization)
        parallel_viz: Whether to use parallel visualization
        eval_env: Whether this is an evaluation environment
        disable_bounds_collision: Ignored. Workspace bounds collision is permanently disabled.
        
    Returns:
        list: List of environments
    """
    # Determine if we should use GUI mode
    gui = viz_speed > 0.0
    
    # Create environments
    envs = []
    
    # Calculate a scaling factor for timeout steps based on number of environments
    # This ensures that as we add more robots, timeouts still occur at roughly
    # the same real-time rate, by reducing the number of steps before timeout
    timeout_scale_factor = 1.0
    if num_envs > 1:
        # Apply a diminishing scale factor as robot count increases
        # Formula: scale = 1.0 / (0.5 + 0.5 * sqrt(num_robots))
        # This gives a more gradual scaling than direct division
        timeout_scale_factor = 1.0 / (0.5 + 0.5 * math.sqrt(num_envs))
    
    # Base timeout steps (before scaling)
    base_timeout_steps = 800 if eval_env else 200  # Base values 
    
    # Apply scaling to get effective timeout steps
    effective_timeout_steps = int(base_timeout_steps * timeout_scale_factor)
    
    # Ensure a minimum reasonable timeout
    effective_timeout_steps = max(effective_timeout_steps, 50)
    
    print(f"Using timeout of {effective_timeout_steps} steps for {num_envs} robots (scale factor: {timeout_scale_factor:.2f})")
    
    # For parallel visualization, we need to create environments with offsets
    if parallel_viz and gui:
        # Calculate offsets for each robot
        # We'll place them in a row along the x-axis
        offsets = []
        for i in range(num_envs):
            # Place robots with 1.6m spacing
            # First robot at -0.8 * (num_envs - 1), last at 0.8 * (num_envs - 1)
            offset = -0.8 * (num_envs - 1) + i * 1.6
            offsets.append(offset)
        
        # Create environments with offsets
        for i in range(num_envs):
            # Set a scaled timeout for environments
            env = RobotPositioningEnv(
                gui=gui,
                gui_delay=0.0,
                clean_viz=(i == 0),  # Only clean visualization for the first environment
                viz_speed=viz_speed,
                verbose=(i == 0),  # Only verbose for the first environment
                parallel_viz=True,
                rank=i,
                offset_x=offsets[i]
                # enable_bounds_collision parameter removed as it's permanently disabled
            )
            
            # Set the scaled timeout steps
            env.timeout_steps = effective_timeout_steps
            
            # Add the environment to the list
            envs.append(env)
            
        # Adjust the camera to show all robots
        adjust_camera_for_robots(envs[0].client_id, num_envs)
    else:
        # Create environments without offsets
        for i in range(num_envs):
            env = RobotPositioningEnv(
                gui=(gui and i == 0),  # Only GUI for the first environment
                gui_delay=0.0,
                clean_viz=True,
                viz_speed=(viz_speed if i == 0 else 0.0),  # Only visualization for the first environment
                verbose=(i == 0),  # Only verbose for the first environment
                rank=i
                # enable_bounds_collision parameter removed as it's permanently disabled
            )
            
            # Set the scaled timeout steps
            env.timeout_steps = effective_timeout_steps
            
            # Wrap with DelayedGUIEnv if visualization is enabled
            if gui and i == 0 and viz_speed > 0.0:
                env = DelayedGUIEnv(env, delay=viz_speed)
            
            # Add the environment to the list
            envs.append(env)
    
    return envs

def create_multiple_robots_in_same_env(num_robots=3, viz_speed=0.1, verbose=False):
    """
    Create multiple robots in the same environment with explicit positioning.
    
    Args:
        num_robots: Number of robots to create
        viz_speed: Visualization speed (delay in seconds)
        verbose: Whether to print verbose output
        
    Returns:
        List of environments
    """
    # Create a shared PyBullet client
    client_id = get_shared_pybullet_client(render=True)
    print(f"Created shared PyBullet client with ID: {client_id}")
    
    # Generate positions for robots dynamically based on the number requested
    # We'll arrange them in a grid pattern with appropriate spacing
    robot_positions = []
    
    # Calculate the grid dimensions based on the number of robots
    # We want a roughly square grid
    grid_size = int(np.ceil(np.sqrt(num_robots)))
    
    # Calculate spacing between robots
    spacing = 1.6  # Distance between robots
    
    # Generate positions in a grid
    for i in range(num_robots):
        # Calculate row and column in the grid
        row = i // grid_size
        col = i % grid_size
        
        # Calculate x and y offsets
        # Center the grid around (0,0)
        x_offset = (col - (grid_size - 1) / 2) * spacing
        y_offset = (row - (grid_size - 1) / 2) * spacing
        
        robot_positions.append((x_offset, y_offset))
    
    # Calculate a scaling factor for timeout steps based on number of robots
    # This ensures that as we add more robots, timeouts still occur at roughly
    # the same real-time rate, by reducing the number of steps before timeout
    timeout_scale_factor = 1.0
    if num_robots > 1:
        # Apply a diminishing scale factor as robot count increases
        # Formula: scale = 1.0 / (0.5 + 0.5 * sqrt(num_robots))
        # This gives a more gradual scaling than direct division
        timeout_scale_factor = 1.0 / (0.5 + 0.5 * math.sqrt(num_robots))
    
    # Base timeout steps (before scaling)
    base_timeout_steps = 200  # Standard training timeout 
    
    # Apply scaling to get effective timeout steps
    effective_timeout_steps = int(base_timeout_steps * timeout_scale_factor)
    
    # Ensure a minimum reasonable timeout
    effective_timeout_steps = max(effective_timeout_steps, 50)
    
    print(f"Using timeout of {effective_timeout_steps} steps for {num_robots} robots (scale factor: {timeout_scale_factor:.2f})")
    
    # Create environments
    envs = []
    for i in range(num_robots):
        # Get position for this robot
        x_offset, y_offset = robot_positions[i]
        
        # Create environment
        env = RobotPositioningEnv(
            gui=True,
            verbose=(i == 0),  # Only the first env should be verbose
            viz_speed=viz_speed,
            parallel_viz=True,
            rank=i,
            offset_x=x_offset
        )
        
        # Set the scaled timeout steps
        env.timeout_steps = effective_timeout_steps
        
        # Apply y offset
        if y_offset != 0.0:
            # Move the robot to its y offset position
            pos, orn = p.getBasePositionAndOrientation(
                env.robot.robot_id,
                physicsClientId=env.robot.client
            )
            p.resetBasePositionAndOrientation(
                env.robot.robot_id,
                [pos[0], y_offset, pos[2]],  # Apply y offset
                orn,
                physicsClientId=env.robot.client
            )
            
            # Update home position to reflect the new position
            env.home_position[1] = y_offset
            if verbose:
                print(f"Robot {i} positioned at [{env.home_position[0]:.2f}, {env.home_position[1]:.2f}, {env.home_position[2]:.2f}]")
        
        # Add the environment to the list
        envs.append(env)
    
    # Adjust the camera to ensure all robots are visible
    adjust_camera_for_robots(client_id, num_robots)
    
    return envs

# Add global variables to track when model weights should be updated
_MODEL_UPDATE_NEEDED = False
_MODEL_UPDATE_ROBOT_RANK = -1  # Store which robot triggered the update
_SHARED_MODEL_VERSION = 0  # Track the shared model version

# Function to set the model update flag
def set_model_update_flag(robot_rank=0):
    """Set the flag to indicate that model weights should be updated."""
    global _MODEL_UPDATE_NEEDED, _MODEL_UPDATE_ROBOT_RANK
    _MODEL_UPDATE_NEEDED = True
    _MODEL_UPDATE_ROBOT_RANK = robot_rank

# Function to check if model update is needed
def is_model_update_needed():
    """Check if model weights should be updated."""
    global _MODEL_UPDATE_NEEDED
    return _MODEL_UPDATE_NEEDED

# Function to get the robot rank that triggered the update
def get_model_update_robot_rank():
    """Get the rank of the robot that triggered the model update."""
    global _MODEL_UPDATE_ROBOT_RANK
    return _MODEL_UPDATE_ROBOT_RANK

# Function to reset the model update flag
def reset_model_update_flag():
    """Reset the flag after model weights have been updated."""
    global _MODEL_UPDATE_NEEDED, _MODEL_UPDATE_ROBOT_RANK
    _MODEL_UPDATE_NEEDED = False
    _MODEL_UPDATE_ROBOT_RANK = -1

# Function to increment the shared model version
def increment_shared_model_version():
    """Increment the shared model version and return the new version."""
    global _SHARED_MODEL_VERSION
    _SHARED_MODEL_VERSION += 1
    return _SHARED_MODEL_VERSION

# Function to get the current shared model version
def get_shared_model_version():
    """Get the current shared model version."""
    global _SHARED_MODEL_VERSION
    return _SHARED_MODEL_VERSION

# Add global variables for target randomization
_TARGET_REACHED_FLAG = False
_TARGET_RANDOMIZATION_TIME = time.time()  # Initialize to current time when module is loaded

# Function to set the target reached flag
def set_target_reached_flag():
    """Set the flag to indicate that a target has been reached and all robots should get new targets."""
    global _TARGET_REACHED_FLAG
    _TARGET_REACHED_FLAG = True

# Function to check if target reached flag is set
def get_target_reached_flag():
    """Check if the target reached flag is set."""
    global _TARGET_REACHED_FLAG
    return _TARGET_REACHED_FLAG

# Function to reset the target reached flag
def reset_target_reached_flag():
    """Reset the flag after all robots have been given new targets."""
    global _TARGET_REACHED_FLAG
    _TARGET_REACHED_FLAG = False

# Function to update the target randomization time
def update_target_randomization_time():
    """Update the global target randomization time."""
    global _TARGET_RANDOMIZATION_TIME
    _TARGET_RANDOMIZATION_TIME = time.time()
    return _TARGET_RANDOMIZATION_TIME

# Function to get the target randomization time
def get_target_randomization_time():
    """Get the global target randomization time."""
    global _TARGET_RANDOMIZATION_TIME
    return _TARGET_RANDOMIZATION_TIME

# Add global variables to track the best-performing robot during timeout
_BEST_TIMEOUT_DISTANCE = float('inf')
_BEST_TIMEOUT_ROBOT_RANK = -1
_BEST_TIMEOUT_REWARD = float('-inf')  # Initialize to negative infinity for reward maximization

# Function to reset the best timeout tracking
def reset_best_timeout_tracking():
    """Reset the global best timeout tracking variables."""
    global _BEST_TIMEOUT_DISTANCE, _BEST_TIMEOUT_ROBOT_RANK, _BEST_TIMEOUT_REWARD
    _BEST_TIMEOUT_DISTANCE = float('inf')
    _BEST_TIMEOUT_ROBOT_RANK = -1
    _BEST_TIMEOUT_REWARD = float('-inf')

# Function to update the best timeout tracking
def update_best_timeout_tracking(distance, robot_rank, total_reward=None):
    """
    Update the global best timeout tracking variables based on the total reward.
    If total_reward is provided, it will be used as the primary metric.
    Otherwise, the distance will be used as before.
    
    Args:
        distance: The distance achieved by the robot
        robot_rank: The rank of the robot
        total_reward: The total reward accumulated by the robot during the episode
        
    Returns:
        bool: True if this robot is now the best, False otherwise
    """
    global _BEST_TIMEOUT_DISTANCE, _BEST_TIMEOUT_ROBOT_RANK, _BEST_TIMEOUT_REWARD
    
    # If total_reward is provided, use it as the primary metric
    if total_reward is not None:
        if total_reward > _BEST_TIMEOUT_REWARD:
            _BEST_TIMEOUT_REWARD = total_reward
            _BEST_TIMEOUT_DISTANCE = distance  # Still track the distance for reference
            _BEST_TIMEOUT_ROBOT_RANK = robot_rank
            return True
    # Otherwise, use distance as before
    elif distance < _BEST_TIMEOUT_DISTANCE:
        _BEST_TIMEOUT_DISTANCE = distance
        _BEST_TIMEOUT_ROBOT_RANK = robot_rank
        return True
    
    return False

# Function to get the best timeout distance
def get_best_timeout_distance():
    """Get the best distance achieved during timeout."""
    global _BEST_TIMEOUT_DISTANCE
    return _BEST_TIMEOUT_DISTANCE

# Function to get the best timeout robot rank
def get_best_timeout_robot_rank():
    """Get the rank of the robot that achieved the best distance during timeout."""
    global _BEST_TIMEOUT_ROBOT_RANK
    return _BEST_TIMEOUT_ROBOT_RANK

# Function to get the best timeout reward
def get_best_timeout_reward():
    """Get the best reward achieved during timeout."""
    global _BEST_TIMEOUT_REWARD
    return _BEST_TIMEOUT_REWARD

# Add a ModelUpdateCallback class to handle model weight updates
class ModelUpdateCallback(BaseCallback):
    """
    Callback for updating the shared model from multiple parallel robots.
    Model updates occur only when a robot's position improved compared to its initial position.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_info_buffer = []
        self.max_episode_info_buffer_size = 3000  # Limit buffer size to prevent memory leaks
        self.skipped_updates = 0  # Track skipped updates due to no improvement
    
    def _on_step(self):
        """Called at each step of training."""
        rewards = []
        distances = []
        improved_envs = []  # Track environments that showed improvement
        
        # Check if any environments have completed episodes
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Get the info for the done environment
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]
                    
                    # Check if this environment's position improved
                    position_improved = info.get("position_improved", False)
                    improvement_amount = info.get("improvement_amount", 0.0)
                    
                    # Store info for logging
                    self.episode_info_buffer.append({
                        'reward': self.locals.get("rewards", [])[i],
                        'distance': info.get("distance_cm", None),
                        'improved': position_improved,
                        'improvement_amount': improvement_amount,
                        'env_id': i
                    })
                    
                    # Limit the buffer size to prevent memory leaks
                    if len(self.episode_info_buffer) > self.max_episode_info_buffer_size:
                        # Remove the oldest items first
                        self.episode_info_buffer = self.episode_info_buffer[-self.max_episode_info_buffer_size:]
                    
                    # Only consider this environment for model update if it showed improvement
                    if position_improved:
                        # This environment improved, so collect its reward and distance
                        rewards.append(self.locals.get("rewards", [])[i])
                        distances.append(info.get("distance_cm", None))
                        improved_envs.append(i)
                    else:
                        if self.verbose > 0:
                            print(f"Env {i}: No position improvement, skipping model update contribution")
        
        # If multiple environments completed episodes and at least one showed improvement, update the model
        if len(rewards) > 0:
            # Calculate average reward and distance for logging
            avg_reward = np.mean(rewards) if rewards else 0
            avg_distance = np.mean([d for d in distances if d is not None]) if distances else None
            
            # Log the model update
            if self.verbose > 0:
                info_str = f"ModelUpdateCallback: Updating shared model with data from {len(improved_envs)} environments"
                if avg_distance is not None:
                    info_str += f", avg_distance: {avg_distance:.2f}cm"
                if avg_reward is not None:
                    info_str += f", avg_reward: {avg_reward:.2f}"
                
                improved_envs_str = ", ".join([str(env_id) for env_id in improved_envs])
                info_str += f", improved envs: [{improved_envs_str}]"
                
                print(info_str)
                
            # Force garbage collection to free memory
            force_garbage_collection()
            
            # Update the shared model version
            increment_shared_model_version()
            
            # Return True to continue training
            return True
        elif len(self.locals.get("dones", [])) > 0 and any(self.locals.get("dones", [])):
            # Episodes completed but no improvements found, so we're skipping the update
            self.skipped_updates += 1
            
            if self.verbose > 0:
                print(f"ModelUpdateCallback: Skipped update ({self.skipped_updates} total) - no robots improved position")
        
        # Return True to continue training
        return True

# Global variables for workspace data
_WORKSPACE_DATA: Optional[Dict[str, Any]] = None
_WORKSPACE_POSITIONS: Optional[np.ndarray] = None
_WORKSPACE_JOINT_CONFIGS: Optional[np.ndarray] = None
_WORKSPACE_MAX_REACH: Optional[float] = None
_WORKSPACE_MIN_REACH: Optional[float] = None
_WORKSPACE_BOUNDS: Optional[Dict[str, List[float]]] = None

def load_workspace_data(workspace_file='robot_workspace.json', verbose=False):
    """
    Load the pre-computed workspace data from a JSON file.
    
    Args:
        workspace_file: Path to the workspace data file
        verbose: Whether to print verbose output
        
    Returns:
        dict: Workspace information
    """
    global _WORKSPACE_DATA, _WORKSPACE_POSITIONS, _WORKSPACE_JOINT_CONFIGS, _WORKSPACE_MAX_REACH, _WORKSPACE_MIN_REACH, _WORKSPACE_BOUNDS
    
    # Check if workspace data is already loaded
    if _WORKSPACE_DATA is not None:
        return _WORKSPACE_DATA
    
    # Check if the workspace file exists
    if not os.path.exists(workspace_file):
        # If not, print a warning and return None
        print(f"WARNING: Workspace file {workspace_file} not found.")
        print("Please run determine_robot_workspace.py first to generate the workspace data.")
        print("Falling back to default workspace parameters.")
        
        # Set default workspace parameters
        _WORKSPACE_MAX_REACH = 0.7  # Default max reach of 70cm
        _WORKSPACE_MIN_REACH = 0.1  # Default min reach of 10cm
        _WORKSPACE_BOUNDS = {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5],
            'z': [0.0, 0.8]
        }
        return None
    
    try:
        # Load the workspace data from the JSON file
        with open(workspace_file, 'r') as f:
            workspace_data = json.load(f)
        
        # Get min_reach or calculate it if not present (for backward compatibility)
        if 'min_reach' not in workspace_data:
            # Calculate minimum reach as 5th percentile of distances
            if 'distances' in workspace_data:
                distances = np.array(workspace_data['distances'])
                min_reach = float(np.percentile(distances, 5))
            else:
                # If distances not available, use a default value
                min_reach = 0.1  # Default 10cm
            workspace_data['min_reach'] = min_reach
        
        if verbose:
            print(f"Loaded workspace data from {workspace_file}")
            print(f"Number of samples: {workspace_data['num_samples']}")
            print(f"Maximum reach: {workspace_data['max_reach']:.3f}m")
            print(f"Minimum reach: {workspace_data['min_reach']:.3f}m")
            print(f"Workspace bounds:")
            bounds = workspace_data['workspace_bounds']
            print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
            print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")
            print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
        
        # Store the workspace data in global variables
        _WORKSPACE_DATA = workspace_data
        _WORKSPACE_POSITIONS = np.array(workspace_data['ee_positions'])
        _WORKSPACE_JOINT_CONFIGS = np.array(workspace_data['joint_configurations'])
        _WORKSPACE_MAX_REACH = workspace_data['max_reach']
        _WORKSPACE_MIN_REACH = workspace_data['min_reach']
        _WORKSPACE_BOUNDS = workspace_data['workspace_bounds']
        
        return workspace_data
    
    except Exception as e:
        print(f"ERROR: Failed to load workspace data: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def find_nearest_position(target_position, max_samples=1000):
    """
    Find the nearest position in the workspace to the given target position,
    and return the corresponding joint configuration.
    
    Args:
        target_position: [x, y, z] position to find the nearest position for
        max_samples: Maximum number of samples to consider (for performance)
        
    Returns:
        tuple: (nearest_position, joint_configuration, distance)
    """
    global _WORKSPACE_POSITIONS, _WORKSPACE_JOINT_CONFIGS
    
    # Check if workspace data is loaded
    if _WORKSPACE_POSITIONS is None or _WORKSPACE_JOINT_CONFIGS is None:
        print("WARNING: Workspace data not loaded. Cannot find nearest position.")
        return None, None, float('inf')
    
    # Convert target position to numpy array
    target_position = np.array(target_position)
    
    # Limit the number of samples to consider for performance
    if len(_WORKSPACE_POSITIONS) > max_samples:
        # Randomly select max_samples indices
        indices = np.random.choice(len(_WORKSPACE_POSITIONS), max_samples, replace=False)
        positions = _WORKSPACE_POSITIONS[indices]
        joint_configs = _WORKSPACE_JOINT_CONFIGS[indices]
    else:
        positions = _WORKSPACE_POSITIONS
        joint_configs = _WORKSPACE_JOINT_CONFIGS
    
    # Calculate distances to all positions
    distances = np.linalg.norm(positions - target_position, axis=1)
    
    # Find the index of the minimum distance
    min_idx = np.argmin(distances)
    
    # Return the nearest position, joint configuration, and distance
    return positions[min_idx], joint_configs[min_idx], distances[min_idx]

# Global variables for model sharing
_SHARED_MODEL: Optional[Any] = None
_MODEL_VERSION = 0

def set_shared_model(model):
    """Set the shared model that all robots will use."""
    global _SHARED_MODEL, _MODEL_VERSION
    _SHARED_MODEL = model
    _MODEL_VERSION += 1
    return _MODEL_VERSION

def get_shared_model():
    """Get the shared model that all robots are using."""
    global _SHARED_MODEL
    return _SHARED_MODEL

def get_model_version():
    """Get the current model version."""
    global _MODEL_VERSION
    return _MODEL_VERSION

# Modify the main function to use the ModelUpdateCallback
def main():
    """Main function for training the robot."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Print hardware information
    print_hardware_info(args)
    
    # Print a message about bounds collision being disabled
    print(f"\nWorkspace bounds collision is permanently disabled in this version")
    
    # Load workspace data
    load_workspace_data(verbose=True)
    
    # Create environments
    if args.parallel_viz and args.viz_speed > 0.0:
        print(f"Using {args.parallel} robots in the same environment with visualization")
        envs = create_multiple_robots_in_same_env(
            num_robots=args.parallel,
            viz_speed=args.viz_speed,
            verbose=args.verbose
        )
    else:
        # For visualization without parallel-viz, use a single environment
        if args.viz_speed > 0.0 and not args.parallel_viz:
            print("Using a single environment with visualization")
            args.parallel = 1
        elif args.viz_speed > 0.0 and args.parallel_viz:
            print(f"Using {args.parallel} environments with parallel visualization")
            
        # Create environments using the standard method
        envs = create_envs(
            num_envs=args.parallel,
            viz_speed=args.viz_speed,
            parallel_viz=args.parallel_viz,
            eval_env=args.eval_only
            # disable_bounds_collision parameter removed as it's permanently disabled
        )
    
    # Wrap environments with VecEnv
    if args.parallel > 1:
        # Use DummyVecEnv for multiple environments to avoid subprocess issues
        vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    else:
        # Use DummyVecEnv for a single environment
        vec_env = DummyVecEnv([lambda: envs[0]])
    
    # Create the model
    if args.load:
        print(f"Loading model from {args.load}")
        model = SAC.load(args.load, env=vec_env)
    else:
        print("Creating new SAC model")
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            verbose=1,
            device="cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        )
    
    # Initialize the shared model
    set_shared_model(model)
    
    # Evaluation mode
    if args.eval_only:
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
    
    # Create the model update callback
    model_update_callback = ModelUpdateCallback(verbose=1)
    
    # Force target repositioning at the beginning of training
    # This ensures all robots get new targets right from the start
    set_target_reached_flag()
    update_target_randomization_time()
    if args.verbose:
        print("Forcing initial target repositioning for all robots")
    
    # Training mode
    print(f"Training for {args.steps} steps")
    print("When any robot reaches the target with 1mm precision:")
    print("1. Model weights will be updated from that successful run")
    print("2. All robots will reset and start a new run with the updated model")
    print("\nIf no robot reaches the target within the time limit:")
    print("1. The robot that achieved the best accuracy will be used for model updates")
    print("2. Model weights will only be updated if the best distance is within 5cm of the target")
    print("3. All robots will reset and start a new run with the updated model")
    
    print("\nCollision detection:")
    print("1. Ground collisions are detected and penalized")
    print("2. End effector self-collisions are detected and penalized")
    print("3. Other robot parts can clip through each other (simplified physical model)")
    
    model.learn(total_timesteps=args.steps, callback=model_update_callback)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./models/{args.algorithm}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/final_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training metrics
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
        plot_dir = "./plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = f"{plot_dir}/{args.algorithm}_{timestamp}_metrics.png"
        plot_training_metrics(model, plot_path)
        print(f"Training metrics saved to {plot_path}")
        
        # Also save a copy in the model directory
        progress_plot_path = f"{model_dir}/training_progress.png"
        plot_training_metrics(model, progress_plot_path)
        print(f"Training progress plot saved to {progress_plot_path}")
    else:
        print("Not enough data to plot")
    
    # Close the environment
    vec_env.close()

def parse_args():
    """Parse command line arguments."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train a robot to position itself in a 3D environment')
    parser.add_argument('--steps', type=int, default=999999999, help='Number of steps to train for (default: 999,999,999)')
    parser.add_argument('--load', type=str, default=None, help='Load a trained model from file')
    parser.add_argument('--save-dir', type=str, default='./models', help='Directory to save models to')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the model, no training')
    parser.add_argument('--gui', action='store_true', help='Enable GUI visualization')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel environments to use (default: 2)')
    parser.add_argument('--parallel-viz', action='store_true', default=True, help='Enable parallel visualization (multiple robots in view), default: True')
    parser.add_argument('--viz-speed', type=float, default=0.1, help='Control visualization speed (delay in seconds, higher = slower, default: 0.1)')
    parser.add_argument('--cuda', dest='use_cuda', action='store_true', help='Use CUDA for training if available')
    parser.add_argument('--cpu', dest='use_cuda', action='store_false', help='Use CPU for training even if CUDA is available')
    parser.add_argument('--workspace-size', type=float, default=0.7, help='Size of the workspace for target positions')
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac'], help='RL algorithm to use (default: sac)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for updates (default: 256, balanced for learning)')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size for SAC (default: 1M, larger for better stability)')
    parser.add_argument('--train-freq', type=int, default=1, help='Update frequency for SAC (default: 1, update every step)')
    parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps per update for SAC (default: 1, balanced learning)')
    parser.add_argument('--eval-freq', type=int, default=2000, help='Evaluation frequency in steps')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--save-freq', type=int, default=5000, help='Model saving frequency in steps')
    parser.add_argument('--clean-viz', action='store_true', help='Enable clean visualization with zoomed-in view of the robot')
    parser.add_argument('--eval-viz', action='store_true', help='Enable visualization for evaluation only')
    parser.add_argument('--high-lr', action='store_true', help='Use a higher learning rate (1e-3) for faster learning')
    parser.add_argument('--optimize-training', action='store_true', help='Enable optimized training settings for faster learning', default=True)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--exploration', type=float, default=0.2, help='Initial exploration rate (higher values = more exploration)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    # Remove the --disable-bounds-collision argument since bounds collision is now permanently disabled
    args = parser.parse_args()
    return args

def print_hardware_info(args):
    """Print information about the hardware being used."""
    # Print CPU/GPU information
    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU for training")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU for training")
    
    # Print learning rate
    print(f"Using learning rate: {args.learning_rate}")
    
    # Print visualization information
    if args.viz_speed > 0.0:
        print(f"Using visualization speed: {args.viz_speed}s delay (slow motion)")
    
    # Print PyTorch threads
    print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")
    
    # Print system memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except ImportError:
        print("psutil not installed, skipping memory information")

def limit_ep_info_buffer(model, max_size=100):
    """
    Limit the size of the episode info buffer to prevent memory growth.
    
    Args:
        model: The model with an ep_info_buffer attribute
        max_size: Maximum number of episodes to keep in the buffer
    """
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > max_size:
        # Keep only the most recent episodes
        model.ep_info_buffer = model.ep_info_buffer[-max_size:]

import gc  # Add garbage collection module

def force_garbage_collection():
    """
    Force garbage collection to free up memory.
    Call this periodically during training to prevent memory leaks.
    """
    gc.collect()
    
    # Try to get memory info if psutil is available
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        print(f"Memory usage after GC: {memory_mb:.1f} MB")
    except ImportError:
        pass  # Silently ignore if psutil is not available

if __name__ == "__main__":
    main()
