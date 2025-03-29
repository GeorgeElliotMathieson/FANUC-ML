#!/usr/bin/env python3
# robot_sim.py - FANUC robot simulation environment
import pybullet as p  # type: ignore
import pybullet_data  # type: ignore
import time
import numpy as np
import os
import sys
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore

# Import from the utils module
from src.utils.pybullet_utils import get_pybullet_client, configure_visualization, determine_reachable_workspace

# Import necessary components for environment
from src.core.env.action_spaces import JointLimitedBox
from src.core.env.visualization import visualize_target
from src.core.env.rewards import calculate_combined_reward

class FANUCRobotEnv:
    """
    Simplified FANUC robot environment that supports loading URDFs and simulating the robot.
    This is the core robot interface used by various learning environments.
    """
    def __init__(self, client=None, render=True, verbose=False, max_force=100, dof=None):
        self.verbose = verbose
        self.max_force = max_force
        # dof will be set after loading the robot
        self.dof = dof
        
        # Control gains for position and velocity control
        self.position_gain = 0.5
        self.velocity_gain = 1.0
        
        # Initialize joint limits dictionary
        self.joint_limits = {}
        
        # Connect to PyBullet
        self.render_mode = render
        if client is None:
            # No client provided, create our own
            if render:
                self.client = p.connect(p.GUI)
                if self.verbose:
                    print(f"Connected to PyBullet in GUI mode with client ID: {self.client}")
            else:
                self.client = p.connect(p.DIRECT)
                if self.verbose:
                    print(f"Connected to PyBullet in DIRECT mode with client ID: {self.client}")
        else:
            # Use the provided client
            self.client = client
            if self.verbose:
                print(f"Using provided PyBullet client with ID: {self.client}")
        
        # Configure PyBullet
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        # Load the robot
        self.robot_id = self._load_robot()
        if self.verbose:
            print(f"Loaded robot with ID: {self.robot_id}")
        
        # Initialize robot to home position
        self.reset()
    
    def _load_robot(self):
        # Load the FANUC robot URDF
        
        # Get project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(current_dir, '../..'))
        
        # Use only the fanuc.urdf with meshes
        urdf_path = os.path.join(project_dir, "robots", "urdf", "fanuc.urdf")
        
        if os.path.exists(urdf_path):
            if self.verbose:
                print(f"Loading FANUC robot URDF from: {urdf_path}")
            robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
            
            # After loading the robot, read the joint limits directly from the URDF
            # This ensures we use the exact limits from the URDF file
            if robot_id is not None:
                # Initialize an empty joint_limits dictionary
                self.joint_limits = {}
                
                # Get the number of joints
                num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
                
                # Count revolute joints to determine DOF
                revolute_joints = []
                
                # Extract joint limits for each joint
                for i in range(num_joints):
                    joint_info = p.getJointInfo(robot_id, i, physicsClientId=self.client)
                    
                    # Only add revolute joints (type 0) to the joint_limits dictionary
                    if joint_info[2] == p.JOINT_REVOLUTE:
                        joint_index = joint_info[0]
                        lower_limit = joint_info[8]
                        upper_limit = joint_info[9]
                        
                        # Store the limits in the dictionary
                        self.joint_limits[joint_index] = [lower_limit, upper_limit]
                        revolute_joints.append(joint_index)
                        
                        if self.verbose:
                            print(f"Joint {joint_index} ({joint_info[1].decode('utf-8')}): Limits [{lower_limit:.6f}, {upper_limit:.6f}]")
                
                # Set DOF based on the number of revolute joints if not explicitly specified
                if self.dof is None:
                    self.dof = len(revolute_joints)
                    if self.verbose:
                        print(f"Detected {self.dof} revolute joints (DOF)")
            
                return robot_id
        
        # If we couldn't find the URDF, print a warning and fall back to a simple robot
        print("WARNING: Could not find FANUC robot URDF file at:", urdf_path)
        print("Current working directory:", os.getcwd())
        
        # Fallback to a simple robot for testing
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        
        # Set default DOF for fallback robot if not explicitly specified
        if self.dof is None:
            # Count revolute joints in fallback robot
            num_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
            revolute_count = 0
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i, physicsClientId=self.client)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    revolute_count += 1
            self.dof = revolute_count
            
        return robot_id
    
    def reset(self):
        # Reset to home position
        home_position = [0] * self.dof  # All joints at 0 position
        for i, pos in enumerate(home_position):
            if i < self.dof:
                p.resetJointState(self.robot_id, i, pos)
        
        # Get current state
        state = self._get_state()
        return state
        
    def step(self, action):
        # Apply action (joint positions) to the robot
        # action can be:
        # 1. A list/array of 5 target joint positions
        # 2. A tuple of (positions, velocities) where each is a list of 5 values
        
        # Check if action is in tuple format (positions, velocities)
        if isinstance(action, tuple) and len(action) == 2:
            positions, velocities = action
            
            # If positions is None, only apply velocity control
            if positions is None and velocities is not None:
                # Apply velocity control
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot_id,
                    jointIndices=range(self.dof),
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=velocities,
                    forces=[self.max_force] * self.dof,
                    velocityGains=[self.velocity_gain] * self.dof,
                    physicsClientId=self.client
                )
            else:
                # Apply both position and velocity control
                # Apply strict joint limits from the URDF
                for i, pos in enumerate(positions):
                    if i in self.joint_limits:
                        limit_low, limit_high = self.joint_limits[i]
                        if pos < limit_low:
                            positions[i] = limit_low
                            if self.verbose:
                                print(f"WARNING: Joint {i} position {pos:.4f} below limit {limit_low:.4f}, clamping to limit")
                        elif pos > limit_high:
                            positions[i] = limit_high
                            if self.verbose:
                                print(f"WARNING: Joint {i} position {pos:.4f} above limit {limit_high:.4f}, clamping to limit")
                
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot_id,
                    jointIndices=range(self.dof),
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=positions,
                    targetVelocities=velocities,
                    forces=[self.max_force] * self.dof,
                    positionGains=[self.position_gain] * self.dof,
                    velocityGains=[self.velocity_gain] * self.dof,
                    physicsClientId=self.client
                )
        else:
            # Original format - just positions
            # Enforce joint limits strictly from the URDF
            limited_action = list(action)  # Create a copy to modify
            
            for i, pos in enumerate(limited_action):
                if i in self.joint_limits:
                    limit_low, limit_high = self.joint_limits[i]
                    if pos < limit_low:
                        limited_action[i] = limit_low
                        if self.verbose:
                            print(f"WARNING: Joint {i} position {pos:.4f} below limit {limit_low:.4f}, clamping to limit")
                    elif pos > limit_high:
                        limited_action[i] = limit_high
                        if self.verbose:
                            print(f"WARNING: Joint {i} position {pos:.4f} above limit {limit_high:.4f}, clamping to limit")
            
            # Set joint positions with enforced limits
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=range(self.dof),
                controlMode=p.POSITION_CONTROL,
                targetPositions=limited_action,
                forces=[self.max_force] * self.dof,
                positionGains=[self.position_gain] * self.dof,
                velocityGains=[self.velocity_gain] * self.dof,
                physicsClientId=self.client
            )
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.client)
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward (placeholder - customize based on your task)
        reward = 0
        
        # Check if done (placeholder - customize based on your task)
        done = False
        
        return next_state, reward, done, {}
    
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
    
    def render(self):
        """
        Render the environment.
        If we're in DIRECT mode, try to switch to GUI mode.
        """
        # If we're already in GUI mode, nothing to do
        if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
            # Force the GUI to update
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=self.client)
            return
        
        # If we're in DIRECT mode, try to switch to GUI
        try:
            # Disconnect from the current client
            p.disconnect(self.client)
            
            # Connect to a new client in GUI mode
            self.client = p.connect(p.GUI)
            self.render_mode = True
            
            if self.verbose:
                print(f"Switched to GUI mode with client ID: {self.client}")
            
            # Reconfigure the environment
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            self.robot_id = self._load_robot()
            self.reset()
        except Exception as e:
            print(f"Warning: Failed to switch to GUI mode: {e}")
            print("Falling back to DIRECT mode")
            self.client = p.connect(p.DIRECT)
            self.render_mode = False
    
    def close(self):
        """
        Clean up resources.
        """
        if self.client is not None:
            p.disconnect(self.client)

# DEPRECATED: Use the function from src.utils.pybullet_utils instead
# This function is kept here only for backward compatibility
def get_shared_pybullet_client(render=True):
    """
    Get or create a shared PyBullet client.
    This is a deprecated function. Use src.utils.pybullet_utils.get_shared_pybullet_client instead.
    """
    from src.utils.pybullet_utils import get_shared_pybullet_client as utils_get_shared_client
    import warnings
    warnings.warn(
        "This get_shared_pybullet_client function in robot_sim.py is deprecated. "
        "Please use src.utils.pybullet_utils.get_shared_pybullet_client instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return utils_get_shared_client(render=render)

# Test the environment
if __name__ == "__main__":
    env = FANUCRobotEnv(render=True)
    
    # Move joints to different positions
    for _ in range(10):
        # Random joint positions within limits
        action = []
        for joint in range(env.dof):
            if joint in env.joint_limits:
                low, high = env.joint_limits[joint]
                action.append(np.random.uniform(low, high))
            else:
                action.append(0)
        
        # Step the environment
        state, reward, done, _ = env.step(action)
        time.sleep(0.1)  # For visualization
    
    env.close()

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
        
        # Import function here to avoid circular imports
        from src.utils.pybullet_utils import get_shared_pybullet_client
        
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
        self.accuracy_threshold = 0.015  # 15mm accuracy
        self.timeout_steps = 150  # Episode length
        
        # Curriculum learning parameters
        self.curriculum_level = 0
        self.max_target_distance = 0.3  # Start with easier targets
        self.target_expansion_increment = 0.05  # Increment target distance by 5cm at a time
        self.consecutive_successful_episodes = 0
        self.last_episode_successful = False
        
        # Import function here to avoid circular imports
        from src.utils.pybullet_utils import determine_reachable_workspace
        
        # Determine reachable workspace
        max_reach, workspace_bounds = determine_reachable_workspace(
            self.robot, 
            n_samples=1000, 
            visualize=verbose
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
        
        # Define action space: Use the JointLimitedBox space 
        self.action_space = JointLimitedBox(self.robot, shape=(self.dof,), dtype=np.float32)
        
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
        # Define sampling strategy to use
        sampling_strategy = "uniform"
        
        # Get workspace bounds for sampling
        ws_bounds = self.workspace_bounds
        
        # Sample target position based on strategy
        if sampling_strategy == "uniform":
            # Sample uniformly within workspace bounds
            x = np.random.uniform(ws_bounds['x'][0], ws_bounds['x'][1])
            y = np.random.uniform(ws_bounds['y'][0], ws_bounds['y'][1])
            z = np.random.uniform(ws_bounds['z'][0], ws_bounds['z'][1])
            target_position = np.array([x, y, z])
        else:
            # Default to uniform sampling
            x = np.random.uniform(ws_bounds['x'][0], ws_bounds['x'][1])
            y = np.random.uniform(ws_bounds['y'][0], ws_bounds['y'][1])
            z = np.random.uniform(ws_bounds['z'][0], ws_bounds['z'][1])
            target_position = np.array([x, y, z])
        
        # Apply offset if needed
        if np.linalg.norm(self.robot_offset) > 0.0:
            target_position += self.robot_offset
        
        return target_position
    
    def _get_observation(self):
        """
        Get the current observation state.
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
        
        # Create current observation
        observation = np.concatenate([
            normalized_joint_positions,   # Current joint positions (normalized 0-1)
            ee_position,                  # Current end effector position
            self.target_position,         # Target position
            [distance_to_target],         # Distance to target
            direction_to_target,          # Direction to target
            self.previous_action,         # Previous action
        ])
        
        # Add history
        for obs in self.observation_history:
            observation = np.concatenate([observation, obs[:self.dof+3]])  # Add joint positions and ee position
        
        # Update history (current becomes history)
        current_basic_observation = np.concatenate([
            normalized_joint_positions,   # Current joint positions (normalized 0-1)
            ee_position,                  # Current end effector position
        ])
        self.observation_history.pop(0)  # Remove oldest
        self.observation_history.append(current_basic_observation)  # Add current
        
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
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not remove previous target visual: {e}")
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
        ee_position = state[12:15]
        
        # Check if visualization should be slowed down
        if self.viz_speed > 0.0:
            time.sleep(self.viz_speed)
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(ee_position - self.target_position)
        
        # Check for success condition
        success = distance_to_target <= self.accuracy_threshold
        
        # Track best distance achieved in this episode
        if distance_to_target < self.best_distance_in_episode:
            self.best_distance_in_episode = distance_to_target
            self.best_position_in_episode = ee_position.copy()
        
        # Calculate reward based on distance, action, and other factors
        reward, reward_info = calculate_combined_reward(
            distance=distance_to_target,
            previous_distance=self.previous_distance,
            initial_distance=self.initial_distance_to_target,
            action=action,
            previous_action=self.previous_action,
            joint_positions=joint_positions,
            joint_limits=self.robot.joint_limits,
            steps=self.steps,
            timeout_steps=self.timeout_steps,
            best_distance=self.best_distance_in_episode,
            accuracy_threshold=self.accuracy_threshold,
            positions_within_limits=positions_within_limits
        )
        
        # Store current distance for next reward calculation
        self.previous_distance = distance_to_target
        
        # Add to total reward
        self.total_reward_in_episode += reward
        
        # Check for done condition (success or timeout)
        done = success or self.steps >= self.timeout_steps
        
        # Record whether this episode was successful
        self.last_episode_successful = success
        
        # Update curriculum if needed
        if done:
            if success:
                self.consecutive_successful_episodes += 1
                # Increase curriculum level after consistent success
                if self.consecutive_successful_episodes >= 5:
                    self.curriculum_level += 1
                    self.consecutive_successful_episodes = 0
            else:
                self.consecutive_successful_episodes = 0
        
        # Get observation
        observation = self._get_observation()
        
        # Create info dict with useful data for logging
        info = {
            'distance': distance_to_target,
            'success': success,
            'best_distance': self.best_distance_in_episode,
            'reward_info': reward_info,
            'episode_length': self.steps,
            'joint_positions': joint_positions,
            'positions_within_limits': positions_within_limits,
            'curriculum_level': self.curriculum_level
        }
        
        return observation, reward, done, info
    
    def close(self):
        """Close the environment and clean up resources."""
        # Disconnect from PyBullet if we created our own client
        if hasattr(self, 'client_id') and self.client_id is not None:
            # Don't disconnect for shared clients
            pass

# For backward compatibility
RobotPositioningEnv = RobotPositioningRevampedEnv
