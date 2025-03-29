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
from src.utils.pybullet_utils import get_pybullet_client, configure_visualization, determine_reachable_workspace, get_shared_pybullet_client

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
        
        # Add a small safety margin for joint limits
        safety_margin = 0.002  # 0.002 radians (about 0.1 degrees)
        
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
                # Apply strict joint limits from the URDF with additional safety margin
                positions = list(positions)  # Make a copy to avoid modifying input
                for i, pos in enumerate(positions):
                    if i in self.joint_limits:
                        limit_low, limit_high = self.joint_limits[i]
                        # Apply safety margin to limits
                        safe_low = limit_low + safety_margin
                        safe_high = limit_high - safety_margin
                        
                        if pos < safe_low:
                            positions[i] = safe_low
                            if self.verbose:
                                print(f"WARNING: Joint {i} position {pos:.6f} below safe limit {safe_low:.6f}, clamping to safe limit")
                        elif pos > safe_high:
                            positions[i] = safe_high
                            if self.verbose:
                                print(f"WARNING: Joint {i} position {pos:.6f} above safe limit {safe_high:.6f}, clamping to safe limit")
                
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
            # Enforce joint limits strictly from the URDF with additional safety margin
            limited_action = list(action)  # Create a copy to modify
            
            for i, pos in enumerate(limited_action):
                if i in self.joint_limits:
                    limit_low, limit_high = self.joint_limits[i]
                    # Apply safety margin to limits
                    safe_low = limit_low + safety_margin
                    safe_high = limit_high - safety_margin
                    
                    if pos < safe_low:
                        limited_action[i] = safe_low
                        if self.verbose:
                            print(f"WARNING: Joint {i} position {pos:.6f} below safe limit {safe_low:.6f}, clamping to safe limit")
                    elif pos > safe_high:
                        limited_action[i] = safe_high
                        if self.verbose:
                            print(f"WARNING: Joint {i} position {pos:.6f} above safe limit {safe_high:.6f}, clamping to safe limit")
            
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
    Environment for robot positioning tasks with improved reachability.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, gui=False, max_episode_steps=150, verbose=False, viz_speed=0.0, training_mode=True):
        super().__init__()
        
        # Store parameters
        self.gui = gui
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self.viz_speed = viz_speed
        self.training_mode = training_mode  # Track if in training or evaluation mode
        self.is_evaluating = not training_mode  # For compatibility with older code
        
        # Import function here to avoid circular imports
        from src.utils.pybullet_utils import get_shared_pybullet_client
        
        # Initialize PyBullet client
        self.client_id = get_shared_pybullet_client(gui=gui)
        
        # Create the robot environment
        self.robot = FANUCRobotEnv(render=gui, verbose=verbose, client=self.client_id)
        
        # Get robot's degrees of freedom
        self.dof = self.robot.dof
        
        # Initialize home position (robot's shoulder/base)
        self.home_position = np.array([0.0, 0.0, 0.0])
        
        # Setup successful target parameters
        self.accuracy_threshold = 0.015  # 15mm accuracy
        self.timeout_steps = max_episode_steps
        
        # Curriculum learning parameters
        self.curriculum_level = 0
        self.curriculum_max_levels = 5
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
        self.max_reach = max_reach
        
        # Storage for reachable positions and corresponding joint configurations
        self.reachable_positions = []
        self.reachable_joint_configs = []
        
        # Distance-categorized positions for curriculum learning
        # We'll divide the workspace into difficulty zones based on distance from home
        self.easy_positions = []      # Positions close to home
        self.medium_positions = []    # Positions at medium distance
        self.hard_positions = []      # Positions near the edge of workspace
        
        # Collect reachable positions and categorize them
        self._collect_reachable_positions()
        
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
        max_position = np.ones(3) * (max_reach * 2.0)  # Position (meters)
        max_target = np.ones(3) * (max_reach * 2.0)  # Target position (meters)
        max_distance = np.array([max_reach * 2.0])  # Distance (meters)
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
    
    def _collect_reachable_positions(self, num_positions=300):
        """
        Collect reachable positions by sampling random joint configurations.
        Categorize positions into easy, medium, and hard based on distance from home.
        """
        if self.verbose:
            print(f"Collecting {num_positions} reachable positions for improved target sampling...")
        
        # Make sure we have determined the workspace
        if not hasattr(self, 'workspace_bounds'):
            self.determine_reachable_workspace()
        
        # Collect positions if not already collected
        if not self.reachable_positions:
            home_pos = self.home_position
            collected_positions = []
            
            # Sample a large number of random joint configurations
            for _ in range(num_positions * 2):  # Sample more than needed to ensure we have enough
                if len(collected_positions) >= num_positions:
                    break
                    
                # Sample random joint positions within limits
                joint_positions = []
                for i in range(self.dof):
                    lower, upper = self.robot.joint_limits[i]
                    # Add a small safety margin to avoid joint limits
                    margin = 0.01  # Larger margin for exploration
                    safe_lower = lower + margin
                    safe_upper = upper - margin
                    joint_positions.append(np.random.uniform(safe_lower, safe_upper))
                
                # Set the joints and get end effector position
                self.robot.step((joint_positions, None))
                end_effector_pos = self.robot._get_state()[12:15]
                
                # Check if the position is valid (not in collision and inside workspace bounds)
                if self._is_position_reachable(end_effector_pos):
                    collected_positions.append((end_effector_pos, np.linalg.norm(end_effector_pos - home_pos)))
            
            # Sort positions by distance from home
            collected_positions.sort(key=lambda x: x[1])
            
            # Take only the requested number of positions
            collected_positions = collected_positions[:num_positions]
            
            # Store all reachable positions
            self.reachable_positions = [pos for pos, _ in collected_positions]
            
            # Categorize positions into easy, medium, and hard based on distance
            n_positions = len(collected_positions)
            easy_end = n_positions // 3
            medium_end = 2 * n_positions // 3
            
            self.easy_positions = [pos for pos, _ in collected_positions[:easy_end]]
            self.medium_positions = [pos for pos, _ in collected_positions[easy_end:medium_end]]
            self.hard_positions = [pos for pos, _ in collected_positions[medium_end:]]
            
            if self.verbose:
                print(f"Collected {len(self.reachable_positions)} reachable positions")
                print(f"- Easy positions: {len(self.easy_positions)}")
                print(f"- Medium positions: {len(self.medium_positions)}")
                print(f"- Hard positions: {len(self.hard_positions)}")
        
        # Reset the robot to home position
        self.robot.reset()
    
    def _is_position_reachable(self, position):
        """Check if a position is reachable by the robot."""
        # Check if position is within workspace bounds
        for i in range(3):
            if position[i] < self.workspace_bounds[i][0] or position[i] > self.workspace_bounds[i][1]:
                return False
        
        # If we have IK enabled, we can also verify using IK
        if hasattr(self, 'robot'):
            # Save current state
            current_joints = [self.robot._get_state()[i*2] for i in range(self.dof)]
            
            # Try to compute IK
            ik_solution = p.calculateInverseKinematics(
                self.robot.robot_id, 
                self.robot.dof-1, 
                position
            )
            
            # Reset to original state
            for i in range(self.dof):
                p.resetJointState(self.robot.robot_id, i, current_joints[i])
            
            # Check if IK solution is valid (all values are finite)
            if all(np.isfinite(ik_solution)):
                # Check if solution respects joint limits
                for i, angle in enumerate(ik_solution[:self.dof]):
                    lower, upper = self.robot.joint_limits[i]
                    if angle < lower or angle > upper:
                        return False
                return True
            return False
        
        # Default to True if we can't verify with IK
        return True
    
    def _sample_target(self):
        """
        Sample a target position from the collected reachable positions based on curriculum level.
        Add small random noise to avoid memorization.
        """
        if not self.reachable_positions:
            # Fallback to uniform sampling if no reachable positions collected
            x = np.random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[0][1])
            y = np.random.uniform(self.workspace_bounds[1][0], self.workspace_bounds[1][1])
            z = np.random.uniform(self.workspace_bounds[2][0], self.workspace_bounds[2][1])
            return np.array([x, y, z])
        
        # Select pool of positions based on curriculum level
        position_pools = []
        
        # Level 0: Only easy positions
        if self.curriculum_level == 0:
            position_pools = [self.easy_positions]
        # Level 1: Mix of easy and medium positions
        elif self.curriculum_level == 1:
            position_pools = [self.easy_positions, self.medium_positions]
        # Level 2: Mostly medium positions
        elif self.curriculum_level == 2:
            position_pools = [self.medium_positions]
        # Level 3: Mix of medium and hard positions
        elif self.curriculum_level == 3:
            position_pools = [self.medium_positions, self.hard_positions]
        # Level 4: Only hard positions
        else:
            position_pools = [self.hard_positions]
        
        # Flatten the pools and sample a random position
        all_positions = []
        for pool in position_pools:
            all_positions.extend(pool)
        
        if not all_positions:
            # Fallback to all positions if the selected pools are empty
            all_positions = self.reachable_positions
        
        position = all_positions[np.random.randint(0, len(all_positions))]
        
        # Add small random noise (but ensure it stays within workspace bounds)
        noise_scale = 0.03  # 3cm noise
        for _ in range(10):  # Try up to 10 times to find a valid position
            noise = np.random.uniform(-noise_scale, noise_scale, size=3)
            noisy_position = position + noise
            
            # Verify the position is still reachable
            if self._is_position_reachable(noisy_position):
                return noisy_position
        
        # If all noise attempts failed, return the original position
        return position
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment and sample a new target based on the current curriculum level.
        """
        # First reset using parent method (resets robot to home position)
        super().reset(seed=seed, options=options)
        
        # Sample a new target position that respects the current curriculum level
        max_attempts = 10
        for _ in range(max_attempts):
            self.target_position = self._sample_target()
            if self._is_position_reachable(self.target_position):
                break
        
        # Update visualization if enabled
        if self.gui:
            self.update_target_visualization()
        
        # Print status
        if self.verbose:
            print(f"Robot {self.robot_id} received a new target at {self.target_position}")
            print(f"Robot {self.robot_id}: Initial distance to target: {self.get_distance_to_target()*100:.2f}cm")
            print(f"Current curriculum level: {self.curriculum_level}/{self.curriculum_max_levels-1}")
        
        # Return the observation
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        """
        Execute one time step within the environment.
        Enhanced with safety checks and reward shaping.
        """
        # Add exploration noise during training (not during evaluation)
        if not self.is_evaluating:
            noise_scale = 0.0005  # Reduced from 0.001 to be more conservative
            action_with_noise = action + np.random.normal(0, noise_scale, size=action.shape)
        else:
            action_with_noise = action
        
        # Apply action with joint limits enforced by the action space
        super().step(action_with_noise)
        
        # Add safety margin to make sure we're not too close to the limits
        safety_margin = 0.001  # in radians (about 0.06 degrees)
        current_joint_positions = self.robot._get_state()[:self.dof*2:2]
        
        # Ensure positions are within safety limits
        for i in range(self.dof):
            lower, upper = self.robot.joint_limits[i]
            safe_lower = lower + safety_margin
            safe_upper = upper - safety_margin
            
            if current_joint_positions[i] < safe_lower:
                current_joint_positions[i] = safe_lower
                if self.verbose:
                    print(f"WARNING: Joint {i} position {current_joint_positions[i]} below safe limit {safe_lower}, clamping to safe limit")
            elif current_joint_positions[i] > safe_upper:
                current_joint_positions[i] = safe_upper
                if self.verbose:
                    print(f"WARNING: Joint {i} position {current_joint_positions[i]} above safe limit {safe_upper}, clamping to safe limit")
        
        # Set the joints to the safe positions
        self.robot.step((current_joint_positions, None))
        
        # Get observation, reward, and done flag
        terminated = False
        distance = self.get_distance_to_target()
        
        # Check if we reached the target (success)
        success = distance < self.accuracy_threshold
        
        # Calculate the reward (enhanced for better learning)
        reward = self._calculate_reward(distance, action, success)
        
        # Check if episode is done
        if success:
            terminated = True
            self.consecutive_successful_episodes += 1
            
            # Check if we should advance the curriculum
            if self.consecutive_successful_episodes >= 5 and self.curriculum_level < self.curriculum_max_levels - 1:
                self.curriculum_level += 1
                if self.verbose:
                    print(f"Advancing to curriculum level {self.curriculum_level}/{self.curriculum_max_levels-1}")
            
            # Reset consecutive successes if failure
            self.consecutive_successful_episodes = 0
        
        # Increment step counter
        self.steps += 1
        
        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()
        info["success"] = success
        info["distance"] = distance
        
        return obs, reward, terminated, False, info
    
    def _calculate_reward(self, distance, action, success):
        """
        Calculate the reward based on distance to target, action magnitude, and success.
        Uses reward shaping to guide the learning process.
        """
        # Distance component: negative distance scaled
        distance_reward = -10.0 * distance
        
        # Action magnitude penalty to encourage smooth movements
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # Success bonus if we reached the target
        success_reward = 100.0 if success else 0.0
        
        # Progressive reward: give more reward as we get closer to the target
        progress_reward = 0.0
        previous_distance = self.previous_distance if hasattr(self, 'previous_distance') else distance
        distance_improvement = previous_distance - distance
        
        # Store current distance for next step
        self.previous_distance = distance
        
        # Give reward for getting closer to the target, penalize for moving away
        progress_scale = 5.0
        progress_reward = progress_scale * distance_improvement
        
        # Combine all reward components
        reward = distance_reward + action_penalty + success_reward + progress_reward
        
        return reward
    
    def _get_obs(self):
        """
        Get the current observation state.
        """
        # Get current state from robot environment
        state = self.robot._get_state()
        
        # Extract joint positions
        joint_positions = state[:self.dof*2:2]  # Extract joint positions
        
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
    
    def _get_info(self):
        """
        Get the current info state.
        """
        info = {
            'curriculum_level': self.curriculum_level
        }
        return info
    
    def get_distance_to_target(self):
        """
        Calculate the current distance from the end effector to the target.
        
        Returns:
            float: Distance in meters
        """
        if self.target_position is None:
            return float('inf')
            
        # Get end effector position
        state = self.robot._get_state()
        ee_position = state[-7:-4]  # Position is in the last 7 elements (position + orientation)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(ee_position - self.target_position)
        
        return distance
    
    def update_target_visualization(self):
        """
        Update the target visualization in the environment.
        """
        if self.target_visual_id:
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.client_id)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not remove previous target visual: {e}")
            self.target_visual_id = None
        
        if self.target_position is not None:
            try:
                self.target_visual_id = visualize_target(self.target_position, self.client_id)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not visualize target: {e}")

# For backward compatibility
RobotPositioningEnv = RobotPositioningRevampedEnv
