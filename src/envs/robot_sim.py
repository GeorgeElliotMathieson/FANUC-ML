#!/usr/bin/env python3
# robot_sim.py - FANUC robot simulation environment
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import sys

# Import from the utils module
from src.utils.pybullet_utils import get_pybullet_client, configure_visualization

class FANUCRobotEnv:
    """
    Simplified FANUC robot environment that supports loading URDFs and simulating the robot.
    This is the core robot interface used by various learning environments.
    """
    def __init__(self, client=None, render=True, verbose=False, max_force=100, dof=5):
        self.verbose = verbose
        self.max_force = max_force
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
        
        # Check for the URDF file in various possible locations, prioritizing the standardized structure
        possible_paths = [
            # Standard locations
            os.path.join(project_dir, "robots", "urdf", "fanuc_lrmate_200ic.urdf"),
            os.path.join(project_dir, "robots", "urdf", "fanuc.urdf"),
            os.path.join(project_dir, "robots", "fanuc", "urdf", "fanuc.urdf"),
            
            # Legacy fallbacks for backwards compatibility
            os.path.join(os.path.dirname(__file__), "../../robots/urdf/fanuc_lrmate_200ic.urdf"),
        ]
        
        # Try each path
        for urdf_path in possible_paths:
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
                            
                            if self.verbose:
                                print(f"Joint {joint_index} ({joint_info[1].decode('utf-8')}): Limits [{lower_limit:.6f}, {upper_limit:.6f}]")
                
                return robot_id
        
        # If we couldn't find the URDF, print a warning and fall back to a simple robot
        print("WARNING: Could not find FANUC robot URDF file. Falling back to default robot.")
        print("Current working directory:", os.getcwd())
        print("Searched paths:", possible_paths)
        
        # Fallback to a simple robot for testing
        return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
    
    def reset(self):
        # Reset to home position
        home_position = [0, 0, 0, 0, 0]  # All joints at 0 position
        for i, pos in enumerate(home_position):
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

# Get a shared PyBullet client
def get_shared_pybullet_client(render=True):
    """
    Get or create a shared PyBullet client.
    This ensures we have a single PyBullet instance across the entire application.
    """
    # Check if PyBullet is already connected
    if p.isConnected():
        # Get existing client ID
        client_id = p.getConnectionInfo()['connectionId']
        print(f"Using existing PyBullet client with ID: {client_id}")
        return client_id
    
    # No existing connection, create a new one
    if render:
        client_id = p.connect(p.GUI)
        print(f"Connected to PyBullet in GUI mode with client ID: {client_id}")
    else:
        client_id = p.connect(p.DIRECT)
        print(f"Connected to PyBullet in DIRECT mode with client ID: {client_id}")
    
    # Configure PyBullet
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    
    return client_id

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