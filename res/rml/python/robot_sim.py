# robot_sim.py
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import sys

# Add the project root directory to sys.path to help imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from the local pybullet_utils module
from res.rml.python.pybullet_utils import get_pybullet_client, configure_visualization

class FANUCRobotEnv:
    def __init__(self, render=True, verbose=False, client=None):
        # Store verbose flag
        self.verbose = verbose
        
        # Store render mode
        self.render_mode = render
        
        # Connect to the physics server using the shared client function or use the provided client
        if client is not None:
            self.client = client
        else:
            self.client = get_pybullet_client(render=render)
            
        if self.verbose:
            print(f"Connected to PyBullet physics server with client ID: {self.client}")
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure visualization if in GUI mode
        if render and p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
            configure_visualization(self.client, clean_viz=True)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Robot parameters from the documentation
        self.dof = 5  # 5 degrees of freedom (we removed joint6/tool0)
        self.max_force = 100  # Maximum force for joint motors
        self.position_gain = 0.3
        self.velocity_gain = 1.0
        
        # Load the robot URDF (you'll need to create this based on the manual specs)
        # For now, we'll use a placeholder
        self.robot_id = self._load_robot()
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        
        # Joint limits are now loaded directly from the URDF file in _load_robot method
        
        # Initial configuration
        self.reset()
        
    def _load_robot(self):
        # Load the FANUC robot URDF
        
        # Add new path to the updated FANUC robot model
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        fanuc_urdf_path = os.path.join(workspace_dir, "fanuc_robot", "urdf", "fanuc.urdf")
        
        # Check for the URDF file in various possible locations, prioritizing the new model
        possible_paths = [
            fanuc_urdf_path,                             # New 5-axis FANUC robot model
            "fanuc_lrmate_200ic.urdf",                  # Current directory
            "res/fanuc_lrmate_200ic.urdf",              # res directory
            "../res/fanuc_lrmate_200ic.urdf",           # One level up
            "../../res/fanuc_lrmate_200ic.urdf",        # Two levels up
            os.path.join(os.path.dirname(__file__), "../../res/fanuc_lrmate_200ic.urdf")  # Relative to this file
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
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # Configure visualization
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1, physicsClientId=self.client)
            
            # Reload the environment
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
            
            # Reload the robot using the _load_robot method to ensure the correct URDF is used
            self.robot_id = self._load_robot()
            
            # Reset to the current state
            self.reset()
            
        except Exception as e:
            print(f"Warning: Could not switch to GUI mode: {e}")
    
    def close(self):
        try:
            # Check if we're still connected before disconnecting
            if p.isConnected(self.client):
                p.disconnect(self.client)
        except Exception as e:
            print(f"Warning: Error closing PyBullet connection: {e}")

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