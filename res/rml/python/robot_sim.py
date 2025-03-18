# robot_sim.py
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
from pybullet_utils import get_pybullet_client, configure_visualization

class FANUCRobotEnv:
    def __init__(self, render=True, verbose=False):
        # Store verbose flag
        self.verbose = verbose
        
        # Store render mode
        self.render_mode = render
        
        # Connect to the physics server using the shared client function
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
            
        # Initial configuration
        self.reset()
        
    def _load_robot(self):
        # Load the URDF for the FANUC LR Mate 200iC
        
        # Check for the URDF file in various possible locations
        possible_paths = [
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
                    print(f"Loading FANUC LR Mate 200iC URDF from: {urdf_path}")
                return p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        
        # If we couldn't find the URDF, print a warning and fall back to a simple robot
        print("WARNING: Could not find FANUC LR Mate 200iC URDF file. Falling back to default robot.")
        print("Current working directory:", os.getcwd())
        print("Searched paths:", possible_paths)
        
        # Fallback to a simple robot for testing
        return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
    
    def reset(self):
        # Reset to home position
        home_position = [0, 0, 0, 0, 0, 0]  # All joints at 0 position
        for i, pos in enumerate(home_position):
            p.resetJointState(self.robot_id, i, pos)
        
        # Get current state
        state = self._get_state()
        return state
        
    def step(self, action):
        # Apply action (joint positions) to the robot
        # action should be a list of 6 target joint positions
        
        # No enforcement of joint limits - allow full range of motion
        # Only prevent extreme values that would cause simulation instability
        for i, a in enumerate(action):
            # Only apply extremely loose limits to prevent simulation crashes
            if a < -10 * np.pi:  # Prevent more than 10 full rotations in negative direction
                action[i] = -10 * np.pi
            elif a > 10 * np.pi:  # Prevent more than 10 full rotations in positive direction
                action[i] = 10 * np.pi
        
        # Set joint positions
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=range(self.dof),
            controlMode=p.POSITION_CONTROL,
            targetPositions=action,
            forces=[self.max_force] * self.dof,
            positionGains=[self.position_gain] * self.dof,
            velocityGains=[self.velocity_gain] * self.dof
        )
        
        # Step simulation
        p.stepSimulation()
        
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