# robot_sim.py
import pybullet as p
import pybullet_data
import time
import numpy as np
import os

class FANUCRobotEnv:
    def __init__(self, render=True):
        # Connect to the physics server
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
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
            0: [-170, 170],  # J1 axis (in degrees)
            1: [-60, 75],    # J2 axis
            2: [-70, 50],    # J3 axis
            3: [-170, 170],  # J4 axis
            4: [-110, 110],  # J5 axis
            5: [-360, 360]   # J6 axis
        }
        
        # Convert to radians
        for joint, limits in self.joint_limits.items():
            self.joint_limits[joint] = [np.deg2rad(limits[0]), np.deg2rad(limits[1])]
            
        # Initial configuration
        self.reset()
        
    def _load_robot(self):
        # In a real implementation, you would load the URDF for the FANUC LR Mate 200iC
        # For this example, we'll use a simple URDF as a placeholder
        # You'll need to create a proper URDF based on the manual specs
        
        # Check if we have a URDF file for the robot (you'll need to create this)
        urdf_path = "fanuc_lrmate_200ic.urdf"
        if os.path.exists(urdf_path):
            return p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        else:
            # Fallback to a simple robot for testing
            return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    
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
        
        # Ensure action is within joint limits
        for i, a in enumerate(action):
            if i in self.joint_limits:
                limit_low, limit_high = self.joint_limits[i]
                action[i] = np.clip(a, limit_low, limit_high)
        
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
        # PyBullet renders automatically in GUI mode
        pass
    
    def close(self):
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