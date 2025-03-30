import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import math

class FanucEnv(gym.Env):
    """Custom Gymnasium environment for the FANUC LRMate 200iD robot arm using PyBullet.

    Args:
        render_mode (str, optional): The rendering mode ('human' or None). Defaults to None.
        max_episode_steps (int, optional): Maximum steps per episode. Defaults to 1000.
        target_accuracy (float, optional): Target accuracy in metres. Defaults to 0.02.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_episode_steps=1000, target_accuracy=0.02):
        super().__init__()

        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._target_accuracy = target_accuracy
        self._step_counter = 0

        # --- PyBullet Setup ---
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Improve rendering (optional)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0,0,0.5])
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100)

        # Load plane and robot
        p.loadURDF("plane.urdf")
        # Construct the path to the URDF file relative to this script
        urdf_file_path = os.path.join(os.path.dirname(__file__), "Fanuc", "urdf", "Fanuc.urdf")
        # Define the mesh path replacement for PyBullet
        mesh_path = os.path.join(os.path.dirname(__file__), "Fanuc")

        # Load URDF with adjusted mesh paths (this requires manual path replacement logic usually,
        # but PyBullet's loadURDF might handle 'package://' if the path exists relative to the URDF.
        # If meshes don't load, manual URDF parsing/editing or using flags might be needed.
        # For simplicity, assume p.setAdditionalSearchPath might help, or that the relative structure works.
        # Adding the Fanuc directory to the search path:
        p.setAdditionalSearchPath(mesh_path)

        # Load the robot - using a fixed base
        self.robot_id = p.loadURDF(
            urdf_file_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            # Flags might be needed if mesh loading fails, e.g., p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )

        # --- Robot Configuration ---
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        # Approximate joint limits (radians) for LRMate 200iD (J1 to J5 based on URDF)
        # Source: FANUC LRMate 200iD Datasheet / General Robotics Knowledge
        # Note: URDF uses continuous joints, so limits are enforced here.
        self.joint_limits_lower = np.array([-2.96, -1.74, -2.37, -3.31, -2.18]) # J1, J2, J3, J4, J5
        self.joint_limits_upper = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18]) # J1, J2, J3, J4, J5
        # Velocity limits (radians/sec) - Estimate reasonable values
        self.velocity_limits = np.array([6.0, 6.0, 6.0, 8.0, 8.0]) * 0.5 # Scaled down

        link_name_to_index = {p.getJointInfo(self.robot_id, i)[12].decode('UTF-8'): i for i in range(self.num_joints)}
        self.end_effector_link_index = -1

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('UTF-8')
            joint_type = info[2]

            # Identify controllable joints (only revolute/continuous are represented as JOINT_REVOLUTE by PyBullet)
            if joint_type == p.JOINT_REVOLUTE:
                # Assuming the 5 controllable joints appear sequentially in the URDF
                if len(self.joint_indices) < 5:
                    self.joint_indices.append(i)
                    self.joint_names.append(joint_name)

            # Find end effector link index (assuming 'Part6' is the EE link)
            link_name = info[12].decode('UTF-8')
            if link_name == 'Part6':
                self.end_effector_link_index = i # Note: This gives the *joint* index connected *to* Part6.
                                                # Pybullet uses link index for getLinkState, which is often joint_index.

        if self.end_effector_link_index == -1:
            print("Warning: Could not find link 'Part6'. Using last joint index as end effector.")
            self.end_effector_link_index = self.joint_indices[-1] if self.joint_indices else self.num_joints - 1

        if len(self.joint_indices) != 5:
             raise ValueError(f"Expected 5 controllable joints, but found {len(self.joint_indices)}. URDF might be different than expected.")

        self.num_controllable_joints = len(self.joint_indices)

        # --- Action Space (Normalised Joint Velocity Control) ---
        # Action is delta velocity for each joint, scaled by velocity_limits
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_controllable_joints,), dtype=np.float32)

        # --- Observation Space ---
        # [joint_positions (5), joint_velocities (5), relative_target_pos (3)] -> 13 dimensions
        obs_low = np.concatenate([
            self.joint_limits_lower,
            -self.velocity_limits * 2, # Allow higher velocity readings than limits
            np.array([-np.inf] * 3, dtype=np.float32) # Relative position can be anything
        ])
        obs_high = np.concatenate([
            self.joint_limits_upper,
            self.velocity_limits * 2, # Allow higher velocity readings than limits
            np.array([np.inf] * 3, dtype=np.float32) # Relative position can be anything
        ])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- Target ---
        self.target_position = np.zeros(3, dtype=np.float32)
        self.initial_joint_positions = np.zeros(self.num_controllable_joints) # Home position (all zeros)
        self.target_visual_shape_id = -1 # Initialise visual shape ID

        print(f"FanucEnv initialised:")
        print(f"  - Joints: {self.joint_names}")
        print(f"  - End Effector Link Index: {self.end_effector_link_index}")
        print(f"  - Action Space Shape: {self.action_space.shape}")
        print(f"  - Observation Space Shape: {self.observation_space.shape}")


    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index, computeForwardKinematics=True)
        ee_position = np.array(ee_state[0], dtype=np.float32) # World position of the link's CoM

        relative_position = self.target_position - ee_position

        return np.concatenate([joint_positions, joint_velocities, relative_position])

    def _get_info(self):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        ee_position = np.array(ee_state[0], dtype=np.float32)
        distance = np.linalg.norm(self.target_position - ee_position)
        return {"distance": distance, "target_position": self.target_position.copy()}

    def _get_random_target_pos(self):
        # Generate a random target position within an approximate reachable sphere
        # LRMate 200iD reach is approx 0.7m. Sample within a sphere of radius 0.6m centered slightly above base.
        radius = 0.2 + np.random.rand() * 0.4 # Sample between 0.2m and 0.6m radius
        theta = np.random.rand() * 2 * math.pi # Azimuthal angle
        phi = (np.random.rand() * 0.6 + 0.1) * math.pi # Polar angle (restricted to upper hemisphere mostly)

        # Spherical to Cartesian conversion
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi) + 0.1 # Offset center slightly upwards

        # Ensure target is not too close to the base (avoids singularity/collision issues)
        min_dist_from_base = 0.15
        if np.linalg.norm([x, y, z]) < min_dist_from_base:
            # If too close, regenerate (simple approach)
            return self._get_random_target_pos()

        return np.array([x, y, z], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_counter = 0

        # Reset joint positions to initial/home state
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, targetValue=self.initial_joint_positions[i], targetVelocity=0)
            # Ensure motor control is reset/disabled initially if needed
            p.setJointMotorControl2(self.robot_id, joint_index, p.VELOCITY_CONTROL, force=0)


        # Generate a new target
        self.target_position = self._get_random_target_pos()

        # Remove previous target visual marker if it exists
        if hasattr(self, 'target_visual_shape_id') and self.target_visual_shape_id >= 0:
             p.removeBody(self.target_visual_shape_id)


        # Visualise the target (optional, useful for debugging)
        if self.render_mode == 'human' or True: # Always create for info calculation, remove if performance hit
            target_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.8])
            target_collision_shape = -1 # No collision for target marker
            self.target_visual_shape_id = p.createMultiBody(baseMass=0,
                                                            baseCollisionShapeIndex=target_collision_shape,
                                                            baseVisualShapeIndex=target_visual_shape,
                                                            basePosition=self.target_position)


        observation = self._get_obs()
        info = self._get_info()

        # print(f"Resetting Env. New Target: {self.target_position}") # Debug

        return observation, info

    def step(self, action):
        # --- Apply Action and Enforce Limits ---
        # Clip action just in case
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_velocities = action * self.velocity_limits

        current_positions = np.array([p.getJointState(self.robot_id, i)[0] for i in self.joint_indices])

        # Simple joint limit enforcement: If a joint is at its limit, prevent motion further in that direction.
        for i, joint_index in enumerate(self.joint_indices):
            pos = current_positions[i]
            vel = target_velocities[i]

            if ((pos <= self.joint_limits_lower[i] and vel < 0) or
                (pos >= self.joint_limits_upper[i] and vel > 0)):
                target_velocities[i] = 0 # Stop motion if trying to exceed limit

            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[i],
                force=100 # Max force - adjust as needed
            )

        # --- Step Simulation ---
        p.stepSimulation()
        self._step_counter += 1

        # --- Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()
        distance = info['distance']

        # --- Calculate Reward ---
        # Dense reward: negative distance to target
        # Encourage faster convergence: reward = -distance**2? or 1/(distance + epsilon)?
        # Simple negative distance is common.
        reward = -distance

        # Bonus for reaching the target accuracy
        success_bonus = 100.0
        terminated = False
        if distance < self._target_accuracy:
            reward += success_bonus
            terminated = True
            # print(f"Target Reached! Steps: {self._step_counter}, Distance: {distance:.4f}") # Debug

        # --- Check Truncation ---
        truncated = False
        if self._step_counter >= self._max_episode_steps:
            truncated = True
            # Optional: Add penalty for truncation? reward -= 10

        # Optional: Penalty for excessive joint velocity?
        # reward -= 0.01 * np.mean(np.abs(observation[self.num_controllable_joints:2*self.num_controllable_joints]))

        # Optional: Check for collisions (more complex)
        # contacts = p.getContactPoints(bodyA=self.robot_id) # Check self-collision / collision with plane
        # if contacts:
        #    reward -= 50 # Collision penalty
        #    terminated = True # End episode on collision

        # Render if in human mode
        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            # PyBullet handles rendering internally when connected with p.GUI
            # We might adjust camera or add debug lines here if needed
            # For example, draw line from EE to target
            # ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            # ee_pos = ee_state[0]
            # p.addUserDebugLine(ee_pos, self.target_position, lineColorRGB=[0, 1, 0], lifeTime=0.05)
            pass # p.GUI handles rendering loop


    def close(self):
        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
            self.physics_client = -1

# Example usage (optional, for testing the environment)
if __name__ == '__main__':
    # Example of running the environment
    env = FanucEnv(render_mode='human')
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    episodes = 3
    for ep in range(episodes):
        print(f"--- Episode {ep+1} ---")
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        while not done:
            action = env.action_space.sample() # Sample random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            # env.render() # Already called in step if render_mode='human'
            # Add a small delay for visualisation
            # import time
            # time.sleep(1./60.)

            if done:
                print(f"Episode finished after {step} steps. Terminated={terminated}, Truncated={truncated}. Total Reward: {total_reward:.2f}. Final Distance: {info['distance']:.4f}")
                break

    env.close()
    print("Environment closed.") 