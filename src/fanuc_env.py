import gymnasium as gym
import numpy as np
import pybullet as p # type: ignore
import pybullet_data # type: ignore
import os
import math
import json # Import json
import random # Import random for obstacle placement
import logging # Import logging
import collections # Import collections for deque

# --- Configure Logging (Root Logger) --- 
# Basic config for use when the env is run standalone or imported
# More sophisticated setup could involve named loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Get a logger for this module
logger = logging.getLogger(__name__)

# Define config filename relative to project root (one level up from src/)
WORKSPACE_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), '..', "workspace_config.json")

class FanucEnv(gym.Env):
    """Custom Gymnasium environment for the FANUC LRMate 200iD robot arm using PyBullet.

    Args:
        render_mode (str, optional): The rendering mode ('human' or None). Defaults to None.
        max_episode_steps (int, optional): Maximum steps per episode. Defaults to 1000.
        target_accuracy (float, optional): Target accuracy in metres. Defaults to 0.02.
        angle_bonus_factor (float, optional): Factor for base rotation reward. Defaults to 5.0.
        start_with_obstacles (bool, optional): Whether to place obstacles at the start. Defaults to True.
        force_outer_radius (bool, optional): If True, forces target generation to the outer edge (80-100% of max radius). Used for specific testing scenarios. Defaults to False.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_episode_steps=1000, target_accuracy=0.02, angle_bonus_factor=5.0, start_with_obstacles=True, force_outer_radius=False):
        super().__init__()

        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._target_accuracy = target_accuracy
        self._step_counter = 0
        self.force_outer_radius = force_outer_radius # Store the new flag

        # --- Load Workspace Config or Use Defaults ---
        min_reach_default = 0.02
        # Update default based on check_workspace.py result (approx)
        max_reach_default = 1.26 # Updated from 1.2
        try:
            if os.path.exists(WORKSPACE_CONFIG_FILENAME):
                with open(WORKSPACE_CONFIG_FILENAME, 'r') as f:
                    workspace_config = json.load(f)
                # Use loaded values, falling back to default if a key is missing
                loaded_min_reach = workspace_config.get('min_reach', min_reach_default)
                loaded_max_reach = workspace_config.get('max_reach', max_reach_default)
                # Use logger
                logger.info(f"Loaded workspace config: Min Reach={loaded_min_reach:.4f}, Max Reach={loaded_max_reach:.4f}")
                # Remove safety factor - target generation offset removed
                # safety_factor = 0.98
                self.min_base_radius = loaded_min_reach
                # Apply 90% safety factor to max reach
                safety_factor = 0.90
                self.max_target_radius = loaded_max_reach * safety_factor
                # Use logger
                logger.info(f"Applied safety factor ({safety_factor*100}%). Using Max Target Radius: {self.max_target_radius:.4f}")
            else:
                # Use logger
                logger.warning(f"{WORKSPACE_CONFIG_FILENAME} not found. Using default workspace radii: Min={min_reach_default}, Max={max_reach_default}")
                self.min_base_radius = min_reach_default
                # Apply 90% safety factor to default max reach
                safety_factor = 0.90
                self.max_target_radius = max_reach_default * safety_factor
                # Use logger
                logger.info(f"Applied safety factor ({safety_factor*100}%). Using Max Target Radius: {self.max_target_radius:.4f}")
        except (IOError, json.JSONDecodeError) as e:
            # Use logger
            logger.warning(f"Warning: Error loading {WORKSPACE_CONFIG_FILENAME}: {e}. Using default workspace radii: Min={min_reach_default}, Max={max_reach_default}")
            self.min_base_radius = min_reach_default
            # Apply 90% safety factor to default max reach
            safety_factor = 0.90
            self.max_target_radius = max_reach_default * safety_factor
            # Use logger
            logger.info(f"Applied safety factor ({safety_factor*100}%). Using Max Target Radius: {self.max_target_radius:.4f}")

        # Calculate minimum target radius (e.g., 10x min base reach)
        self.min_target_radius_multiplier = 10.0
        self.min_target_radius = self.min_base_radius * self.min_target_radius_multiplier
        logger.info(f"Calculated Min Target Radius: {self.min_target_radius:.4f} (Base: {self.min_base_radius:.4f} * {self.min_target_radius_multiplier}) ")

        # --- Curriculum Learning Parameters (Decreasing Minimum Radius) ---
        self.fixed_max_target_radius = self.max_target_radius # Max radius remains constant
        self.final_min_target_radius = self.min_target_radius # Store the ultimate goal minimum radius
        # Start with the minimum target radius equal to the maximum, forcing outer sampling initially
        self.initial_min_target_radius = self.fixed_max_target_radius
        self.current_min_target_radius = self.initial_min_target_radius # Current operational minimum radius

        # Parameters for update logic
        self.success_rate_window_size = 20 # Check success rate over this many episodes
        self.success_rate_threshold = 0.75 # Decrease difficulty (min radius) if success rate exceeds this
        self.radius_decrease_step = 0.05 # How much to decrease min_radius by (in meters)
        self.episode_results = collections.deque(maxlen=self.success_rate_window_size)

        # Base rotation reward shaping
        self.previous_base_angle_error = None
        self.angle_bonus_factor = angle_bonus_factor # Factor for base rotation reward
        # self.max_target_radius = 0.6 # Now loaded or defaulted above
        # self.min_base_radius = 0.15 # Now loaded or defaulted above

        # --- Obstacle Setup ---
        self.num_obstacles = 1 # Start with one obstacle
        self.obstacle_radius = 0.05 # Example radius
        self.obstacle_ids = []
        self.obstacle_positions = []
        self.obstacle_collision_penalty = -100.0 # Penalty for hitting an obstacle
        # Define safe zones for placing obstacles (avoid base and initial target area)
        self.obstacle_safe_dist_from_base = 0.3
        self.obstacle_safe_dist_from_target = 0.2
        self.start_with_obstacles = start_with_obstacles # Store flag

        # --- PyBullet Setup ---
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # Enable wireframe visualisation (often shows collision shapes)
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
        self.plane_id = p.loadURDF("plane.urdf") # Store plane ID

        # Construct the paths relative to this script's location (src/)
        script_dir = os.path.dirname(__file__)
        project_root = os.path.join(script_dir, '..') # Go one level up to project root
        # Path to Fanuc model relative to project root
        fanuc_model_dir = os.path.join(project_root, "Fanuc")
        urdf_file_path = os.path.join(fanuc_model_dir, "urdf", "Fanuc.urdf")
        mesh_path = fanuc_model_dir # Main directory for meshes

        # Load URDF with adjusted mesh paths
        # Add the Fanuc directory (relative to project root) to the search path:
        p.setAdditionalSearchPath(mesh_path)

        # Load the robot - using a fixed base
        self.robot_id = p.loadURDF(
            urdf_file_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
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
            # Use logger
            logger.warning("Warning: Could not find link 'Part6'. Using last joint index as end effector.")
            # Fallback: use the last joint index. Check URDF if this happens.
            self.end_effector_link_index = p.getNumJoints(self.robot_id) - 1

        if len(self.joint_indices) != 5:
             raise ValueError(f"Expected 5 controllable joints, but found {len(self.joint_indices)}. URDF might be different than expected.")

        self.num_controllable_joints = len(self.joint_indices)

        # --- Action Space (Normalised Joint Velocity Control) ---
        # Action is delta velocity for each joint, scaled by velocity_limits
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_controllable_joints,), dtype=np.float32)

        # --- Observation Space ---
        # [joint_pos(5), joint_vel(5), rel_target(3), norm_limits(5), rel_obstacle(3)] -> 21 dimensions
        obs_low = np.concatenate([
            self.joint_limits_lower,
            -self.velocity_limits * 2, # Allow higher velocity readings than limits
            np.array([-np.inf] * 3, dtype=np.float32), # Relative position to target
            np.array([-1.0] * self.num_controllable_joints, dtype=np.float32), # Normalised limit proximity
            np.array([-np.inf] * 3, dtype=np.float32) # Relative position to nearest obstacle
        ])
        obs_high = np.concatenate([
            self.joint_limits_upper,
            self.velocity_limits * 2, # Allow higher velocity readings than limits
            np.array([np.inf] * 3, dtype=np.float32), # Relative position to target
            np.array([1.0] * self.num_controllable_joints, dtype=np.float32), # Normalised limit proximity
            np.array([np.inf] * 3, dtype=np.float32) # Relative position to nearest obstacle
        ])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- Target ---
        self.target_position = np.zeros(3, dtype=np.float32)
        self.initial_joint_positions = np.zeros(self.num_controllable_joints) # Home position (all zeros)
        self.target_visual_shape_id = -1 # Initialise visual shape ID

        # --- Parent Link Identification (for collision filtering) ---
        self.parent_link_index = -1
        try:
            ee_joint_info = p.getJointInfo(self.robot_id, self.end_effector_link_index)
            # PyBullet's getJointInfo returns parent link index at index 16
            self.parent_link_index = ee_joint_info[16]
        except Exception as e:
            # Use logger
            logger.warning(f"Warning: Could not get parent link index for end-effector. Error: {e}")

        # Use logger for multi-line info
        init_log_message = (
            f"FanucEnv initialised:\n"
            f"  - Joints: {self.joint_names}\n"
            f"  - End Effector Link Index: {self.end_effector_link_index}\n"
            f"  - Action Space Shape: {self.action_space.shape}\n"
            f"  - Observation Space Shape: {self.observation_space.shape}\n"
            # Update curriculum log message
            f"  - Curriculum (Dec Min Radius): Fixed MaxR={self.fixed_max_target_radius:.3f}, Target MinR={self.final_min_target_radius:.3f}, Initial MinR={self.initial_min_target_radius:.3f}, Window={self.success_rate_window_size}, Threshold={self.success_rate_threshold:.2f}, Step={self.radius_decrease_step}\n"
            f"  - Angle Bonus Factor: {self.angle_bonus_factor}\n"
            f"  - Start with Obstacles: {self.start_with_obstacles}"
        )
        logger.info(init_log_message)


    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        # Calculate normalised joint positions relative to limits (-1 to 1)
        joint_ranges = self.joint_limits_upper - self.joint_limits_lower
        # Prevent division by zero if a joint has no range (shouldn't happen here)
        joint_ranges[joint_ranges == 0] = 1e-6
        normalised_joint_positions = 2 * (joint_positions - self.joint_limits_lower) / joint_ranges - 1.0
        # Clip just in case, due to float precision
        normalised_joint_positions = np.clip(normalised_joint_positions, -1.0, 1.0).astype(np.float32)

        # Use link origin [4] instead of CoM [0] for EE position
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index, computeForwardKinematics=True)
        ee_position = np.array(ee_state[4], dtype=np.float32) # World position of the link origin

        relative_position = self.target_position - ee_position

        # Calculate relative position to the nearest obstacle (currently only one)
        if self.obstacle_positions:
            # Assuming only one obstacle for now
            relative_obstacle_pos = self.obstacle_positions[0] - ee_position
        else:
            # Return a default value if no obstacles are present (e.g., large distance)
            # Using infinity might be problematic, use a large number or zero vector?
            # Let's use a zero vector indicating no obstacle info available
            relative_obstacle_pos = np.zeros(3, dtype=np.float32)

        return np.concatenate([
            joint_positions,
            joint_velocities,
            relative_position,
            normalised_joint_positions, # Add normalised limit proximity
            relative_obstacle_pos      # Add relative obstacle position
        ])

    def _get_info(self):
        # Use link origin [4] instead of CoM [0] for EE position
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        ee_position = np.array(ee_state[4], dtype=np.float32) # World position of the link origin
        distance = np.linalg.norm(self.target_position - ee_position)
        # Update info dict - remove curriculum stage/advancement info
        return {
            "distance": distance,
            "target_position": self.target_position.copy(),
            "collision": False,
            "success": False,
            "current_min_target_radius": self.current_min_target_radius, # Use updated key name
            "obstacle_collision": False # Add flag for obstacle collision
        }

    def _get_random_target_pos(self):
        # Determine radius range based on continuous curriculum OR forced outer edge
        if self.force_outer_radius:
            # Force sampling near the maximum possible radius (80-100%)
            min_radius = self.fixed_max_target_radius * 0.80 # Use fixed max
            max_radius = self.fixed_max_target_radius # Use fixed max
            min_radius = min(min_radius, max_radius - 1e-6)
            logger.info("[Test Mode] Forcing target generation to outer edge.")
        else:
            # Normal curriculum-based sampling
            min_radius = self.current_min_target_radius # Use the CURRENT minimum radius
            max_radius = self.fixed_max_target_radius # Use the FIXED maximum radius

        # Ensure max_radius is not less than min_radius (can happen if current min is very large)
        max_radius = max(max_radius, min_radius + 1e-6)

        # Ensure min_radius respects the absolute minimum calculated (final_min_target_radius)
        min_radius = max(min_radius, self.final_min_target_radius)
        # Ensure min_radius does not exceed max_radius after clamping
        min_radius = min(min_radius, max_radius - 1e-6)

        # Generate a random target position within the calculated radius range
        radius = min_radius + np.random.rand() * (max_radius - min_radius)
        theta = np.random.rand() * 2 * math.pi # Azimuthal angle
        phi = (np.random.rand() * 0.6 + 0.1) * math.pi # Polar angle (restricted to upper hemisphere mostly)

        # Spherical to Cartesian conversion (Centered at Origin 0,0,0)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi) # REMOVED + 0.1 offset

        # Ensure target is above a minimum ground clearance
        min_z_height = 0.05
        if z < min_z_height:
            # If too low, regenerate
            return self._get_random_target_pos()

        # --- Detailed Debug Print for Visual Test --- 
        if self.render_mode == 'human':
            # Update debug print to show current min radius
            print(f"[Env Debug] CurrentMinR={self.current_min_target_radius:.3f}, "
                  f"Bounds=[{min_radius:.3f}-{max_radius:.3f}], "
                  f"Sampled R={radius:.3f}, Phi={phi:.3f}, Theta={theta:.3f}, "
                  f"Target=[{x:.3f}, {y:.3f}, {z:.3f}]")

        # print(f"[Stage {self.current_curriculum_stage}] Target Radius: {radius:.3f} -> Pos: ({x:.3f}, {y:.3f}, {z:.3f})") # Debug
        return np.array([x, y, z], dtype=np.float32)

    def _place_obstacles(self):
        """Places obstacles randomly in the environment, avoiding the base and target."""
        # Remove existing obstacles first
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids.clear()
        self.obstacle_positions.clear()

        for _ in range(self.num_obstacles):
            placed = False
            max_placement_attempts = 50
            for attempt in range(max_placement_attempts):
                # Generate random position within a reasonable area
                # Use the actual max_target_radius (already includes 90% factor)
                placement_radius_limit = self.fixed_max_target_radius
                radius = self.obstacle_safe_dist_from_base + np.random.rand() * (placement_radius_limit - self.obstacle_safe_dist_from_base)
                theta = np.random.rand() * 2 * math.pi
                # Place obstacles mostly on the plane or slightly above
                phi = (np.random.rand() * 0.2 + 0.4) * math.pi # Restrict phi more (closer to horizontal plane)

                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = self.obstacle_radius * 1.1 # Place slightly above ground plane based on radius
                # z = radius * math.cos(phi) # Old spherical placement

                pos = np.array([x, y, z], dtype=np.float32)

                # Check distance from base and target
                dist_from_base = np.linalg.norm(pos)
                dist_from_target = np.linalg.norm(pos - self.target_position)

                if (dist_from_base > self.obstacle_safe_dist_from_base and
                    dist_from_target > self.obstacle_safe_dist_from_target):

                    # Create the obstacle temporarily to check for initial collisions
                    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                          radius=self.obstacle_radius,
                                                          rgbaColor=[0.2, 0.2, 0.8, 0.8]) # Blueish color
                    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                               radius=self.obstacle_radius)
                    obstacle_id = p.createMultiBody(baseMass=0, # Static obstacle
                                                  baseCollisionShapeIndex=collision_shape_id,
                                                  baseVisualShapeIndex=visual_shape_id,
                                                  basePosition=pos)

                    # Check for immediate collisions after creation
                    p.performCollisionDetection() # Update collision states
                    contacts_plane = p.getContactPoints(bodyA=obstacle_id, bodyB=self.plane_id)
                    contacts_robot = p.getContactPoints(bodyA=obstacle_id, bodyB=self.robot_id)

                    if not contacts_plane and not contacts_robot:
                        # If no initial collisions, keep the obstacle
                        self.obstacle_ids.append(obstacle_id)
                        self.obstacle_positions.append(pos)
                        placed = True
                        if self.render_mode == 'human': # Print debug info only if rendering
                             # Keep print for this specific debug scenario if desired, or change to logger.debug
                             print(f"[Env Debug] Placed obstacle at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        break # Exit placement attempt loop
                    else:
                        # If colliding, remove it and try again
                        p.removeBody(obstacle_id)
                        # print(f"[Env Debug] Obstacle placement attempt {attempt+1} failed due to initial collision.") # Debug

            if not placed:
                # Use logger
                logger.warning(f"Warning: Could not place obstacle {_ + 1} after {max_placement_attempts} attempts.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_counter = 0

        # Reset the MINIMUM target radius to its initial value (which is the fixed max radius)
        self.current_min_target_radius = self.initial_min_target_radius
        # Clear recent results on reset
        self.episode_results.clear()

        # Reset joint positions to initial/home state
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, targetValue=self.initial_joint_positions[i], targetVelocity=0)
            # Ensure motor control is reset/disabled initially if needed
            p.setJointMotorControl2(self.robot_id, joint_index, p.VELOCITY_CONTROL, force=0)


        # Generate a new target using the potentially updated curriculum state
        # (uses current_min_target_radius and fixed_max_target_radius)
        self.target_position = self._get_random_target_pos()

        # --- Reset Base Angle Error Tracking --- 
        # Get initial base angle
        initial_joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        initial_base_angle = initial_joint_states[0][0] # Position of the first joint (base)
        # Calculate initial target azimuth
        target_azimuth = math.atan2(self.target_position[1], self.target_position[0])
        # Calculate initial angle error
        angle_diff_raw = target_azimuth - initial_base_angle
        initial_angle_error = math.atan2(math.sin(angle_diff_raw), math.cos(angle_diff_raw))
        self.previous_base_angle_error = abs(initial_angle_error)

        # Remove previous target visual marker if it exists
        if hasattr(self, 'target_visual_shape_id') and self.target_visual_shape_id >= 0:
             p.removeBody(self.target_visual_shape_id)

        # --- Place Obstacles (Conditional) --- 
        if self.start_with_obstacles:
             self._place_obstacles()
        else:
             # Ensure any old obstacles are removed if starting without them
             for obs_id in self.obstacle_ids:
                 try:
                      p.removeBody(obs_id)
                 except p.error:
                      pass # Ignore error if body already removed
             self.obstacle_ids.clear()
             self.obstacle_positions.clear()

        # Visualise the target (optional, useful for debugging)
        if self.render_mode == 'human' or True: # Always create for info calculation, remove if performance hit
            target_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.8])
            target_collision_shape = -1 # No collision for target marker
            self.target_visual_shape_id = p.createMultiBody(baseMass=0,
                                                            baseCollisionShapeIndex=target_collision_shape,
                                                            baseVisualShapeIndex=target_visual_shape,
                                                            basePosition=self.target_position)


        observation = self._get_obs()
        info = self._get_info() # Info now includes current_min_target_radius

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
        info = self._get_info() # Gets initial info dict with curriculum flags set to False
        distance = info['distance']

        # --- Calculate Reward ---
        # Dense reward: negative squared distance to target
        reward = -(distance**2)

        # --- Base Rotation Reward (Potential-Based) --- 
        base_angle = current_positions[0] # Base joint is index 0
        target_azimuth = math.atan2(self.target_position[1], self.target_position[0])
        angle_diff_raw = target_azimuth - base_angle
        current_angle_error_signed = math.atan2(math.sin(angle_diff_raw), math.cos(angle_diff_raw))
        current_angle_error_abs = abs(current_angle_error_signed)

        if self.previous_base_angle_error is not None:
            angle_improvement = self.previous_base_angle_error - current_angle_error_abs
            rotation_reward = self.angle_bonus_factor * angle_improvement
            reward += rotation_reward
            # print(f"Angle Err: {current_angle_error_abs:.3f}, Prev: {self.previous_base_angle_error:.3f}, Improv: {angle_improvement:.3f}, RotRew: {rotation_reward:.3f}") # Debug

        # Update for next step
        self.previous_base_angle_error = current_angle_error_abs

        # Bonus for reaching the target accuracy
        success_bonus = 100.0
        terminated = False
        success = False
        if distance < self._target_accuracy:
            reward += success_bonus
            terminated = True
            success = True
            info['success'] = True # Mark success in info

        # --- Check Truncation ---
        # Initialise truncated here first to satisfy linter, will be properly set below
        truncated = False

        # --- Continuous Curriculum Update --- 
        # Store result of the episode (success or failure) when episode ends
        # Only update curriculum state when an episode actually finishes
        if terminated or truncated:
             self.episode_results.append(success)
             # Check if we have enough data and if success rate is high enough
             if len(self.episode_results) == self.success_rate_window_size:
                 success_rate = sum(self.episode_results) / self.success_rate_window_size
                 # Check if success rate is high and min radius is not already at its final minimum value
                 if success_rate >= self.success_rate_threshold and \
                    self.current_min_target_radius > self.final_min_target_radius:
                      
                      old_radius = self.current_min_target_radius
                      # Decrease the minimum radius, ensuring it doesn't go below the final target
                      self.current_min_target_radius = max(
                          self.current_min_target_radius - self.radius_decrease_step,
                          self.final_min_target_radius
                      )
                      logger.info(f"Curriculum Update: Success rate {success_rate:.2f} >= {self.success_rate_threshold:.2f}. Decreasing min target radius from {old_radius:.3f} to {self.current_min_target_radius:.3f}.")
                      # Clear the deque to require a new window of success
                      self.episode_results.clear()
                 # else: Log if needed: print(f"Success rate {success_rate:.2f} < {self.success_rate_threshold:.2f}, not decreasing radius.")

        # --- Actual Truncation Check ---
        # Reset truncated to False before checking step count
        truncated = False 
        if self._step_counter >= self._max_episode_steps:
            truncated = True
            # Reset success streak if truncated without success
            # if not success: # Logic moved to curriculum update
            #     pass

        # --- Check for End-Effector Collisions ---
        collision_penalty = -50.0 # Define penalty for collision
        collision_detected = False

        # 1. Check collision with the plane
        plane_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=self.end_effector_link_index)
        if plane_contacts:
            collision_detected = True
            # print(f"Step {self._step_counter}: Collision detected - End-effector hit plane.") # Debug

        # 2. Check self-collision involving the end-effector
        if not collision_detected: # Only check if not already collided with plane
            self_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
            # print(f"Step {self._step_counter}: Raw self-contacts: {self_contacts}") # Debug raw contacts
            for contact in self_contacts:
                link_a = contact[3]
                link_b = contact[4]

                # Check if the contact involves the end-effector link and another different link,
                # AND ignore contact with the direct parent link.
                is_ee_involved = (link_a == self.end_effector_link_index or link_b == self.end_effector_link_index)
                is_ee_parent_contact = (self.parent_link_index != -1 and \
                                        ((link_a == self.end_effector_link_index and link_b == self.parent_link_index) or \
                                         (link_b == self.end_effector_link_index and link_a == self.parent_link_index)))

                if is_ee_involved and not is_ee_parent_contact and link_a != link_b:
                    collision_detected = True
                    colliding_link = link_a if link_b == self.end_effector_link_index else link_b
                    # print(f"Step {self._step_counter}: Collision detected - End-effector (link {self.end_effector_link_index}) hit link {colliding_link}.") # Debug
                    break # Exit loop once a relevant self-collision is found

        # 3. Check collision with obstacles
        if not terminated: # Only check if not already terminated by other collisions
            for obstacle_id in self.obstacle_ids:
                # Check contacts between any robot link and the obstacle
                obstacle_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obstacle_id)
                if obstacle_contacts:
                    # print(f"Step {self._step_counter}: Collision detected - Robot hit obstacle {obstacle_id}.") # Debug
                    reward += self.obstacle_collision_penalty
                    terminated = True
                    info['obstacle_collision'] = True # Set specific flag
                    info['collision'] = True # Also set general collision flag
                    break # Exit obstacle check loop

        # Ensure 'is_success' key is present in info for SB3 logging buffer
        info['is_success'] = success

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
        # Remove obstacles before disconnecting
        if self.physics_client >= 0:
            for obs_id in self.obstacle_ids:
                try:
                    p.removeBody(obs_id)
                except p.error as e:
                    # Sometimes body might already be removed, ignore error
                    # print(f"Warning: Could not remove obstacle {obs_id} on close: {e}")
                    pass
            self.obstacle_ids.clear()
            self.obstacle_positions.clear()

        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
            self.physics_client = -1

# Example usage (optional, for testing the environment)
if __name__ == '__main__':
    # --- REMOVE sys.path adjustment and path redefinitions ---
    # Paths are defined globally now and should work when run via `python -m src.fanuc_env`
    # import sys
    # if '' not in sys.path:
    #      sys.path.insert(0, os.path.dirname(__file__))
    # WORKSPACE_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), '..', "workspace_config.json")
    # logger.info(f"Executing {__file__} directly. WORKSPACE_CONFIG_FILENAME set to: {WORKSPACE_CONFIG_FILENAME}")

    # Example run uses global WORKSPACE_CONFIG_FILENAME implicitly via constructor
    env = FanucEnv(render_mode='human')
    obs, info = env.reset()
    # Use logger for standalone testing info
    logger.info(f"Standalone Test: Initial Observation: {obs}")
    logger.info(f"Standalone Test: Initial Info: {info}")

    episodes = 20 # Increased episodes for testing
    for ep in range(episodes):
        # Use logger
        logger.info(f"--- Standalone Test: Episode {ep+1} ---")
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
                # Use logger
                logger.info(f"Standalone Test: Episode finished after {step} steps. Terminated={terminated}, Truncated={truncated}. Total Reward: {total_reward:.2f}. Final Distance: {info['distance']:.4f}")
                break

    env.close()
    # Use logger
    logger.info("Standalone Test: Environment closed.") 