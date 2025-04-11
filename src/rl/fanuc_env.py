import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import math
import json
import random
import logging
import collections
import traceback

from config import robot_config

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Workspace config file
WORKSPACE_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), '..', '..', "config", "workspace_config.json")

class FanucEnv(gym.Env):
    """Gymnasium environment for FANUC LRMate 200iD using PyBullet."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_episode_steps=1000, target_accuracy=0.02, angle_bonus_factor=5.0, start_with_obstacles=True, force_outer_radius=False, distance_reward_multiplier=1.0):
        super().__init__()
        # Render mode
        self.render_mode = render_mode
        # Max steps per episode
        self._max_episode_steps = max_episode_steps
        # Target accuracy (m)
        self._target_accuracy = target_accuracy
        self._step_counter = 0
        self.force_outer_radius = force_outer_radius
        # Distance reward factor
        self.distance_reward_multiplier = distance_reward_multiplier
        # Min reach default (m)
        min_reach_default = 0.02
        # Max reach default (m)
        max_reach_default = 1.26
        # Safety factor
        safety_factor = 0.95
        loaded_min_reach = min_reach_default
        loaded_max_reach = max_reach_default
        # Reach at midpoint (m)
        loaded_reach_at_midpoint = 0.0

        try:
            if os.path.exists(WORKSPACE_CONFIG_FILENAME):
                with open(WORKSPACE_CONFIG_FILENAME, 'r') as f:
                    workspace_config = json.load(f)
                # Load or use defaults
                loaded_min_reach = workspace_config.get('min_reach', min_reach_default)
                loaded_max_reach = workspace_config.get('max_reach', max_reach_default)
                loaded_reach_at_midpoint = workspace_config.get('reach_at_midpoint_z', 0.0)
                logger.info(f"Loaded workspace config: Min={loaded_min_reach:.4f}, AbsMax={loaded_max_reach:.4f}, MidpointZReach={loaded_reach_at_midpoint:.4f}")
            else:
                logger.warning(f"{WORKSPACE_CONFIG_FILENAME} not found. Using default workspace radii: Min={min_reach_default}, Max={max_reach_default}")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Warning: Error loading {WORKSPACE_CONFIG_FILENAME}: {e}. Using default workspace radii.")

        # Min base radius (m)
        self.min_base_radius = loaded_min_reach

        if loaded_reach_at_midpoint > 0:
            base_radius = loaded_reach_at_midpoint
            radius_source = "reach near midpoint Z"
        else:
            # Fallback
            base_radius = loaded_max_reach
            radius_source = "absolute max reach"
            logger.warning(f"Reach at midpoint Z not available or invalid. Using fallback {radius_source}.")

        # Max target radius (m)
        self.max_target_radius = base_radius * safety_factor
        logger.info(f"Using Max Target Radius: {self.max_target_radius:.4f} (Based on {radius_source}={base_radius:.4f} * {safety_factor})")

        # Min target radius multiplier
        self.min_target_radius_multiplier = 10.0
        # Min target radius (m)
        self.min_target_radius = self.min_base_radius * self.min_target_radius_multiplier
        logger.info(f"Calculated Min Target Radius: {self.min_target_radius:.4f} (Base: {self.min_base_radius:.4f} * {self.min_target_radius_multiplier}) ")

        # Fixed max target radius (m)
        self.fixed_max_target_radius = self.max_target_radius
        # Final min target radius (m)
        self.final_min_target_radius = self.min_target_radius
        # Initial min target radius (m)
        self.initial_min_target_radius = self.fixed_max_target_radius
        # Current min target radius (m)
        self.current_min_target_radius = self.initial_min_target_radius

        # Success rate window size (episodes)
        self.success_rate_window_size = 20
        # Success rate threshold
        self.success_rate_threshold = 0.5
        # Radius decrease step (m)
        self.radius_decrease_step = 0.05
        self.episode_results = collections.deque(maxlen=self.success_rate_window_size)

        # Previous base angle error (rad)
        self.previous_base_angle_error = None
        # Angle bonus factor
        self.angle_bonus_factor = angle_bonus_factor

        # Number of obstacles
        self.num_obstacles = 1
        # Obstacle radius (m)
        self.obstacle_radius = 0.05
        self.obstacle_ids = []
        self.obstacle_positions = []
        # Obstacle collision penalty
        self.obstacle_collision_penalty = -100.0
        # Safe distance from base (m)
        self.obstacle_safe_dist_from_base = 0.3
        # Safe distance from target (m)
        self.obstacle_safe_dist_from_target = 0.2
        self.start_with_obstacles = start_with_obstacles

        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100)

        self.plane_id = p.loadURDF("plane.urdf")

        script_dir = os.path.dirname(__file__)
        project_root = os.path.join(script_dir, '..', '..')
        robot_model_dir = os.path.join(project_root, "assets", "robot_model")
        urdf_file_path = os.path.join(robot_model_dir, "urdf", "Fanuc.urdf")

        p.setAdditionalSearchPath(robot_model_dir)

        self.robot_id = p.loadURDF(
            urdf_file_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        # Joint limits lower (rad)
        self.joint_limits_lower = robot_config.JOINT_LIMITS_LOWER_RAD
        # Joint limits upper (rad)
        self.joint_limits_upper = robot_config.JOINT_LIMITS_UPPER_RAD
        # Velocity limits (rad/s)
        self.velocity_limits = robot_config.VELOCITY_LIMITS_RAD_S
        # Number of controllable joints
        self.num_controllable_joints = robot_config.NUM_CONTROLLED_JOINTS

        link_name_to_index = {p.getJointInfo(self.robot_id, i)[12].decode('UTF-8'): i for i in range(self.num_joints)}
        self.end_effector_link_index = -1

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('UTF-8')
            joint_type = info[2]

            if joint_type == p.JOINT_REVOLUTE:
                if len(self.joint_indices) < self.num_controllable_joints:
                    self.joint_indices.append(i)
                    self.joint_names.append(joint_name)

            link_name = info[12].decode('UTF-8')
            if link_name == robot_config.END_EFFECTOR_LINK_NAME:
                self.end_effector_link_index = i

        if self.end_effector_link_index == -1:
            logger.warning("Warning: Could not find link 'Part6'. Using last joint index as end effector.")
            self.end_effector_link_index = p.getNumJoints(self.robot_id) - 1

        if len(self.joint_indices) != 5:
             raise ValueError(f"Expected 5 controllable joints, but found {len(self.joint_indices)}. URDF might be different than expected.")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_controllable_joints,), dtype=np.float32)

        obs_low = np.concatenate([
            self.joint_limits_lower,
            -self.velocity_limits * 2,
            np.array([-np.inf] * 3, dtype=np.float32),
            np.array([-1.0] * self.num_controllable_joints, dtype=np.float32),
            np.array([-np.inf] * 3, dtype=np.float32)
        ])
        obs_high = np.concatenate([
            self.joint_limits_upper,
            self.velocity_limits * 2,
            np.array([np.inf] * 3, dtype=np.float32),
            np.array([1.0] * self.num_controllable_joints, dtype=np.float32),
            np.array([np.inf] * 3, dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Target position (m)
        self.target_position = np.zeros(3, dtype=np.float32)
        # Initial joint positions (rad)
        self.initial_joint_positions = np.zeros(self.num_controllable_joints)
        self.target_visual_shape_id = -1

        self.parent_link_index = -1
        try:
            ee_joint_info = p.getJointInfo(self.robot_id, self.end_effector_link_index)
            self.parent_link_index = ee_joint_info[16]
        except Exception as e:
            logger.warning(f"Warning: Could not get parent link index for end-effector. Error: {e}")

        init_log_message = (
            f"FanucEnv initialised:\\n"
            f"  - Joints: {self.joint_names}\\n"
            f"  - End Effector Link Index: {self.end_effector_link_index}\\n"
            f"  - Action Space Shape: {self.action_space.shape}\\n"
            f"  - Observation Space Shape: {self.observation_space.shape}\\n"
            f"  - Curriculum (Dec Min Radius): Fixed MaxR={self.fixed_max_target_radius:.3f}, Target MinR={self.final_min_target_radius:.3f}, Initial MinR={self.initial_min_target_radius:.3f}, Window={self.success_rate_window_size}, Threshold={self.success_rate_threshold:.2f}, Step={self.radius_decrease_step}\\n"
            f"  - Angle Bonus Factor: {self.angle_bonus_factor}\\n"
            f"  - Start with Obstacles: {self.start_with_obstacles}"
        )
        logger.info(init_log_message)


    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        # Joint positions (rad)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        # Joint velocities (rad/s)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        joint_ranges = self.joint_limits_upper - self.joint_limits_lower
        # Avoid div zero
        joint_ranges[joint_ranges == 0] = 1e-6
        normalised_joint_positions = 2 * (joint_positions - self.joint_limits_lower) / joint_ranges - 1.0
        normalised_joint_positions = np.clip(normalised_joint_positions, -1.0, 1.0).astype(np.float32)

        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index, computeForwardKinematics=True)
        # End effector position (m)
        ee_position = np.array(ee_state[4], dtype=np.float32)
        # Relative position to target (m)
        relative_position = self.target_position - ee_position

        if self.obstacle_positions:
            # (m)
            relative_obstacle_pos = self.obstacle_positions[0] - ee_position
        else:
            # Default if no obstacles
            relative_obstacle_pos = np.zeros(3, dtype=np.float32)

        return np.concatenate([
            joint_positions,
            joint_velocities,
            relative_position,
            normalised_joint_positions,
            relative_obstacle_pos
        ])

    def _get_info(self):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        # End effector position (m)
        ee_position = np.array(ee_state[4], dtype=np.float32)
        # Distance to target (m)
        distance = np.linalg.norm(self.target_position - ee_position)
        return {
            "distance": distance,
            "target_position": self.target_position.copy(),
            "collision": False,
            "success": False,
            # Current min target radius (m)
            "current_min_target_radius": self.current_min_target_radius,
            "obstacle_collision": False
        }

    def _get_random_target_pos(self):
        if self.force_outer_radius:
            # Min radius (m)
            min_radius = self.fixed_max_target_radius * 0.80
            # Max radius (m)
            max_radius = self.fixed_max_target_radius
            # Ensure min < max
            min_radius = min(min_radius, max_radius - 1e-6)
            logger.info("[Test Mode] Forcing target generation to outer edge.")
        else:
            # Min radius (m)
            min_radius = self.current_min_target_radius
            # Max radius (m)
            max_radius = self.fixed_max_target_radius

        # Ensure max >= min
        max_radius = max(max_radius, min_radius + 1e-6)

        min_radius = max(min_radius, self.final_min_target_radius)
        # Re-ensure min < max
        min_radius = min(min_radius, max_radius - 1e-6)

        # (m)
        radius = min_radius + np.random.rand() * (max_radius - min_radius)
        # (rad)
        theta = np.random.rand() * 2 * math.pi
        # (rad)
        phi = (np.random.rand() * 0.6 + 0.1) * math.pi

        # (m)
        x = radius * math.sin(phi) * math.cos(theta)
        # (m)
        y = radius * math.sin(phi) * math.sin(theta)
        # (m)
        z = radius * math.cos(phi)

        # (m)
        min_z_height = 0.05
        if z < min_z_height:
            # Regenerate
            return self._get_random_target_pos()

        if self.render_mode == 'human':
            print(f"[Env Debug] CurrentMinR={self.current_min_target_radius:.3f}, "
                  f"Bounds=[{min_radius:.3f}-{max_radius:.3f}], "
                  f"Sampled R={radius:.3f}, Phi={phi:.3f}, Theta={theta:.3f}, "
                  f"Target=[{x:.3f}, {y:.3f}, {z:.3f}]")

        return np.array([x, y, z], dtype=np.float32)

    def _place_obstacles(self):
        """Places obstacles randomly, avoiding base and target."""
        # Remove existing obstacles
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids.clear()
        self.obstacle_positions.clear()

        for _ in range(self.num_obstacles):
            placed = False
            # Max placement attempts
            max_placement_attempts = 50
            for attempt in range(max_placement_attempts):
                # Placement radius limit (m)
                placement_radius_limit = self.fixed_max_target_radius
                # Radius (m)
                radius = self.obstacle_safe_dist_from_base + np.random.rand() * (placement_radius_limit - self.obstacle_safe_dist_from_base)
                # Theta (rad)
                theta = np.random.rand() * 2 * math.pi
                # Phi (rad)
                phi = (np.random.rand() * 0.2 + 0.4) * math.pi

                # (m)
                x = radius * math.sin(phi) * math.cos(theta)
                # (m)
                y = radius * math.sin(phi) * math.sin(theta)
                # (m)
                z = self.obstacle_radius * 1.1 # Slightly above ground

                # (m)
                pos = np.array([x, y, z], dtype=np.float32)

                # Check distance from base/target
                # (m)
                dist_from_base = np.linalg.norm(pos)
                # (m)
                dist_from_target = np.linalg.norm(pos - self.target_position)

                if (dist_from_base > self.obstacle_safe_dist_from_base and
                    dist_from_target > self.obstacle_safe_dist_from_target):

                    # Create obstacle to check collisions
                    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                          radius=self.obstacle_radius,
                                                          rgbaColor=[0.2, 0.2, 0.8, 0.8])
                    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                               radius=self.obstacle_radius)
                    obstacle_id = p.createMultiBody(baseMass=0, # Massless static object
                                                  baseCollisionShapeIndex=collision_shape_id,
                                                  baseVisualShapeIndex=visual_shape_id,
                                                  basePosition=pos)

                    # Check initial collisions
                    p.performCollisionDetection()
                    contacts_plane = p.getContactPoints(bodyA=obstacle_id, bodyB=self.plane_id)
                    contacts_robot = p.getContactPoints(bodyA=obstacle_id, bodyB=self.robot_id)

                    if not contacts_plane and not contacts_robot:
                        # Keep obstacle
                        self.obstacle_ids.append(obstacle_id)
                        self.obstacle_positions.append(pos)
                        placed = True
                        if self.render_mode == 'human':
                             print(f"[Env Debug] Placed obstacle at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                        break
                    else:
                        # Remove colliding obstacle
                        p.removeBody(obstacle_id)

            if not placed:
                logger.warning(f"Warning: Could not place obstacle {_ + 1} after {max_placement_attempts} attempts.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_counter = 0
        # Current min target radius (m)
        self.current_min_target_radius = self.initial_min_target_radius
        self.episode_results.clear()

        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, targetValue=self.initial_joint_positions[i], targetVelocity=0)
            p.setJointMotorControl2(self.robot_id, joint_index, p.VELOCITY_CONTROL, force=0)

        # Target position (m)
        self.target_position = self._get_random_target_pos()

        initial_joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        # Initial base angle (rad)
        initial_base_angle = initial_joint_states[0][0]
        # Target azimuth (rad)
        target_azimuth = math.atan2(self.target_position[1], self.target_position[0])
        angle_diff_raw = target_azimuth - initial_base_angle
        # Initial angle error (rad)
        initial_angle_error = math.atan2(math.sin(angle_diff_raw), math.cos(angle_diff_raw))
        # Previous base angle error (rad)
        self.previous_base_angle_error = abs(initial_angle_error)

        if hasattr(self, 'target_visual_shape_id') and self.target_visual_shape_id >= 0:
             p.removeBody(self.target_visual_shape_id)

        if self.start_with_obstacles:
             self._place_obstacles()
        else:
             # Remove old obstacles
             for obs_id in self.obstacle_ids:
                 try:
                      p.removeBody(obs_id)
                 except p.error:
                      # Ignore error if already removed
                      pass
             self.obstacle_ids.clear()
             self.obstacle_positions.clear()

        if self.render_mode == 'human' or True:
            target_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.8])
            # Visual only, no collision physics
            target_collision_shape = -1
            self.target_visual_shape_id = p.createMultiBody(baseMass=0,
                                                            baseCollisionShapeIndex=target_collision_shape,
                                                            baseVisualShapeIndex=target_visual_shape,
                                                            basePosition=self.target_position)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Target velocities (rad/s)
        target_velocities = action * self.velocity_limits

        # Current positions (rad)
        current_positions = np.array([p.getJointState(self.robot_id, i)[0] for i in self.joint_indices])

        for i, joint_index in enumerate(self.joint_indices):
            # Joint position (rad)
            pos = current_positions[i]
            # Joint velocity (rad/s)
            vel = target_velocities[i]

            if ((pos <= self.joint_limits_lower[i] and vel < 0) or
                (pos >= self.joint_limits_upper[i] and vel > 0)):
                # Stop motion
                target_velocities[i] = 0

            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[i],
                # Force applied (N*m or N)
                force=100
            )

        p.stepSimulation()
        self._step_counter += 1

        observation = self._get_obs()
        info = self._get_info()
        # Distance to target (m)
        distance = info['distance']

        # Linear distance penalty
        reward = -distance * self.distance_reward_multiplier

        # Base angle (rad)
        base_angle = current_positions[0]
        # Target azimuth (rad)
        target_azimuth = math.atan2(self.target_position[1], self.target_position[0])
        # Angle difference raw (rad)
        angle_diff_raw = target_azimuth - base_angle
        # Current angle error signed (rad)
        current_angle_error_signed = math.atan2(math.sin(angle_diff_raw), math.cos(angle_diff_raw))
        # Current angle error absolute (rad)
        current_angle_error_abs = abs(current_angle_error_signed)

        if self.previous_base_angle_error is not None:
            # Angle improvement (rad)
            angle_improvement = self.previous_base_angle_error - current_angle_error_abs
            # Rotation reward factor
            rotation_reward = self.angle_bonus_factor * angle_improvement
            reward += rotation_reward

        # Update previous angle error (rad)
        self.previous_base_angle_error = current_angle_error_abs

        # Success bonus reward
        success_bonus = 1000.0
        terminated = False
        success = False
        if distance < self._target_accuracy:
            reward += success_bonus
            terminated = True
            success = True
            info['success'] = True

        truncated = False

        if terminated or truncated:
             self.episode_results.append(success)
             if len(self.episode_results) == self.success_rate_window_size:
                 # Success rate calculation
                 success_rate = sum(self.episode_results) / self.success_rate_window_size
                 if success_rate >= self.success_rate_threshold and \
                    self.current_min_target_radius > self.final_min_target_radius:

                      # Old radius value (m)
                      old_radius = self.current_min_target_radius
                      # Decrease min radius (m)
                      self.current_min_target_radius = max(
                          self.current_min_target_radius - self.radius_decrease_step,
                          self.final_min_target_radius
                      )
                      logger.info(f"Curriculum Update: Success rate {success_rate:.2f} >= {self.success_rate_threshold:.2f}. Decreasing min target radius from {old_radius:.3f} to {self.current_min_target_radius:.3f}.")
                      self.episode_results.clear()

        # Check max steps exceeded
        truncated = False
        if self._step_counter >= self._max_episode_steps:
            truncated = True

        # Collision penalty value
        collision_penalty = -50.0
        collision_detected = False

        # Check for plane collision
        plane_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=self.end_effector_link_index)
        if plane_contacts:
            collision_detected = True

        if not collision_detected:
            # Check for self-collision
            self_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
            for contact in self_contacts:
                link_a = contact[3]
                link_b = contact[4]

                # Determine if end effector is involved
                is_ee_involved = (link_a == self.end_effector_link_index or link_b == self.end_effector_link_index)
                # Check if contact is with parent link
                is_ee_parent_contact = (self.parent_link_index != -1 and \
                                        ((link_a == self.end_effector_link_index and link_b == self.parent_link_index) or \
                                         (link_b == self.end_effector_link_index and link_a == self.parent_link_index)))

                if is_ee_involved and not is_ee_parent_contact and link_a != link_b:
                    collision_detected = True
                    # Identify colliding link
                    colliding_link = link_a if link_b == self.end_effector_link_index else link_b
                    break

        if not terminated:
            for obstacle_id in self.obstacle_ids:
                # Check for obstacle collision
                obstacle_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obstacle_id)
                if obstacle_contacts:
                    reward += self.obstacle_collision_penalty
                    terminated = True
                    info['obstacle_collision'] = True
                    info['collision'] = True
                    break

        if collision_detected and not terminated:
            # Apply collision penalty
            reward += collision_penalty
            terminated = True
            info['collision'] = True

        info['is_success'] = success

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            pass


    def close(self):
        if self.physics_client >= 0:
            for obs_id in self.obstacle_ids:
                try:
                    p.removeBody(obs_id)
                except p.error:
                    pass
            self.obstacle_ids.clear()
            self.obstacle_positions.clear()

        if self.physics_client >= 0:
            p.disconnect(self.physics_client)
            self.physics_client = -1

if __name__ == '__main__':

    env = FanucEnv(render_mode='human')
    obs, info = env.reset()
    logger.info(f"Standalone Test: Initial Observation: {obs}")
    logger.info(f"Standalone Test: Initial Info: {info}")

    # Number of episodes
    episodes = 20
    for ep in range(episodes):
        logger.info(f"--- Standalone Test: Episode {ep+1} ---")
        obs, info = env.reset()
        done = False
        # Step counter
        step = 0
        # Total reward accumulator
        total_reward = 0
        while not done:
            # Random action sample
            action = env.action_space.sample()
            # Step environment with action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if done:
                logger.info(f"Standalone Test: Episode finished after {step} steps. Terminated={terminated}, Truncated={truncated}. Total Reward: {total_reward:.2f}. Final Distance: {info['distance']:.4f}")
                break

    # Close environment
    env.close()
    logger.info("Standalone Test: Environment closed.") 