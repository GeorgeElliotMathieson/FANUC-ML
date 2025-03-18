# rl_env.py
import gym
from gym import spaces
import numpy as np
from domain_randomisation import DomainRandomizedEnv

class DomainRandomizedRLEnv(gym.Env):
    def __init__(self, render=False, randomize=True, randomize_target_every_step=False):
        super(DomainRandomizedRLEnv, self).__init__()
        
        # Create the PyBullet simulation with domain randomization
        self.robot_env = DomainRandomizedEnv(render=render, randomize=randomize)
        
        # Define action and observation space
        # Actions: 6 joint positions
        self.action_space = spaces.Box(
            low=np.array([self.robot_env.joint_limits[i][0] for i in range(6)]),
            high=np.array([self.robot_env.joint_limits[i][1] for i in range(6)]),
            dtype=np.float32
        )
        
        # Observations: joint positions, velocities, end-effector pose, and relative position to target
        # 6 joint positions + 6 joint velocities + 3 EE position + 4 EE orientation + 3 relative position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        
        # Target position for the task
        self.target_position = np.array([0.5, 0.0, 0.5])  # Example target
        
        # Default workspace size
        self.workspace_size = 0.5
        
        # Target accuracy threshold (6cm)
        self.accuracy_threshold = 0.06
        
        # Whether to randomize the target position for each step
        self.randomize_target_every_step = randomize_target_every_step
        
    def seed(self, seed=None):
        """Set the random seed for the environment"""
        np.random.seed(seed)
        return [seed]
    
    def reset(self):
        # Reset the robot environment (includes domain randomization)
        state = self.robot_env.reset()
        
        # Set a new random target within the workspace
        self.target_position = self._sample_target()
        
        # Add relative position to the observation
        enhanced_state = self._enhance_observation(state)
        
        return enhanced_state
    
    def step(self, action):
        # Take a step in the environment
        next_state, _, _, _ = self.robot_env.step(action)
        
        # Randomize the target position for each step
        if hasattr(self, 'randomize_target_every_step') and self.randomize_target_every_step:
            self.target_position = self._sample_target()
        
        # Calculate reward based on distance to target
        ee_position = next_state[12:15]  # Extract end-effector position from state
        distance = np.linalg.norm(ee_position - self.target_position)
        
        # Enhanced reward function:
        # 1. Base reward is negative distance (closer is better)
        # 2. Add bonus reward when within accuracy threshold
        # 3. Add penalty for large actions to encourage smooth motion
        action_penalty = 0.01 * np.sum(np.square(action))
        
        # Base reward is negative distance
        reward = -distance
        
        # Add bonus reward when within accuracy threshold
        if distance <= self.accuracy_threshold:
            accuracy_bonus = 10.0 * (self.accuracy_threshold - distance) / self.accuracy_threshold
            reward += accuracy_bonus
        
        # Apply action penalty
        reward -= action_penalty
        
        # Task is done if we're close enough to the target
        done = distance < self.accuracy_threshold
        
        # Additional info
        info = {
            "distance": distance,
            "target_reached": done,
            "accuracy_threshold": self.accuracy_threshold,
            "target_position": self.target_position  # Include target position in info
        }
        
        # Add relative position to the observation
        enhanced_state = self._enhance_observation(next_state)
        
        return enhanced_state, reward, done, info
    
    def _sample_target(self):
        """
        Sample a random target position within the robot's reachable workspace.
        
        This method ensures that the target position is reachable by the robot arm,
        considering the joint angle limitations and the physical constraints of the
        FANUC LR Mate 200iC robot arm.
        
        Returns:
            np.array: A 3D position [x, y, z] for the target
        """
        # The FANUC LR Mate 200iC has a reach of approximately 0.7m
        # We'll use this to constrain our sampling to ensure reachable positions
        
        # Calculate the maximum reach based on workspace_size
        # But ensure it doesn't exceed the physical capabilities of the robot
        max_reach = min(0.7, self.workspace_size)
        
        # Set minimum distance from robot base to avoid targets too close to the robot
        min_distance = 0.25  # Minimum 25cm from robot base
        
        # Choose a sampling method based on a random number
        sampling_method = np.random.randint(0, 4)  # 0-3 for four different methods
        
        if sampling_method == 0:
            # METHOD 1: Fixed test positions to verify coverage
            # These positions are at the extremes of the workspace
            # IMPORTANT: The robot's base is at approximately [0, 0, 0]
            # Positive X is forward, positive Y is left, positive Z is up
            test_positions = [
                [0.5, 0.0, 0.3],     # Front
                [-0.5, 0.0, 0.3],    # Behind (negative X)
                [0.0, 0.5, 0.3],     # Left side (positive Y)
                [0.0, -0.5, 0.3],    # Right side (negative Y)
                [0.0, 0.0, 0.7],     # Above
                [0.3, 0.0, 0.3],     # Front middle (not too close)
                [0.5, 0.5, 0.3],     # Front left
                [0.5, -0.5, 0.3],    # Front right
                [-0.5, 0.5, 0.3],    # Back left
                [-0.5, -0.5, 0.3],   # Back right
                [0.5, 0.0, 0.7],     # Front high
                [-0.5, 0.0, 0.7],    # Back high
                [0.5, 0.0, 0.3],     # Front middle
                [-0.5, 0.0, 0.3],    # Back middle
                [0.0, 0.5, 0.7],     # Left high
                [0.0, -0.5, 0.7],    # Right high
                [0.3, 0.5, 0.3],     # Front left middle
                [0.3, -0.5, 0.3],    # Front right middle
                [0.4, 0.4, 0.6],     # Front left high
                [0.4, -0.4, 0.6],    # Front right high
                [-0.4, 0.4, 0.6],    # Back left high
                [-0.4, -0.4, 0.6],   # Back right high
                [0.4, 0.4, 0.3],     # Front left middle
                [0.4, -0.4, 0.3],    # Front right middle
                [-0.4, 0.4, 0.3],    # Back left middle
                [-0.4, -0.4, 0.3]    # Back right middle
            ]
            return np.array(test_positions[np.random.randint(0, len(test_positions))])
            
        elif sampling_method == 1:
            # METHOD 2: Full spherical sampling with no restrictions
            # Sample radius - use range from min_distance to max_reach
            radius = np.random.uniform(min_distance, max_reach)
            
            # Sample angles - use full ranges
            azimuth = np.random.uniform(-np.pi, np.pi)  # Full 360 degrees
            elevation = np.random.uniform(-np.pi/2, np.pi/2)  # Full 180 degrees
            
            # Convert spherical coordinates to cartesian
            # IMPORTANT: Center at [0, 0, 0] which is the robot base
            x = radius * np.cos(elevation) * np.cos(azimuth)
            y = radius * np.cos(elevation) * np.sin(azimuth)
            z = radius * np.sin(elevation)
            
        elif sampling_method == 2:
            # METHOD 3: Grid-based sampling
            # Create a grid of positions throughout the workspace
            grid_size = 5  # 5x5x5 grid
            
            # Sample grid indices
            i = np.random.randint(0, grid_size)
            j = np.random.randint(0, grid_size)
            k = np.random.randint(0, grid_size)
            
            # Convert to coordinates - centered at [0,0,0]
            x = -0.6 + (i / (grid_size-1)) * 1.2  # -0.6 to 0.6
            y = -0.6 + (j / (grid_size-1)) * 1.2  # -0.6 to 0.6
            z = 0.1 + (k / (grid_size-1)) * 0.6   # 0.1 to 0.7
            
        else:  # sampling_method == 3
            # METHOD 4: Uniform sampling in a cube
            x = np.random.uniform(-0.6, 0.6)
            y = np.random.uniform(-0.6, 0.6)
            z = np.random.uniform(0.1, 0.7)
        
        # Ensure z is at least 0.05m above the ground
        z = max(z, 0.05)
        
        # Calculate distance from robot base
        distance = np.sqrt(x*x + y*y + z*z)
        
        # Ensure the point is not too close to the robot
        if distance < min_distance:
            # Scale the position to be at minimum distance
            scale_factor = min_distance / distance
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
            distance = min_distance  # Update distance after scaling
        
        # Ensure the point is within the maximum reach
        if distance > max_reach:
            # Scale the position to be within max_reach
            scale_factor = max_reach / distance * 0.95  # 95% of max reach for safety
            x *= scale_factor
            y *= scale_factor
            z *= scale_factor
        
        return np.array([x, y, z])
    
    def _enhance_observation(self, state):
        """
        Enhance the observation with relative position between end effector and target.
        This helps the model better understand the spatial relationship.
        """
        # Extract end-effector position from state
        ee_position = state[12:15]
        
        # Calculate relative position (target - ee)
        relative_position = self.target_position - ee_position
        
        # Create enhanced observation by appending relative position
        # Original state: 19 dimensions
        # Enhanced state: 22 dimensions (19 original + 3 relative position)
        enhanced_state = np.concatenate([state, relative_position])
        
        return enhanced_state
    
    def render(self, mode='human'):
        self.robot_env.render()
    
    def close(self):
        self.robot_env.close()