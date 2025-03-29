"""
Custom action spaces for robot control that enforce joint limits.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

def ensure_joint_limits(robot, joint_positions):
    """
    Strictly enforce joint limits from the URDF model.
    
    Args:
        robot: The robot environment instance
        joint_positions: Array of joint positions to check and enforce
        
    Returns:
        Array of joint positions with limits enforced
    """
    limited_positions = joint_positions.copy()
    
    for i, pos in enumerate(joint_positions):
        if i in robot.joint_limits:
            limit_low, limit_high = robot.joint_limits[i]
            # Strictly enforce limits
            if pos < limit_low:
                limited_positions[i] = limit_low
                print(f"WARNING: Joint {i} below limit ({pos:.4f} < {limit_low:.4f}), enforcing limit")
            elif pos > limit_high:
                limited_positions[i] = limit_high
                print(f"WARNING: Joint {i} above limit ({pos:.4f} > {limit_high:.4f}), enforcing limit")
    
    return limited_positions

class JointLimitedBox(spaces.Box):
    """
    A gym.spaces.Box variant that inherently respects joint limits.
    
    This space represents actions as normalized values in [-1, 1], which are then
    mapped to the corresponding joint limits. This ensures that any action sampled
    from this space is always within the physical limits of the robot.
    
    Additional features:
    - Provides metadata about joint limits for policy networks
    - Implements custom sampling that respects safe margins near limits
    - Includes helper methods for action normalization/unnormalization
    """
    def __init__(self, robot, shape=(5,), dtype=np.float32):
        """
        Initialize JointLimitedBox action space.
        
        Args:
            robot: The FANUCRobotEnv instance with joint_limits dictionary
            shape: The shape of the action space (default is 5 for the FANUC robot)
            dtype: The data type of the action space
        """
        super().__init__(low=-1.0, high=1.0, shape=shape, dtype=dtype)
        self.robot = robot
        
        # Store joint limits for quick access
        self.joint_limits = {}
        for i in range(shape[0]):
            if i in robot.joint_limits:
                self.joint_limits[i] = robot.joint_limits[i]
        
        # Calculate midpoints and ranges for each joint for efficient unnormalization
        self.joint_mids = {}
        self.joint_ranges = {}
        
        for i, limits in self.joint_limits.items():
            low, high = limits
            self.joint_mids[i] = (high + low) / 2.0
            self.joint_ranges[i] = (high - low) / 2.0
            
    def sample(self):
        """
        Sample a random action while respecting joint limits.
        
        Returns:
            A random action in the [-1, 1] range for each joint
        """
        # Sample slightly away from limits to avoid numerical issues
        safe_margin = 0.05  # 5% safety margin
        sample = np.random.uniform(
            low=self.low + safe_margin,
            high=self.high - safe_margin,
            size=self.shape
        ).astype(self.dtype)
        
        return sample
    
    def contains(self, x):
        """Check if x is contained in the space"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=self.dtype)
            
        return bool(np.all(x >= self.low) and np.all(x <= self.high))
    
    def normalize_action(self, joint_positions):
        """
        Convert joint positions to normalized actions in [-1, 1] range.
        
        Args:
            joint_positions: Array of joint positions in radians
            
        Returns:
            Normalized actions in [-1, 1] range
        """
        normalized = np.zeros(len(joint_positions), dtype=self.dtype)
        
        for i, pos in enumerate(joint_positions):
            if i in self.joint_limits:
                low, high = self.joint_limits[i]
                mid = self.joint_mids[i]
                range_half = self.joint_ranges[i]
                
                # Normalize to [-1, 1]
                normalized[i] = (pos - mid) / range_half
                
                # Ensure normalization exactly matches our bounds
                normalized[i] = np.clip(normalized[i], -1.0, 1.0)
            else:
                # Default for joints without limits
                normalized[i] = 0.0
                
        return normalized
    
    def unnormalize_action(self, normalized_action):
        """
        Convert normalized actions in [-1, 1] range to joint positions in radians.
        
        Args:
            normalized_action: Array of normalized actions in [-1, 1] range
            
        Returns:
            Joint positions in radians
        """
        joint_positions = np.zeros(len(normalized_action), dtype=np.float32)
        
        for i, norm_action in enumerate(normalized_action):
            if i in self.joint_limits:
                # Clip to ensure we stay within normalized range
                norm_action_clipped = np.clip(norm_action, -1.0, 1.0)
                
                # Convert from [-1, 1] to actual joint position
                mid = self.joint_mids[i]
                range_half = self.joint_ranges[i]
                
                # Compute joint position
                joint_positions[i] = mid + norm_action_clipped * range_half
            else:
                # Default for joints without limits (should not happen)
                joint_positions[i] = 0.0
                
        return joint_positions

class JointLimitEnforcingEnv(gym.Wrapper):
    """
    Environment wrapper that works with JointLimitedBox action space to enforce joint limits.
    With JointLimitedBox, this wrapper is mostly for backward compatibility and monitoring.
    """
    def __init__(self, env):
        super().__init__(env)
        self.robot = env.robot
        
        # Check if we're using the JointLimitedBox action space
        if isinstance(env.action_space, JointLimitedBox):
            print("Using JointLimitedBox action space - joint limits are inherently enforced")
            self.using_joint_limited_box = True
        else:
            print("LEGACY MODE: Joint limits will be enforced by the environment wrapper")
            self.using_joint_limited_box = False
    
    def step(self, action):
        # If using JointLimitedBox, the limits are already enforced by the action space
        if self.using_joint_limited_box:
            # Just pass the action through to the underlying environment
            return self.env.step(action)
        
        # Legacy mode: manually enforce limits
        else:
            # Extract the underlying robot action from the environment's step method
            if hasattr(self.env, 'robot') and hasattr(self.env.robot, 'step'):
                # Get current joint positions
                state = self.robot._get_state()
                current_joint_positions = state[:self.robot.dof*2:2]
                
                # For delta joint position control, calculate new positions
                new_joint_positions = []
                for i, delta in enumerate(action):
                    # Current position plus delta
                    new_pos = current_joint_positions[i] + delta
                    new_joint_positions.append(new_pos)
                
                # Enforce joint limits
                limited_positions = ensure_joint_limits(self.robot, new_joint_positions)
                
                # Create zero velocities for the robot step
                zero_velocities = [0.0] * len(limited_positions)
                
                # Call the original step method with enforced limits
                next_state = self.robot.step((limited_positions, zero_velocities))
                
                # Pass to parent step with the original action (the environment will re-apply limits internally)
                return self.env.step(action)
            else:
                # Fallback if the environment doesn't match our expected structure
                return self.env.step(action) 