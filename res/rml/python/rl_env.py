# update_rl_env.py
import gym
from gym import spaces
import numpy as np
from domain_randomization import DomainRandomizedEnv

class DomainRandomizedRLEnv(gym.Env):
    def __init__(self, render=False, randomize=True):
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
        
        # Observations: joint positions, velocities, end-effector pose
        # 6 joint positions + 6 joint velocities + 3 EE position + 4 EE orientation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        
        # Target position for the task
        self.target_position = np.array([0.5, 0.0, 0.5])  # Example target
        
    def reset(self):
        # Reset the robot environment (includes domain randomization)
        state = self.robot_env.reset()
        
        # Set a new random target within the workspace
        self.target_position = self._sample_target()
        
        return state
    
    def step(self, action):
        # Take a step in the environment
        next_state, _, _, _ = self.robot_env.step(action)
        
        # Calculate reward based on distance to target
        ee_position = next_state[12:15]  # Extract end-effector position from state
        distance = np.linalg.norm(ee_position - self.target_position)
        
        # Reward is negative distance (closer is better)
        # Add a small penalty for large actions to encourage smooth motion
        action_penalty = 0.01 * np.sum(np.square(action))
        reward = -distance - action_penalty
        
        # Task is done if we're close enough to the target
        done = distance < 0.05
        
        # Additional info
        info = {"distance": distance}
        
        return next_state, reward, done, info
    
    def _sample_target(self):
        # Sample a random target position within the robot's workspace
        # This is a simplified version - you should adjust based on the actual workspace
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.2, 0.7)
        return np.array([x, y, z])
    
    def render(self, mode='human'):
        self.robot_env.render()
    
    def close(self):
        self.robot_env.close()