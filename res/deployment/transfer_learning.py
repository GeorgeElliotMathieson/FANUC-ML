# transfer_learning.py
import torch
import numpy as np
from stable_baselines3 import PPO

class RobotTransfer:
    def __init__(self, model_path="./models/fanuc_final_model.zip"):
        # Load the trained model
        self.model = PPO.load(model_path)
        
        # State normalization parameters (can be updated during real-world calibration)
        self.state_mean = np.zeros(19)  # Placeholder for mean
        self.state_std = np.ones(19)    # Placeholder for std
        
        # Action correction parameters
        self.action_offset = np.zeros(6)  # Placeholder for offset
        self.action_scale = np.ones(6)    # Placeholder for scale
    
    def calibrate(self, sim_states, real_states, sim_actions, real_actions):
        """
        Calibrate transfer parameters using paired sim and real data
        
        Args:
            sim_states: Array of simulation states
            real_states: Array of corresponding real robot states
            sim_actions: Array of simulation actions
            real_actions: Array of corresponding real robot actions
        """
        # Update state normalization
        if len(sim_states) > 0 and len(real_states) > 0:
            # Calculate mean and std of real states
            self.state_mean = np.mean(real_states, axis=0)
            self.state_std = np.std(real_states, axis=0) + 1e-8  # Add small epsilon to avoid div by zero
        
        # Update action correction
        if len(sim_actions) > 0 and len(real_actions) > 0:
            # Simple linear regression to map sim actions to real actions
            for i in range(6):
                sim_a = sim_actions[:, i]
                real_a = real_actions[:, i]
                
                # Linear regression: real_a = scale * sim_a + offset
                A = np.vstack([sim_a, np.ones(len(sim_a))]).T
                scale, offset = np.linalg.lstsq(A, real_a, rcond=None)[0]
                
                self.action_scale[i] = scale
                self.action_offset[i] = offset
    
    def normalize_state(self, state):
        """Normalize a state from the real robot to match simulation distribution"""
        normalized_state = (state - self.state_mean) / self.state_std
        return normalized_state
    
    def correct_action(self, action):
        """Correct an action from simulation to work on the real robot"""
        corrected_action = action * self.action_scale + self.action_offset
        return corrected_action
    
    def predict(self, state):
        """
        Predict an action for the real robot
        
        Args:
            state: State observation from the real robot
            
        Returns:
            corrected_action: Action corrected for the real robot
        """
        # Normalize the state
        normalized_state = self.normalize_state(state)
        
        # Get prediction from the model
        action, _ = self.model.predict(normalized_state, deterministic=True)
        
        # Correct the action
        corrected_action = self.correct_action(action)
        
        return corrected_action