# domain_rand.py
import numpy as np
from robot_sim import FANUCRobotEnv

class DomainRandomizedEnv(FANUCRobotEnv):
    def __init__(self, render=True, randomize=True):
        super(DomainRandomizedEnv, self).__init__(render=render)
        self.randomize = randomize
        
        # Default parameters
        self.default_params = {
            'joint_damping': 0.1,
            'joint_friction': 0.1,
            'link_mass': 1.0,
            'gravity': -9.81,
            'link_inertia': 0.1,
            'sensor_noise': 0.01
        }
        
        # Current parameters
        self.current_params = self.default_params.copy()
        
        # Apply initial parameters
        self._apply_parameters()
        
    def reset(self):
        # Randomize parameters if enabled
        if self.randomize:
            self._randomize_parameters()
            
        # Apply parameters
        self._apply_parameters()
        
        # Reset the environment
        return super().reset()
    
    def _randomize_parameters(self):
        """Randomize physical parameters"""
        # Randomize joint properties
        self.current_params['joint_damping'] = np.random.uniform(0.05, 0.3)
        self.current_params['joint_friction'] = np.random.uniform(0.05, 0.3)
        
        # Randomize link properties
        self.current_params['link_mass'] = np.random.uniform(0.8, 1.2)
        self.current_params['link_inertia'] = np.random.uniform(0.08, 0.12)
        
        # Randomize gravity
        self.current_params['gravity'] = np.random.uniform(-10.5, -9.0)
        
        # Randomize sensor noise
        self.current_params['sensor_noise'] = np.random.uniform(0.0, 0.05)
    
    def _apply_parameters(self):
        """Apply current parameters to the simulation"""
        import pybullet as p
        
        # Set gravity
        p.setGravity(0, 0, self.current_params['gravity'], physicsClientId=self.client)
        
        # Set joint properties
        for joint_idx in range(self.dof):
            p.changeDynamics(
                self.robot_id,
                joint_idx,
                linearDamping=self.current_params['joint_damping'],
                angularDamping=self.current_params['joint_damping'],
                jointDamping=self.current_params['joint_damping'],
                lateralFriction=self.current_params['joint_friction'],
                mass=self.current_params['link_mass'],
                physicsClientId=self.client
            )
    
    def _get_state(self):
        """Get state with optional noise"""
        # Get clean state from parent class
        clean_state = super()._get_state()
        
        # Add noise to sensor readings if randomization is enabled
        if self.randomize:
            noise_scale = self.current_params['sensor_noise']
            noise = np.random.normal(0, noise_scale, size=clean_state.shape)
            noisy_state = clean_state + noise
            return noisy_state
        else:
            return clean_state