# train_robot_visual.py - Training script with visualization for FANUC robot arm
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import multiprocessing
import sys
import traceback
import pybullet as p

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a robot arm with visualization')
parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
parser.add_argument('--steps', type=int, default=500000, help='Total number of training steps')
parser.add_argument('--target', type=float, default=2.0, help='Target accuracy in cm')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
parser.add_argument('--load', type=str, default='', help='Load a pre-trained model to continue training')
parser.add_argument('--eval-only', action='store_true', help='Only run evaluation on a pre-trained model')
parser.add_argument('--gui-delay', type=float, default=0.05, help='Delay between steps for better visualization (seconds)')
parser.add_argument('--random-targets', action='store_true', help='Use random targets during training instead of fixed target')
parser.add_argument('--no-randomize-every-step', action='store_true', help='Disable randomizing target position for every step')
parser.add_argument('--eval-fixed-sequence', action='store_true', help='Use a fixed sequence of targets for evaluation instead of random ones')
parser.add_argument('--fullscreen', action='store_true', help='Enable fullscreen mode with zoomed-in view of the robot')
parser.add_argument('--parallel', type=int, default=0, help='Number of parallel environments (0=auto)')
parser.add_argument('--workspace-size', type=float, default=0.7, help='Size of the workspace for target positions')
parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for the optimizer')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')
parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
parser.add_argument('--ent-coef', type=float, default=0.3, help='Entropy coefficient')
parser.add_argument('--n-steps', type=int, default=256, help='Number of steps per update')
parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size for updates')
parser.add_argument('--n-epochs', type=int, default=8, help='Number of epochs when optimizing the surrogate loss')
parser.add_argument('--sim-steps-per-action', type=int, default=24, help='Number of simulation steps per action (higher values give the robot more time to position itself)')
args = parser.parse_args()

# Enable debug mode
debug_mode = args.debug
target_accuracy_cm = args.target
target_accuracy_m = target_accuracy_cm / 100.0  # Convert to meters
gui_delay = args.gui_delay  # Delay for better visualization
use_random_targets = args.random_targets  # Whether to use random targets
randomize_every_step = not args.no_randomize_every_step  # Randomize target position for every step by default
use_eval_fixed_sequence = args.eval_fixed_sequence  # Whether to use a fixed sequence of targets for evaluation
use_fullscreen = args.fullscreen  # Whether to use fullscreen mode
workspace_size = args.workspace_size  # Size of the workspace
sim_steps_per_action = 24  # Always use 24 simulation steps per action regardless of command line argument

# Determine number of parallel environments
if args.eval_only or args.parallel == 1:
    # For evaluation or when explicitly specified, use only 1 environment
    n_parallel_envs = 1
    print("Using 1 environment as specified")
elif randomize_every_step and args.parallel <= 1:
    # For randomizing targets every step with visualization, use 1 environment
    # Changed from 2 to 1 to ensure visualization works properly
    n_parallel_envs = 1
    print("Using 1 environment for randomized targets with visualization")
elif args.parallel > 1:
    # Use the specified number of parallel environments
    n_parallel_envs = args.parallel
    print(f"Using {n_parallel_envs} parallel environments as specified")
else:
    # Auto-detect based on CPU cores with optimized allocation
    cpu_count = multiprocessing.cpu_count()
    
    # Always reserve at least 4 cores for system processes as requested
    if cpu_count >= 8:
        # For systems with 8+ cores, use all but 4 cores
        n_parallel_envs = cpu_count - 4
    else:
        # For systems with fewer than 8 cores, use half the cores
        n_parallel_envs = max(1, cpu_count // 2)
    
    print(f"Auto-detected {cpu_count} CPU cores, using {n_parallel_envs} parallel environments (reserving at least 4 cores for system processes)")

# Optimize torch for better parallelization
if torch.cuda.is_available():
    # If CUDA is available, use it but with optimized settings
    device = torch.device("cuda")
    # Set optimal CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print(f"Using GPU acceleration with optimized settings: {torch.cuda.get_device_name(0)}")
else:
    # For CPU, optimize thread settings
    torch.set_num_threads(max(4, n_parallel_envs))
    torch.set_num_interop_threads(max(4, n_parallel_envs // 2))
    device = torch.device("cpu")
    print(f"Using CPU with optimized thread settings: {torch.get_num_threads()} compute threads, {torch.get_num_interop_threads()} interop threads")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import environment
from rl_env import DomainRandomizedRLEnv

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Custom network architecture with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_channels, n_channels * 6),  # Increased from 4x to 6x
            nn.ReLU(),
            nn.Linear(n_channels * 6, n_channels * 3),  # Increased intermediate layer
            nn.ReLU(),
            nn.Linear(n_channels * 3, n_channels)  # Output layer
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.network(x)
        out += identity  # Residual connection
        return self.relu(out)

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=2048):  # Increased from 1024 to 2048
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        
        # Input size from observation space
        n_input = int(np.prod(observation_space.shape))
        
        # Separate networks for processing joint states and end-effector states
        # Increased hidden dimensions from 256 to 512
        self.joint_network = nn.Sequential(
            nn.Linear(6, 512),  # Process 6 joint positions - increased from 256 to 512
            nn.ReLU(),
            ResidualBlock(512),  # Increased from 256 to 512
            ResidualBlock(512),  # Kept second residual block
            ResidualBlock(512)   # Added third residual block
        )
        
        self.joint_vel_network = nn.Sequential(
            nn.Linear(6, 512),  # Process 6 joint velocities - increased from 256 to 512
            nn.ReLU(),
            ResidualBlock(512),  # Increased from 256 to 512
            ResidualBlock(512),  # Kept second residual block
            ResidualBlock(512)   # Added third residual block
        )
        
        self.ee_network = nn.Sequential(
            nn.Linear(7, 512),  # Process end-effector position (3) and orientation (4) - increased from 256 to 512
            nn.ReLU(),
            ResidualBlock(512),  # Increased from 256 to 512
            ResidualBlock(512),  # Kept second residual block
            ResidualBlock(512)   # Added third residual block
        )
        
        # Combine all features
        self.combined_network = nn.Sequential(
            nn.Linear(512 * 3, 2048),  # Combine joint, velocity, and ee features - increased from 1024 to 2048
            nn.ReLU(),
            ResidualBlock(2048),
            ResidualBlock(2048),
            ResidualBlock(2048),
            ResidualBlock(2048),  # Added an extra residual block
            nn.Linear(2048, features_dim)
        )
        
    def forward(self, observations):
        # Split observation into components
        joint_pos = observations[:, :6]  # Joint positions
        joint_vel = observations[:, 6:12]  # Joint velocities
        ee_state = observations[:, 12:19]  # End-effector position and orientation
        
        # Process each component
        joint_features = self.joint_network(joint_pos)
        vel_features = self.joint_vel_network(joint_vel)
        ee_features = self.ee_network(ee_state)
        
        # Combine features
        combined = torch.cat([joint_features, vel_features, ee_features], dim=1)
        return self.combined_network(combined)

# Custom policy kwargs with our network architecture
policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    features_extractor_kwargs=dict(features_dim=2048),  # Increased from 1024 to 2048
    net_arch=dict(
        pi=[1024, 512, 512, 256],  # Policy network - increased from [512, 512, 256] to [1024, 512, 512, 256]
        vf=[1024, 512, 512, 256]   # Value network - increased from [512, 512, 256] to [1024, 512, 512, 256]
    ),
    activation_fn=nn.ReLU
)

# Function to visualize target position in PyBullet
def visualize_target(target_position, client_id, replace_id=None):
    """
    Create a visual marker for the target position in PyBullet
    
    Args:
        target_position: [x, y, z] position of the target
        client_id: PyBullet physics client ID
        replace_id: ID of existing visual to replace (or None to create new)
        
    Returns:
        ID of the created visual marker
    """
    # Define visual properties
    radius = 0.05  # Further increased size for better visibility
    color = [0.0, 0.3, 1.0, 1.0]  # Bright blue with full opacity for better distinction
    
    # If we're replacing an existing visual, remove it first
    if replace_id is not None:
        try:
            p.removeBody(replace_id, physicsClientId=client_id)
            # Add a small delay to ensure the removal completes
            time.sleep(0.01)
        except:
            pass
    
    # Create a visual sphere at the target position
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id
    )
    
    # Create a body with the visual shape but no mass (visual only)
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=target_position,
        physicsClientId=client_id
    )
    
    return target_id

# Function to configure PyBullet camera for fullscreen view of the robot
def configure_camera_fullscreen(client_id):
    """
    Configure the PyBullet camera for a fullscreen view of the robot
    
    Args:
        client_id: PyBullet physics client ID
    """
    # Set the debug visualizer to use the full window if fullscreen is enabled
    if use_fullscreen:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_FULLSCREEN, 1, physicsClientId=client_id)
    else:
        # Still hide the GUI panel for a cleaner view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
    
    # Position the camera to focus on the robot
    # Parameters: target position, distance from target, yaw, pitch, roll, up-axis
    target_pos = [0.5, 0.0, 0.3]  # Center of the workspace
    
    # Closer view if fullscreen is enabled
    camera_distance = 0.8 if use_fullscreen else 1.2
    
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=45,        # View angle around z-axis
        cameraPitch=-20,     # View angle from horizontal plane
        cameraTargetPosition=target_pos,
        physicsClientId=client_id
    )
    
    # Additional visualization settings for better view
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)
    
    if use_fullscreen:
        print("Camera configured for fullscreen view of the robot")
    else:
        print("Camera configured for zoomed-in view of the robot")

# Function to visualize end-effector trajectory
def visualize_ee_position(ee_position, client_id, color=[0, 1, 0, 0.7], radius=0.01):
    """
    Create a visual marker for the end-effector position to track its trajectory
    
    Args:
        ee_position: [x, y, z] position of the end-effector
        client_id: PyBullet physics client ID
        color: RGBA color for the marker
        radius: Size of the marker
        
    Returns:
        ID of the created visual marker
    """
    # Create a visual sphere at the end-effector position
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id
    )
    
    # Create a body with the visual shape but no mass (visual only)
    marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=ee_position,
        physicsClientId=client_id
    )
    
    return marker_id

# Enhanced reward calculation function
def calculate_reward(ee_position, target_position, action, prev_distance=None, joint_positions=None):
    """
    Calculate a simplified reward with a strong gradient for closeness
    
    Args:
        ee_position: Current end-effector position
        target_position: Target position
        action: Action taken
        prev_distance: Previous distance to target (if available)
        joint_positions: Current joint positions (if available)
        
    Returns:
        Calculated reward value and components dictionary
    """
    # Calculate distance to target
    distance = np.linalg.norm(ee_position - target_position)
    
    # Base reward is exponentially scaled inverse distance
    # Creates an extremely strong gradient as the robot gets closer
    # The closer to the target, the exponentially higher the reward
    # Increased exponent from 8.0 to 10.0 for even stronger gradient
    base_reward = 20.0 * np.exp(10.0 * (1.0 - distance))
    
    # Progress reward - heavily reward moving closer to the target
    progress_reward = 0.0
    if prev_distance is not None:
        # Calculate how much closer we got to the target
        progress = prev_distance - distance
        
        # Scale progress reward exponentially based on current distance
        # This creates a stronger gradient as we get closer
        # Increased exponent from 5.0 to 10.0 for stronger gradient
        progress_scale = 100.0 * np.exp(10.0 * (1.0 - distance))
        progress_reward = progress * progress_scale
    
    # Combine all reward components
    reward = base_reward + progress_reward
    
    # Success is determined by distance only
    success = distance < 0.06  # 6cm threshold for success
    
    return reward, {
        'distance': distance,
        'base_reward': base_reward,
        'progress_reward': progress_reward,
        'success': success
    }

# Training callback that shows progress and evaluates accuracy
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, global_step_offset=0):
        super(TrainingCallback, self).__init__(verbose)
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_steps = 0
        self.best_accuracy = float('inf')
        self.progress_interval = 10  # Show progress every 10 steps
        # Fixed target position for evaluation (center of workspace)
        self.target_position = np.array([0.5, 0.0, 0.5])
        # ID for the target visualization
        self.target_visual_id = None
        # For tracking trajectory
        self.trajectory_markers = []
        self.max_trajectory_markers = args.n_steps  # Set directly to the actual batch size from args
        # For tracking the best distance during training
        self.current_distance = float('inf')
        self.best_distance_this_episode = float('inf')
        self.episode_step_count = 0
        # Store the last observation
        self.last_observation = None
        # For tracking average distance
        self.distance_sum = 0.0
        self.distance_count = 0
        self.avg_distance = float('inf')
        self.best_avg_distance = float('inf')
        # For tracking the final average before each weight update
        self.last_final_avg = float('inf')
        self.best_final_avg = float('inf')
        # For tracking performance changes between weight updates
        self.prev_final_avg = float('inf')
        self.performance_improved = False
        self.consecutive_improvements = 0
        self.consecutive_deteriorations = 0
        # For dynamic batch size and simulation steps adjustment
        self.current_batch_size = args.batch_size
        # List to store all distances for statistics
        self.distances = []
        # Flag to indicate if target accuracy was reached
        self.target_reached = False
        # Global step counter offset for continuous numbering
        self.global_step_offset = global_step_offset
        # Flag to track if we've initialized the target visualization
        self.target_initialized = False
        # Store previous distance for reward calculation
        self.prev_distance = None
        # For tracking reward components
        self.reward_components = {
            'base_reward': 0.0,
            'progress_reward': 0.0,
            'success': False
        }
        # For tracking success rate
        self.success_count = 0
        self.episode_count = 0
        self.success_rate = 0.0
        # For tracking progress percentage
        self.progress_percentage = 0.0
        # For tracking minimum distance in episode
        self.min_distance_in_episode = float('inf')
        # Visualization update frequency - ensure target and trajectory are updated at same rate
        self.visualization_update_freq = 1  # Update visualization every step
        # Track when weights were last updated to clear trajectory markers
        self.last_weights_update_step = 0
        # Flag to ensure camera is configured
        self.camera_configured = False
        # Store the PyBullet client ID for visualization
        self.client_id = None
        # Flag to track if visualization is working
        self.visualization_working = False
        # For adaptive entropy coefficient - will be properly initialized in _init_callback
        self.initial_ent_coef = 0.3  # Default value, will be updated from model in _init_callback
        self.current_ent_coef = 0.3  # Default value, will be updated from model in _init_callback
        # Add caps for entropy coefficient
        self.min_ent_coef = 0.05  # Minimum entropy coefficient (increased from 0.001)
        self.max_ent_coef = 1.0    # Maximum entropy coefficient (updated to 1.0 as standard range for entropy)
        # Accuracy thresholds for entropy adjustment
        self.excellent_accuracy = 0.10  # 10cm
        self.good_accuracy = 0.20  # 20cm
        self.poor_accuracy = 0.40  # 40cm
        # For tracking performance plateaus
        self.plateau_detection_window = 5  # Number of weight updates to consider for plateau detection
        self.recent_final_avgs = []  # Store recent final averages
        self.plateau_detected = False
        self.plateau_counter = 0
        self.forced_exploration_phase = False
        self.forced_exploration_duration = 3  # Number of weight updates to maintain forced exploration
        # For periodic forced exploration (regardless of performance)
        self.update_counter = 0
        self.forced_exploration_frequency = 5  # Force exploration every N weight updates
        self.initial_forced_exploration_frequency = 5  # Store initial value
        self.exploration_intensity = 1.0  # Multiplier for exploration intensity
        
    def _init_callback(self):
        """
        Initialize callback with proper values from the model.
        This is called when the callback is first created.
        """
        super()._init_callback()
        
        # Initialize entropy coefficient from the model
        if hasattr(self.model, 'ent_coef'):
            self.initial_ent_coef = self.model.ent_coef
            self.current_ent_coef = self.model.ent_coef
            if debug_mode:
                print(f"Initialized entropy coefficient from model: {self.initial_ent_coef}")
        else:
            # Fallback to command line argument
            self.initial_ent_coef = args.ent_coef
            self.current_ent_coef = args.ent_coef
            if debug_mode:
                print(f"Initialized entropy coefficient from args: {self.initial_ent_coef}")

    def _update_entropy_coefficient(self):
        """
        Update the entropy coefficient based on the current accuracy, exploration status,
        and performance plateau detection.
        """
        if not hasattr(self.model, 'ent_coef') or self.avg_distance == float('inf'):
            return
            
        # Check if exploration is triggered in any environment
        exploration_triggered = False
        if self.locals is not None and 'infos' in self.locals and len(self.locals['infos']) > 0:
            exploration_triggered = self.locals['infos'][0].get('exploration_mode', False)
        
        # Check for performance plateau
        if len(self.recent_final_avgs) >= self.plateau_detection_window:
            # Calculate the standard deviation of recent performances
            if len(self.recent_final_avgs) > 0:
                avg = sum(self.recent_final_avgs) / len(self.recent_final_avgs)
                variance = sum((x - avg) ** 2 for x in self.recent_final_avgs) / len(self.recent_final_avgs)
                std_dev = variance ** 0.5
                
                # If standard deviation is very small, we're in a plateau
                plateau_threshold = 0.02  # 2cm standard deviation threshold (increased from 1cm)
                if std_dev < plateau_threshold and not self.forced_exploration_phase:
                    self.plateau_counter += 1
                    if self.plateau_counter >= 1:  # Require only 1 detection (reduced from 2)
                        self.plateau_detected = True
                        self.forced_exploration_phase = True
                        self.plateau_counter = 0
                        # Increase exploration intensity each time we detect a plateau
                        self.exploration_intensity = min(3.0, self.exploration_intensity * 1.2)
                        if debug_mode:
                            print(f"PLATEAU DETECTED! Standard deviation: {std_dev*100:.4f}cm. Forcing exploration phase with intensity {self.exploration_intensity:.2f}.")
                else:
                    self.plateau_counter = 0
        
        # Check for periodic forced exploration
        if hasattr(self.model, 'n_steps') and self.n_calls % self.model.n_steps == 0 and self.n_calls > 0:
            self.update_counter += 1
            if self.update_counter >= self.forced_exploration_frequency and not self.forced_exploration_phase:
                self.forced_exploration_phase = True
                self.update_counter = 0
                # Gradually increase exploration frequency as training progresses
                self.forced_exploration_frequency = min(20, self.initial_forced_exploration_frequency + self.update_counter // 10)
                if debug_mode:
                    print(f"PERIODIC FORCED EXPLORATION triggered. Next forced exploration in {self.forced_exploration_frequency} updates.")
        
        # If we're in forced exploration phase, maintain high entropy
        if self.forced_exploration_phase:
            # Use a high entropy coefficient during forced exploration, scaled by exploration intensity
            target_ent_coef = min(self.max_ent_coef, max(0.5, self.current_ent_coef * (1.5 * self.exploration_intensity)))
            
            # Use a faster update rate for forced exploration
            update_rate = 0.7  # 70% update rate for quick adaptation (increased from 50%)
            self.current_ent_coef = (1 - update_rate) * self.current_ent_coef + update_rate * target_ent_coef
            
            # Decrement the forced exploration duration counter
            self.forced_exploration_duration -= 1
            if self.forced_exploration_duration <= 0:
                self.forced_exploration_phase = False
                self.forced_exploration_duration = 3  # Reset for next time
                if debug_mode:
                    print("Forced exploration phase ended.")
        # If exploration is triggered, increase entropy to encourage exploration
        elif exploration_triggered:
            # Increase entropy coefficient to encourage exploration - with cap
            target_ent_coef = min(self.max_ent_coef, self.current_ent_coef * 2.0)  # Double the current entropy, but cap it
            
            # Use consistent update rate for both increase and decrease
            update_rate = 0.3  # 30% update rate for both increase and decrease
            self.current_ent_coef = (1 - update_rate) * self.current_ent_coef + update_rate * target_ent_coef
        else:
            # Calculate the new entropy coefficient based on average distance
            # More balanced scaling with appropriate caps
            if self.avg_distance < self.excellent_accuracy:
                # Very accurate - reduce entropy but maintain minimum exploration
                target_ent_coef = max(self.min_ent_coef, self.current_ent_coef * 0.8)  # Reduce by 20% but maintain minimum (changed from 30%)
            elif self.avg_distance < self.good_accuracy:
                # Good accuracy - slight reduction
                ratio = (self.avg_distance - self.excellent_accuracy) / (self.good_accuracy - self.excellent_accuracy)
                reduction_factor = 0.8 + 0.1 * ratio  # Scale from 0.8 to 0.9 (changed from 0.7-0.9)
                target_ent_coef = max(self.min_ent_coef, self.current_ent_coef * reduction_factor)
            elif self.avg_distance < self.poor_accuracy:
                # Moderate accuracy - maintain or slightly increase
                ratio = (self.avg_distance - self.good_accuracy) / (self.poor_accuracy - self.good_accuracy)
                scale_factor = 1.0 + 0.5 * ratio  # Scale from 1.0 to 1.5
                target_ent_coef = min(self.max_ent_coef, self.current_ent_coef * scale_factor)
            else:
                # Poor accuracy - increase entropy with cap
                # Cap the scaling factor to avoid extreme values
                max_scale_factor = 2.0
                scale_factor = min(max_scale_factor, 1.5 + (self.avg_distance - self.poor_accuracy))
                target_ent_coef = min(self.max_ent_coef, self.current_ent_coef * scale_factor)
            
            # Use consistent update rate for both increase and decrease
            update_rate = 0.3  # 30% update rate for both increase and decrease
            self.current_ent_coef = (1 - update_rate) * self.current_ent_coef + update_rate * target_ent_coef
        
        # Final safety check to ensure entropy stays within bounds
        self.current_ent_coef = max(self.min_ent_coef, min(self.max_ent_coef, self.current_ent_coef))
        
        # Update the model's entropy coefficient
        self.model.ent_coef = self.current_ent_coef
        
        if debug_mode and self.n_calls % 100 == 0:
            exploration_status = "FORCED EXPLORATION" if self.forced_exploration_phase else "normal"
            print(f"Updated entropy coefficient to {self.current_ent_coef:.6f} based on avg distance: {self.avg_distance*100:.2f} cm (mode: {exploration_status})")
    
    def _adjust_dynamic_parameters(self):
        """
        Adjust dynamic parameters based on performance changes between weight updates.
        """
        if self.prev_final_avg == float('inf'):
            # First update, just store the value
            self.prev_final_avg = self.last_final_avg
            return
        
        # Calculate performance change
        if self.last_final_avg < self.prev_final_avg:
            # Performance improved
            self.performance_improved = True
            self.consecutive_improvements += 1
            self.consecutive_deteriorations = 0
        else:
            # Performance deteriorated or stayed the same
            self.performance_improved = False
            self.consecutive_improvements = 0
            self.consecutive_deteriorations += 1
        
        # Store current value for next comparison
        self.prev_final_avg = self.last_final_avg

    def _on_step(self):
        # Get the current observation to track progress
        try:
            # Get the PyBullet client ID if not already set
            if self.client_id is None:
                try:
                    # Access the underlying environment to get the PyBullet client ID
                    # For single environment or first environment in VecEnv
                    if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                        # VecEnv case
                        env = self.training_env.envs[0].robot_env
                        self.client_id = env.client
                        if debug_mode:
                            print(f"Got PyBullet client ID: {self.client_id} from VecEnv")
                    elif hasattr(self.training_env, 'robot_env'):
                        # Direct environment case
                        env = self.training_env.robot_env
                        self.client_id = env.client
                        if debug_mode:
                            print(f"Got PyBullet client ID: {self.client_id} from direct env")
                    
                    if self.client_id is not None:
                        print("Successfully connected to PyBullet visualization")
                    else:
                        print("WARNING: Could not get PyBullet client ID for visualization")
                except Exception as e:
                    if debug_mode:
                        print(f"Error getting PyBullet client ID: {e}")
                        traceback.print_exc()
            
            # Configure camera if not already done and we have a client ID
            if not self.camera_configured and self.client_id is not None:
                try:
                    # Configure camera for fullscreen view
                    configure_camera_fullscreen(self.client_id)
                    
                    self.camera_configured = True
                    print("Camera configured for visualization")
                    
                    # Test visualization by creating a temporary marker
                    test_marker_id = visualize_ee_position([0.5, 0.0, 0.5], self.client_id, color=[1, 0, 1, 1], radius=0.02)
                    if test_marker_id is not None:
                        self.visualization_working = True
                        print("Visualization test successful - markers are working")
                        # Remove the test marker after a short delay
                        time.sleep(0.1)
                        try:
                            p.removeBody(test_marker_id, physicsClientId=self.client_id)
                        except:
                            pass
                except Exception as e:
                    if debug_mode:
                        print(f"Could not configure camera: {e}")
                        traceback.print_exc()
                    
            # Get the current observation directly from the step method
            if self.locals is not None and 'new_obs' in self.locals and 'infos' in self.locals:
                obs = self.locals['new_obs']
                infos = self.locals['infos']
                self.last_observation = obs  # Store for later use
                
                # Check if weights were updated (happens every n_steps)
                # PPO updates weights after collecting n_steps of experience
                if hasattr(self.model, 'n_steps') and self.n_calls % self.model.n_steps == 0 and self.n_calls > 0:
                    # Store the final average before resetting
                    if self.distance_count > 0 and np.isfinite(self.distance_sum):
                        self.last_final_avg = self.distance_sum / self.distance_count
                        
                        # Store for plateau detection
                        self.recent_final_avgs.append(self.last_final_avg)
                        if len(self.recent_final_avgs) > self.plateau_detection_window:
                            self.recent_final_avgs.pop(0)  # Keep only the most recent values
                        
                        # Update best final average if better
                        if self.last_final_avg < self.best_final_avg:
                            # Calculate improvement percentage for uncapped reward
                            if self.best_final_avg > 0 and np.isfinite(self.best_final_avg) and np.isfinite(self.last_final_avg):
                                improvement_pct = (self.best_final_avg - self.last_final_avg) / self.best_final_avg * 100
                            else:
                                improvement_pct = 0  # Avoid division by zero or invalid values
                            self.best_final_avg = self.last_final_avg
                            
                            if debug_mode:
                                print(f"New best final average: {self.best_final_avg*100:.2f} cm (improved by {improvement_pct:.2f}%)")
                        else:
                            # Calculate deterioration percentage for uncapped penalty
                            if self.best_final_avg > 0 and np.isfinite(self.best_final_avg) and np.isfinite(self.last_final_avg):
                                deterioration_pct = (self.last_final_avg - self.best_final_avg) / self.best_final_avg * 100
                            else:
                                deterioration_pct = 0  # Avoid division by zero or invalid values
                            
                            if debug_mode and deterioration_pct > 5:
                                print(f"Performance deteriorated by {deterioration_pct:.2f}%")
                    
                    # Adjust dynamic parameters based on performance changes
                    self._adjust_dynamic_parameters()
                    
                    # Update entropy coefficient based on current accuracy
                    self._update_entropy_coefficient()
                    
                    # Clear trajectory markers when weights are updated
                    if self.client_id is not None:
                        try:
                            # Remove all trajectory markers
                            for marker_id in self.trajectory_markers:
                                try:
                                    p.removeBody(marker_id, physicsClientId=self.client_id)
                                except Exception as e:
                                    if debug_mode:
                                        print(f"Could not remove trajectory marker: {e}")
                            self.trajectory_markers = []
                            self.last_weights_update_step = self.n_calls
                            
                            # Reset average distance calculation
                            self.distance_sum = 0.0
                            self.distance_count = 0
                            self.avg_distance = float('inf')
                            
                            if debug_mode:
                                print(f"Cleared trajectory markers and reset average at step {self.n_calls} (weights update)")
                                print(f"Final average before reset: {self.last_final_avg*100:.2f} cm")
                                print(f"Using batch size of {args.n_steps} steps for trajectory visualization")
                        except Exception as e:
                            if debug_mode:
                                print(f"Could not clear trajectory markers: {e}")
                
                if obs is not None:
                    ee_position = obs[0, 12:15]  # First dimension is for the environment index in VecEnv
                    
                    # Always get the current target position from info
                    # This ensures we're using the environment's target, not our own fixed one
                    if 'target_position' in infos[0]:
                        current_target = infos[0]['target_position']
                        # Update our target position
                        self.target_position = current_target
                        
                        # Update target visualization when target changes
                        if (self.n_calls % self.visualization_update_freq == 0 or not self.target_initialized) and self.client_id is not None:
                            try:
                                # Only clear trajectory markers if target position changed significantly
                                if self.target_visual_id is not None:
                                    # Check if target position changed significantly
                                    old_target_pos = p.getBasePositionAndOrientation(self.target_visual_id, physicsClientId=self.client_id)[0]
                                    target_changed = np.linalg.norm(np.array(old_target_pos) - self.target_position) > 0.01
                                    
                                    if target_changed:
                                        # Remove the target visualization
                                        try:
                                            p.removeBody(self.target_visual_id, physicsClientId=self.client_id)
                                            # Add a small delay to ensure the removal completes
                                            time.sleep(0.01)
                                        except Exception as e:
                                            if debug_mode:
                                                print(f"Could not remove old target visualization: {e}")
                                else:
                                    target_changed = True
                                
                                # Create a new visualization with a fresh ID if needed
                                if target_changed or self.target_visual_id is None:
                                    self.target_visual_id = visualize_target(
                                        self.target_position, 
                                        self.client_id
                                    )
                                    
                                    # Mark that we've initialized the target
                                    self.target_initialized = True
                                    
                                    if debug_mode:
                                        print(f"Updated target visualization")
                                        
                                    # Add a small delay to ensure visualization updates properly
                                    time.sleep(0.01)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Could not update target visualization: {e}")
                                    traceback.print_exc()
                    
                    # Calculate distance to target
                    distance = np.linalg.norm(ee_position - self.target_position)
                    self.current_distance = distance
                    
                    # Track best distance in this episode
                    if distance < self.best_distance_this_episode:
                        self.best_distance_this_episode = distance
                    
                    # Update best overall accuracy if better
                    if distance < self.best_accuracy:
                        self.best_accuracy = distance
                    
                    # Add trajectory marker at regular intervals
                    if self.n_calls % self.visualization_update_freq == 0 and self.client_id is not None:
                        try:
                            # Normalize distance for color mapping (0-30cm range)
                            norm_dist = min(1.0, distance * 100 / 30)
                            # Green to red color gradient
                            color = [norm_dist, 1.0 - norm_dist, 0.0, 0.7]  # Increased opacity
                            
                            # Make markers smaller to reduce clutter
                            marker_radius = 0.008  # Reduced from 0.01 to 0.008 for better performance
                            
                            # Add trajectory marker
                            marker_id = visualize_ee_position(ee_position, self.client_id, color=color, radius=marker_radius)
                            self.trajectory_markers.append(marker_id)
                            
                            # Only remove oldest markers if we exceed the maximum
                            if len(self.trajectory_markers) > self.max_trajectory_markers:
                                try:
                                    p.removeBody(self.trajectory_markers.pop(0), physicsClientId=self.client_id)
                                except Exception as e:
                                    if debug_mode:
                                        print(f"Could not remove old trajectory marker: {e}")
                        except Exception as e:
                            if debug_mode:
                                print(f"Could not add trajectory marker: {e}")
                    
                    # Track distances for statistics
                    self.distances.append(distance)
                    self.distance_sum += distance
                    self.distance_count += 1
                    
                    # Calculate current average distance
                    if self.distance_count > 0 and np.isfinite(self.distance_sum):
                        self.avg_distance = self.distance_sum / self.distance_count
                        
                        # Update best average distance
                        if self.avg_distance < self.best_avg_distance:
                            self.best_avg_distance = self.avg_distance
                    
                    # Track progress percentage
                    if 'progress_percentage' in infos[0]:
                        self.progress_percentage = infos[0]['progress_percentage']
                    
                    # Track minimum distance in episode
                    if 'min_distance_in_episode' in infos[0]:
                        self.min_distance_in_episode = infos[0]['min_distance_in_episode']
                    
                    # Check for episode completion
                    if 'episode' in infos[0]:
                        # Episode has ended
                        self.episode_count += 1
                        
                        # Check if target was reached
                        if 'target_reached' in infos[0] and infos[0]['target_reached']:
                            self.success_count += 1
                        
                        # Calculate success rate
                        if self.episode_count > 0:
                            self.success_rate = self.success_count / self.episode_count
                        
                        # Reset best distance for next episode
                        self.best_distance_this_episode = float('inf')
                    
                    # Check if target accuracy was reached based on average distance
                    if self.avg_distance * 100.0 <= target_accuracy_cm:
                        self.target_reached = True
                        
                        print(f"\n{'!'*60}")
                        print(f"TARGET AVERAGE ACCURACY OF {target_accuracy_cm:.2f} cm REACHED!")
                        print(f"Current average distance: {self.avg_distance * 100.0:.2f} cm")
                        print(f"{'!'*60}")
                    
        except Exception as e:
            if debug_mode:
                print(f"Error tracking progress: {e}")
                traceback.print_exc()
        
        # Print progress at regular intervals
        if self.n_calls % self.progress_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            steps_per_second = self.n_calls / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate current average distance
            if self.distance_count > 0 and np.isfinite(self.distance_sum):
                current_avg = self.distance_sum / self.distance_count
            else:
                current_avg = float('inf')
            
            # Use global step counter for continuous numbering
            global_step = self.global_step_offset + self.n_calls
            
            # Print simplified progress with current distance and average
            if self.current_distance < float('inf'):
                # Add batch info to the output
                steps_since_update = self.n_calls - self.last_weights_update_step
                batch_progress = f"Batch: {steps_since_update}/{self.model.n_steps}" if hasattr(self.model, 'n_steps') else ""
                
                # Get consecutive no progress counter from info if available
                consecutive_no_progress = infos[0].get('consecutive_no_progress', 0) if 'infos' in self.locals and len(self.locals['infos']) > 0 else 0
                
                # Show best final average instead of best overall average and current entropy coefficient
                # Add update frequency indicator to show more frequent updates
                updates_count = self.n_calls // self.model.n_steps if hasattr(self.model, 'n_steps') else 0
                print(f"Step {global_step} | Updates: {updates_count} | Current: {self.current_distance*100:.2f} cm | Avg: {current_avg*100:.2f} cm | Best Final Avg: {self.best_final_avg*100:.2f} cm | Ent: {self.current_ent_coef:.4f} | {batch_progress}", flush=True)
            else:
                print(f"Step {global_step}", flush=True)
            
            # Add a small delay for better visualization
            if gui_delay > 0:
                time.sleep(gui_delay)
                
            # Ensure target is visualized
            if self.target_visual_id is None and self.client_id is not None:
                try:
                    # Visualize the target
                    self.target_visual_id = visualize_target(self.target_position, self.client_id)
                    if debug_mode:
                        print(f"Created target visualization")
                except Exception as e:
                    if debug_mode:
                        print(f"Could not visualize target: {e}")
                
        return True

# Environment setup helpers
def make_env(rank, seed=0, randomize=True, render=False, randomize_target=None):
    """
    Helper function to create an environment with proper settings
    
    Args:
        rank: process rank for multi-processing
        seed: random seed
        randomize: whether to use domain randomization
        render: whether to render the environment
        randomize_target: override for randomize_every_step (useful for evaluation)
    """
    def _init():
        # Use the provided randomize_target value if specified, otherwise use global setting
        use_randomize_target = randomize_target if randomize_target is not None else randomize_every_step
        
        # Create a custom environment wrapper that uses our enhanced reward function
        class EnhancedRewardEnv(DomainRandomizedRLEnv):
            def __init__(self, render=False, randomize=True, randomize_target_every_step=False):
                super(EnhancedRewardEnv, self).__init__(
                    render=render, 
                    randomize=randomize,
                    randomize_target_every_step=randomize_target_every_step
                )
                self.prev_distance = None
                self.episode_step_count = 0
                self.sim_steps_per_action = 24  # Always use 24 simulation steps per action
                # For tracking progress
                self.min_distance_in_episode = float('inf')
                self.starting_distance = None
                # For tracking consecutive steps without progress
                self.consecutive_no_progress = 0
                self.max_penalty = -10.0  # Maximum penalty cap
                self.base_penalty = -0.5  # Base penalty
                # For tracking stagnation over longer periods
                self.stagnation_steps = 0
                self.last_min_distance = float('inf')
                self.stagnation_threshold = 50  # Steps without meaningful progress
                self.stagnation_distance_threshold = 0.01  # 1cm improvement threshold
                # For forced exploration when stuck
                self.exploration_triggered = False
                self.exploration_steps_remaining = 0
                self.exploration_noise_scale = 0.3  # Scale of random noise to add during exploration
                
            def reset(self):
                state = super().reset()
                # Reset previous distance
                ee_position = state[12:15]
                self.prev_distance = np.linalg.norm(ee_position - self.target_position)
                self.episode_step_count = 0
                # Reset progress tracking
                self.min_distance_in_episode = float('inf')
                self.starting_distance = self.prev_distance
                # Reset consecutive no progress counter
                self.consecutive_no_progress = 0
                # Reset stagnation tracking
                self.stagnation_steps = 0
                self.last_min_distance = float('inf')
                # Reset exploration mode
                self.exploration_triggered = False
                self.exploration_steps_remaining = 0
                return state
                
            def step(self, action):
                # If in exploration mode, add random noise to the action
                if self.exploration_triggered and self.exploration_steps_remaining > 0:
                    # Add scaled random noise to the action
                    noise = np.random.normal(0, self.exploration_noise_scale, size=action.shape)
                    action = action + noise
                    # Clip action to valid range
                    action = np.clip(action, self.action_space.low, self.action_space.high)
                    self.exploration_steps_remaining -= 1
                    
                    # If exploration period is over, reset the flag
                    if self.exploration_steps_remaining <= 0:
                        self.exploration_triggered = False
                        if debug_mode:
                            print(f"Exploration mode ended after adding noise to actions")
                
                # Add small random noise to actions regardless of exploration mode
                # This helps prevent getting stuck in local optima
                small_noise = np.random.normal(0, 0.05, size=action.shape)  # 5% noise
                action = action + small_noise
                action = np.clip(action, self.action_space.low, self.action_space.high)
                
                # Take multiple simulation steps for each action to allow the robot
                # more time to position itself before the next action is taken
                for _ in range(self.sim_steps_per_action - 1):
                    # Take intermediate steps with the same action
                    intermediate_state, _, intermediate_done, _ = super().step(action)
                    
                    # If the episode is done in an intermediate step, return early
                    if intermediate_done:
                        break
                
                # Take the final step and get the result
                next_state, _, done, info = super().step(action)
                
                # Extract end-effector position
                ee_position = next_state[12:15]
                
                # Calculate current distance to target
                current_distance = np.linalg.norm(ee_position - self.target_position)
                
                # Check if we made progress compared to previous step
                made_progress = False
                step_penalty = 0.0
                
                if self.prev_distance is not None:
                    # We made progress if current distance is less than previous distance
                    progress = self.prev_distance - current_distance
                    made_progress = progress > 0.001  # Small threshold to account for numerical errors
                    
                    # Update consecutive no progress counter
                    if made_progress:
                        # Reset counter if we made progress
                        self.consecutive_no_progress = 0
                    else:
                        # Increment counter if we didn't make progress
                        self.consecutive_no_progress += 1
                        
                        # Calculate penalty based on consecutive steps without progress
                        # The penalty grows with cap
                        step_penalty = max(self.max_penalty, self.base_penalty * (2.0 ** min(15, self.consecutive_no_progress/2)))
                
                # Update minimum distance achieved in this episode
                if current_distance < self.min_distance_in_episode:
                    self.min_distance_in_episode = current_distance
                
                # Track stagnation over longer periods
                if self.episode_step_count % 20 == 0:  # Check every 20 steps
                    if self.last_min_distance != float('inf'):
                        # Check if we've made meaningful progress in minimum distance
                        if self.last_min_distance - self.min_distance_in_episode < self.stagnation_distance_threshold:
                            self.stagnation_steps += 20
                        else:
                            # Reset stagnation counter if we made meaningful progress
                            self.stagnation_steps = 0
                    
                    # Update last minimum distance
                    self.last_min_distance = self.min_distance_in_episode
                    
                    # If stagnation exceeds threshold, trigger exploration mode
                    if self.stagnation_steps >= self.stagnation_threshold and not self.exploration_triggered:
                        self.exploration_triggered = True
                        self.exploration_steps_remaining = 30  # Explore for 30 steps
                        if debug_mode:
                            print(f"Stagnation detected! Triggering exploration mode for 30 steps")
                            print(f"Current min distance: {self.min_distance_in_episode*100:.2f}cm, Last min: {self.last_min_distance*100:.2f}cm")
                
                # Calculate enhanced reward
                joint_positions = next_state[:6]  # First 6 values are joint positions
                reward, reward_components = calculate_reward(
                    ee_position, 
                    self.target_position, 
                    action,
                    self.prev_distance,
                    joint_positions
                )
                
                # Apply step penalty for lack of progress - with cap
                if not made_progress:
                    # Exponential penalty that grows with cap
                    step_penalty = max(self.max_penalty, self.base_penalty * (2.0 ** min(15, self.consecutive_no_progress/2)))
                    reward += step_penalty
                    reward_components['step_penalty'] = step_penalty
                else:
                    reward_components['step_penalty'] = 0.0
                
                # Add a distance-based penalty that increases as we get further from the target
                # This creates a stronger gradient pushing away from bad positions
                # Quadratic penalty with cap
                max_distance_penalty = -10.0  # Cap for distance penalty
                distance_penalty = max(max_distance_penalty, -1.0 * current_distance * current_distance)
                reward += distance_penalty
                reward_components['distance_penalty'] = distance_penalty
                
                # Add a strong penalty if we're moving away from the target
                if self.prev_distance is not None and current_distance > self.prev_distance:
                    # Penalty for moving away - with cap
                    max_moving_away_penalty = -10.0  # Cap for moving away penalty (reduced from -15.0)
                    moving_away_penalty = max(max_moving_away_penalty, -1.5 * (current_distance - self.prev_distance) * 20.0)  # Reduced from -2.0
                    reward += moving_away_penalty
                    reward_components['moving_away_penalty'] = moving_away_penalty
                else:
                    reward_components['moving_away_penalty'] = 0.0
                
                # Add a bonus for getting closer to the target
                if self.prev_distance is not None and current_distance < self.prev_distance:
                    # Bonus for progress - with cap
                    max_progress_bonus = 10.0  # Cap for progress bonus (reduced from 15.0)
                    progress_bonus = min(max_progress_bonus, 1.5 * (self.prev_distance - current_distance) * 20.0)  # Reduced from 2.0
                    reward += progress_bonus
                    reward_components['progress_bonus'] = progress_bonus
                else:
                    reward_components['progress_bonus'] = 0.0
                
                # Update previous distance
                self.prev_distance = current_distance
                
                # Episode is done if we're close enough to the target (6cm)
                done = current_distance < 0.06
                
                # Add reward components to info
                info.update(reward_components)
                info['min_distance_in_episode'] = self.min_distance_in_episode
                info['consecutive_no_progress'] = self.consecutive_no_progress
                info['made_progress'] = made_progress
                info['stagnation_steps'] = self.stagnation_steps
                info['exploration_mode'] = self.exploration_triggered
                
                # Calculate progress percentage
                if self.starting_distance > 0:
                    progress_pct = 100 * (self.starting_distance - self.min_distance_in_episode) / self.starting_distance
                else:
                    progress_pct = 0  # Avoid division by zero
                info['progress_percentage'] = progress_pct
                
                # Increment step counter
                self.episode_step_count += 1
                info['episode_step'] = self.episode_step_count
                
                # Add target reached info
                info['target_reached'] = done
                
                return next_state, reward, done, info
        
        # Create the enhanced environment
        env = EnhancedRewardEnv(
            render=render, 
            randomize=randomize,
            randomize_target_every_step=use_randomize_target
        )
        
        # Set workspace size directly from the global parameter
        env.workspace_size = workspace_size
        
        # Set random seed
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Function to generate a sequence of fixed targets for evaluation
def generate_fixed_target_sequence(num_targets=5, workspace_size=0.7):
    """
    Generate a sequence of fixed targets for evaluation
    
    Args:
        num_targets: Number of targets to generate
        workspace_size: Size of the workspace
        
    Returns:
        List of target positions that are reachable by the robot arm
    """
    targets = []
    
    # The FANUC LR Mate 200iC has a reach of approximately 0.7m
    # We'll use this to constrain our sampling to ensure reachable positions
    max_reach = min(0.7, workspace_size)
    
    # Generate targets in a structured pattern to ensure good coverage
    # We'll use spherical coordinates to ensure reachable positions
    
    # Define a set of radii
    radii = np.linspace(0.25, max_reach, 3)
    
    # Define a set of azimuth angles (around z-axis)
    # We limit this to the front 180 degrees since the robot can't easily reach behind itself
    azimuths = np.linspace(-np.pi/2, np.pi/2, 5)
    
    # Define a set of elevation angles (from xy-plane)
    elevations = np.linspace(-np.pi/4, np.pi/3, 3)
    
    # Generate combinations of these parameters
    for radius in radii:
        for azimuth in azimuths:
            for elevation in elevations:
                if len(targets) >= num_targets:
                    break
                
                # Convert spherical coordinates to cartesian
                x = radius * np.cos(elevation) * np.cos(azimuth) + 0.3  # Offset from base
                y = radius * np.cos(elevation) * np.sin(azimuth)
                z = radius * np.sin(elevation) + 0.3  # Offset from ground
                
                # Ensure z is at least 0.1m above the ground
                z = max(z, 0.1)
                
                # Ensure the position is within the workspace boundaries
                x = np.clip(x, 0.2, 0.2 + workspace_size)
                y = np.clip(y, -workspace_size/2, workspace_size/2)
                z = np.clip(z, 0.1, 0.1 + workspace_size)
                
                targets.append(np.array([x, y, z]))
                
                if len(targets) >= num_targets:
                    break
            if len(targets) >= num_targets:
                break
        if len(targets) >= num_targets:
            break
    
    # If we didn't generate enough targets, add some random ones
    while len(targets) < num_targets:
        # Sample radius (distance from base)
        radius = np.random.uniform(0.2, max_reach)
        
        # Sample angles
        azimuth = np.random.uniform(-np.pi/2, np.pi/2)
        elevation = np.random.uniform(-np.pi/4, np.pi/3)
        
        # Convert spherical coordinates to cartesian
        x = radius * np.cos(elevation) * np.cos(azimuth) + 0.3  # Offset from base
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation) + 0.3  # Offset from ground
        
        # Ensure z is at least 0.1m above the ground
        z = max(z, 0.1)
        
        # Ensure the position is within the workspace boundaries
        x = np.clip(x, 0.2, 0.2 + workspace_size)
        y = np.clip(y, -workspace_size/2, workspace_size/2)
        z = np.clip(z, 0.1, 0.1 + workspace_size)
        
        targets.append(np.array([x, y, z]))
    
    return targets

# Function to run evaluation with visualization
def run_evaluation(model_path):
    print(f"\nRunning visual evaluation on model: {model_path}")
    
    # Always use a single environment for evaluation to ensure visualization works
    print("Using a single environment for evaluation visualization")
    
    # For evaluation, we'll use a fixed target or randomize less frequently
    eval_randomize_every_step = False
    
    if use_eval_fixed_sequence:
        print("Using a fixed sequence of targets for evaluation")
        # Generate a sequence of fixed targets
        fixed_targets = generate_fixed_target_sequence(num_targets=5, workspace_size=workspace_size)
        print(f"Generated {len(fixed_targets)} fixed targets for evaluation")
        eval_randomize_every_step = False
    elif randomize_every_step:
        print("Note: Using less frequent target randomization for better visualization during evaluation")
        eval_randomize_every_step = True
    
    # Create evaluation environment with rendering
    eval_env = DummyVecEnv([make_env(0, randomize=False, render=True, 
                                     randomize_target=eval_randomize_every_step)])
    
    # Load the model
    model = PPO.load(model_path, env=eval_env)
    
    # Run evaluation
    print("\nStarting visual evaluation...")
    try:
        obs = eval_env.reset()
        done = False
        step_count = 0
        max_steps = 1000
        
        # Get initial target position from environment info
        # This ensures we're using the environment's target
        _, _, _, info = eval_env.step(np.zeros(eval_env.action_space.shape))
        if 'target_position' in info[0]:
            target_position = info[0]['target_position']
        else:
            # Fixed target position for evaluation (center of workspace)
            if use_eval_fixed_sequence:
                # Start with the first target in the sequence
                target_idx = 0
                target_position = fixed_targets[target_idx]
                target_change_interval = 200  # Change target every 200 steps for better visualization
            else:
                target_position = np.array([0.5, 0.0, 0.5])
        
        # Visualization settings
        visualization_update_freq = 1  # Update visualizations every step
        max_trajectory_markers = args.n_steps  # Use the same batch size as training
        
        # Visualize the target position
        try:
            # Access the underlying environment to get the PyBullet client ID
            env = eval_env.envs[0].robot_env
            
            # Configure camera for fullscreen view
            configure_camera_fullscreen(env.client)
            
            # Create a visual marker for the target
            target_visual_id = visualize_target(target_position, env.client)
            print("Target position visualized as a red sphere")
            
            # Add a small delay to ensure visualization is set up properly
            time.sleep(0.5)
        except Exception as e:
            if debug_mode:
                print(f"Could not visualize target: {e}")
                traceback.print_exc()
            else:
                print("Warning: Could not set up visualization properly. Continuing anyway.")
        
        # For tracking distances
        best_distance_in_episode = float('inf')
        all_distances = []
        distance_sum = 0.0
        
        # For tracking trajectory
        trajectory_markers = []
        
        print("\nWatching the robot arm move...")
        print("Press Ctrl+C to stop the evaluation at any time")
        print("Target position is shown as a red sphere")
        print("Green to red dots show the end-effector trajectory (green = closer to target)")
        
        # Add a small delay before starting to ensure visualization is ready
        time.sleep(1.0)
        
        while not done and step_count < max_steps:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # If using fixed sequence, check if it's time to change the target
            if use_eval_fixed_sequence and step_count > 0 and step_count % target_change_interval == 0:
                # Move to the next target in the sequence
                target_idx = (target_idx + 1) % len(fixed_targets)
                target_position = fixed_targets[target_idx]
                
                # Update target visualization
                try:
                    # Clear all trajectory markers when target changes
                    for marker_id in trajectory_markers:
                        try:
                            p.removeBody(marker_id, physicsClientId=env.client)
                        except Exception as e:
                            if debug_mode:
                                print(f"Could not remove trajectory marker: {e}")
                    trajectory_markers = []
                    
                    if target_visual_id is not None:
                        try:
                            # First try to remove the old visualization
                            p.removeBody(target_visual_id, physicsClientId=env.client)
                            # Add a small delay to ensure the removal completes
                            time.sleep(0.05)
                        except Exception as e:
                            if debug_mode:
                                print(f"Could not remove old target visualization: {e}")
                    
                    # Create a new visualization
                    target_visual_id = visualize_target(target_position, env.client)
                    print(f"Changed to target {target_idx+1}/{len(fixed_targets)}")
                    
                    # Add a small delay to ensure the new visualization is set up properly
                    time.sleep(0.1)
                except Exception as e:
                    if debug_mode:
                        print(f"Could not update target visualization: {e}")
                        traceback.print_exc()
                    else:
                        print(f"Warning: Could not update target visualization. Continuing anyway.")
            
            # Take step in environment
            obs, reward, done, info = eval_env.step(action)
            
            # Extract end-effector position from observation (indices 12-15 contain XYZ position)
            ee_position = obs[0, 12:15]  # First dimension is for the environment index in VecEnv
            
            # If target position is in info (for randomized targets), update it
            if not use_eval_fixed_sequence and 'target_position' in info[0]:
                new_target_position = info[0]['target_position']
                
                # Only update visualization if target actually changed
                if not np.array_equal(target_position, new_target_position):
                    target_position = new_target_position
                    
                    # Update target visualization
                    try:
                        # Clear all trajectory markers when target changes
                        for marker_id in trajectory_markers:
                            try:
                                p.removeBody(marker_id, physicsClientId=env.client)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Could not remove trajectory marker: {e}")
                        trajectory_markers = []
                        
                        # Try to update the target visualization
                        if target_visual_id is not None:
                            try:
                                # First try to remove the old visualization
                                p.removeBody(target_visual_id, physicsClientId=env.client)
                                # Add a small delay to ensure the removal completes
                                time.sleep(0.05)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Could not remove old target visualization: {e}")
                        
                        # Create a new visualization
                        target_visual_id = visualize_target(target_position, env.client)
                        
                        if debug_mode:
                            print(f"Updated target visualization")
                            
                            # Add a small delay to ensure visualization updates properly
                            time.sleep(0.01)
                    except Exception as e:
                        if debug_mode:
                            print(f"Could not update target visualization: {e}")
                            traceback.print_exc()
            
            # Calculate distance to target
            distance = np.linalg.norm(ee_position - target_position)
            
            # Track distances
            all_distances.append(distance)
            distance_sum += distance
            
            # Track best distance
            if distance < best_distance_in_episode:
                best_distance_in_episode = distance
            
            # Add trajectory marker at regular intervals
            if step_count % visualization_update_freq == 0:
                try:
                    # Normalize distance for color mapping (0-30cm range)
                    norm_dist = min(1.0, distance * 100 / 30)
                    # Green to red color gradient
                    color = [norm_dist, 1.0 - norm_dist, 0.0, 0.7]  # Increased opacity
                    
                    # Make markers smaller to reduce clutter
                    marker_radius = 0.008  # Reduced from 0.01 to 0.008 for better performance
                    marker_id = visualize_ee_position(ee_position, env.client, color=color, radius=marker_radius)
                    trajectory_markers.append(marker_id)
                    
                    # Only remove oldest markers if we exceed the maximum
                    if len(trajectory_markers) > max_trajectory_markers:
                        try:
                            p.removeBody(trajectory_markers.pop(0), physicsClientId=env.client)
                        except Exception as e:
                            if debug_mode:
                                print(f"Could not remove old trajectory marker: {e}")
                except Exception as e:
                    if debug_mode:
                        print(f"Could not add trajectory marker: {e}")
            
            step_count += 1
            
            # Calculate current average
            current_avg = distance_sum / step_count
            
            # Print current distance every 100 steps
            if step_count % 100 == 0 or step_count == 1:
                print(f"Step {step_count}: Current: {distance*100:.2f} cm | Avg: {current_avg*100:.2f} cm | Best: {best_distance_in_episode*100:.2f} cm")
            
            # Add a small delay for better visualization
            time.sleep(gui_delay)
        
        # Calculate final statistics
        avg_distance = sum(all_distances) / len(all_distances) if all_distances else float('inf')
        median_distance = sorted(all_distances)[len(all_distances) // 2] if all_distances else float('inf')
        min_distance = min(all_distances) if all_distances else float('inf')
        
        # Convert to cm for display
        avg_distance_cm = avg_distance * 100.0
        median_distance_cm = median_distance * 100.0
        min_distance_cm = min_distance * 100.0
        
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION RESULTS:")
        print(f"Average distance: {float(avg_distance_cm):.2f} cm")
        print(f"Median distance: {float(median_distance_cm):.2f} cm")
        print(f"Best distance: {float(min_distance_cm):.2f} cm")
        print(f"TARGET ACCURACY: {float(target_accuracy_cm):.2f} cm")
        print(f"EVALUATION STEPS: {step_count}")
        print(f"{'='*60}\n")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        if debug_mode:
            traceback.print_exc()
    
    # Clean up
    eval_env.close()
    print("Evaluation completed.")

# Main training function with simpler approach
def train_robot():
    # Training parameters
    total_steps = args.steps
    
    print(f"\nTraining for {total_steps} steps")
    print(f"Target accuracy: {target_accuracy_cm:.2f} cm")
    print(f"Workspace size: {workspace_size:.2f} m")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Initial entropy coefficient: {args.ent_coef} (will adapt dynamically with balanced rates and caps)")
    print(f"Entropy bounds: [0.05, 1.0] (prevents extreme exploration/exploitation)")
    print(f"Network size: 2048 hidden units (significantly increased parameter count)")
    print(f"Update frequency: Every {args.n_steps} steps (very frequent weight updates)")
    print(f"Simulation steps per action: 24 (fixed)")
    print(f"Parallel environments: {n_parallel_envs} (reserving at least 4 cores for system processes)")
    print(f"Step-based penalty: Enhanced (capped exponential penalty for consecutive steps without progress)")
    print(f"Reward/penalty caps: Enabled (prevents extreme reward values for stability)")
    
    # Display target randomization settings
    if randomize_every_step:
        print("Target position will be completely randomized for EVERY step")
        print("This ensures the model learns to reach any position in the workspace")
    elif use_random_targets:
        print("Using random targets during training (randomized on environment reset)")
    else:
        print("Using fixed target at [0.5, 0.0, 0.5]")
    
    print(f"Training will continue until target accuracy of {target_accuracy_cm:.2f} cm is reached or {total_steps} steps are completed")
    
    # Initialize the environments with optimized settings
    if n_parallel_envs > 1:
        # Create multiple environments for parallel training with optimized settings
        # Always render the first environment
        env_fns = []
        for i in range(n_parallel_envs):
            # Only render the first environment
            should_render = (i == 0)
            if should_render:
                print(f"Environment {i} will be rendered for visualization")
            env_fns.append(make_env(i, render=should_render))
        
        # Use SubprocVecEnv with optimized settings
        env = SubprocVecEnv(env_fns, start_method='spawn' if sys.platform == 'win32' else 'fork')
        
        # Only the first environment will be rendered
        render_env_idx = 0
        print(f"Using {n_parallel_envs} parallel environments with optimized process management")
        print(f"Note: Only environment 0 will be visualized")
    else:
        # Single environment with rendering
        env = DummyVecEnv([make_env(0, render=True)])
        render_env_idx = 0
        print(f"Visualization: Enabled (delay: {gui_delay} sec)")
    
    # Verify that rendering is working by checking the client ID
    try:
        # Access the underlying environment to get the PyBullet client ID
        pybullet_env = env.envs[render_env_idx].robot_env
        client_id = pybullet_env.client
        
        # Test if we can create a visual marker
        test_marker_id = visualize_ee_position([0.5, 0.0, 0.5], client_id, color=[1, 0, 1, 1], radius=0.02)
        if test_marker_id is not None:
            print("Visualization test successful - markers are working")
            # Remove the test marker after a short delay
            time.sleep(0.1)
            try:
                p.removeBody(test_marker_id, physicsClientId=client_id)
            except:
                pass
        
        # Configure camera for fullscreen view
        configure_camera_fullscreen(client_id)
        print("Camera configured for visualization")
    except Exception as e:
        print(f"WARNING: Could not verify visualization: {e}")
        if debug_mode:
            traceback.print_exc()
    
    # Initialize or load the model
    if args.load:
        print(f"Loading pre-trained model from {args.load}")
        model = PPO.load(args.load, env=env)
        # Ensure the loaded model uses the specified entropy coefficient
        model.ent_coef = args.ent_coef
        print(f"Set entropy coefficient of loaded model to {args.ent_coef}")
    else:
        # Initialize a new model with increased n_steps (less frequent updates)
        # and larger batch size for more stable updates
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log="./logs/",
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=min(args.batch_size, args.n_steps // n_parallel_envs * 2),  # Optimize batch size based on parallel envs
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,  # Use the specified entropy coefficient
            device=device,
            policy_kwargs=policy_kwargs
        )
        print(f"Initialized new model with entropy coefficient {args.ent_coef}")
    
    # Create training callback
    training_callback = TrainingCallback(verbose=1)
    
    # Train the model
    try:
        print("\nStarting training...")
        # Use standard learn method
        model.learn(
            total_timesteps=total_steps,
            callback=training_callback,
            tb_log_name="PPO_training",
            reset_num_timesteps=False
        )
        
        # Save the final model
        model.save("./models/fanuc_final_model")
        print("Final model saved to ./models/fanuc_final_model")
        
        # Check if target accuracy was reached
        if training_callback.target_reached:
            print(f"\nTarget accuracy of {target_accuracy_cm:.2f} cm reached!")
        else:
            print(f"\nTraining completed without reaching target accuracy.")
            print(f"Best average distance: {training_callback.best_avg_distance * 100.0:.2f} cm")
            print(f"Target accuracy: {target_accuracy_cm:.2f} cm")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save("./models/fanuc_model_interrupted")
        print("Model saved to ./models/fanuc_model_interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        if debug_mode:
            traceback.print_exc()
        print("Saving model before exiting...")
        model.save("./models/fanuc_model_error")
        print("Model saved to ./models/fanuc_model_error")
        sys.exit(1)
    
    # Clean up
    env.close()
    
    # Run a final visual evaluation with the trained model
    print("\nRunning final visual evaluation...")
    run_evaluation("./models/fanuc_final_model")

if __name__ == "__main__":
    try:
        # Print banner
        print("\n" + "="*78)
        print(" FANUC Robot Arm Training - Visual Version ".center(78, "="))
        print("="*78 + "\n")
        
        # Check if we're only running evaluation
        if args.eval_only:
            if not args.load:
                print("Error: Must specify a model to load with --load when using --eval-only")
                sys.exit(1)
            run_evaluation(args.load)
        else:
            # Start training
            train_robot()
        
        # Print final message with instructions
        print("\nTraining completed!")
        print("\nUsage examples:")
        print("  Basic visual training (with randomized targets and 0.05s delay):")
        print("    python train_robot_visual.py")
        print("  Faster training with fewer steps:")
        print("    python train_robot_visual.py --steps 250000")
        print("  Different target accuracy:")
        print("    python train_robot_visual.py --target 5.0")
        print("  Adjust workspace size:")
        print("    python train_robot_visual.py --workspace-size 0.5")
        print("  Faster visualization (less delay):")
        print("    python train_robot_visual.py --gui-delay 0.01")
        print("  Disable target randomization on every step (use fixed targets):")
        print("    python train_robot_visual.py --no-randomize-every-step")
        print("  Evaluate with a fixed sequence of targets (better visualization):")
        print("    python train_robot_visual.py --eval-only --load ./models/fanuc_final_model --eval-fixed-sequence")
        print("  Fullscreen mode with zoomed-in view of the robot:")
        print("    python train_robot_visual.py --fullscreen")
        print("  Load a pre-trained model:")
        print("    python train_robot_visual.py --load ./models/fanuc_model_interrupted")
        print("  Evaluate a pre-trained model (visualization only):")
        print("    python train_robot_visual.py --eval-only --load ./models/fanuc_final_model")
        print("  Specify number of parallel environments:")
        print("    python train_robot_visual.py --parallel 8")
        print("  Combine options:")
        print("    python train_robot_visual.py --fullscreen --workspace-size 0.8 --steps 300000")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        if debug_mode:
            traceback.print_exc()
        sys.exit(1) 