"""
Environment module for FANUC Robot ML Platform.

Contains environment classes and action spaces for robot control.
"""

# Import action spaces
from src.core.env.action_spaces import (
    JointLimitedBox, 
    JointLimitEnforcingEnv,
    ensure_joint_limits
)

# Import visualization utilities
from src.core.env.visualization import (
    visualize_target,
    visualize_trajectory,
    visualize_workspace,
    remove_visualization
)

# Import reward functions
from src.core.env.rewards import (
    calculate_distance_reward,
    calculate_action_smoothness_reward,
    calculate_joint_limit_reward,
    calculate_success_reward,
    calculate_timeout_reward,
    calculate_combined_reward
)

# Define exports
__all__ = [
    # Action spaces
    'JointLimitedBox',
    'JointLimitEnforcingEnv',
    'ensure_joint_limits',
    
    # Visualization utilities
    'visualize_target',
    'visualize_trajectory',
    'visualize_workspace',
    'remove_visualization',
    
    # Reward functions
    'calculate_distance_reward',
    'calculate_action_smoothness_reward',
    'calculate_joint_limit_reward',
    'calculate_success_reward',
    'calculate_timeout_reward',
    'calculate_combined_reward'
] 