"""
Robot Trainer Module (DirectML Edition)

This module provides the core training functionality for FANUC robot positioning tasks.
It's a more concisely named alias to the original implementation.
"""

# Import all components from the original implementation
from .train_robot_rl_positioning_revamped import (
    RobotPositioningRevampedEnv,
    CustomFeatureExtractor,
    JointLimitedBox,
    JointLimitEnforcingEnv,
    create_revamped_envs,
    train_revamped_robot,
    evaluate_model,
    run_evaluation_sequence,
    CustomPPO
)

# Define aliases for better naming
train_robot = train_revamped_robot
RobotPositioningEnv = RobotPositioningRevampedEnv
create_envs = create_revamped_envs

# Export all components
__all__ = [
    'RobotPositioningEnv',
    'RobotPositioningRevampedEnv',
    'CustomFeatureExtractor',
    'JointLimitedBox',
    'JointLimitEnforcingEnv',
    'create_envs',
    'create_revamped_envs',
    'train_robot',
    'train_revamped_robot',
    'evaluate_model',
    'run_evaluation_sequence',
    'CustomPPO'
] 