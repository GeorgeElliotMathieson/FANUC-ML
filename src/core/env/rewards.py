"""
Reward functions for the robot environment.
"""

import numpy as np
import math

def calculate_distance_reward(distance, previous_distance, initial_distance, max_distance=1.0):
    """
    Calculate reward based on distance to target.
    
    Args:
        distance: Current distance to target
        previous_distance: Previous distance to target
        initial_distance: Initial distance to target at start of episode
        max_distance: Maximum expected distance for normalization
        
    Returns:
        Distance reward component
    """
    # Base reward is negative, proportional to distance (closer = less negative)
    # Scale to roughly -1.0 when at max_distance
    base_reward = -5.0 * (distance / max_distance)
    
    # Reward for getting closer to target
    delta_reward = 0.0
    if previous_distance is not None:
        delta = previous_distance - distance  # Positive when getting closer
        
        # Scale to provide stronger reward for improvement
        if delta > 0:  # Getting closer
            delta_reward = 10.0 * delta / max_distance
        else:  # Getting further away
            delta_reward = 15.0 * delta / max_distance  # Larger penalty for moving away
    
    # Progress reward based on overall improvement from initial state
    progress_reward = 0.0
    if initial_distance is not None:
        progress = (initial_distance - distance) / initial_distance
        # Only reward progress beyond a threshold to avoid tiny movements
        if progress > 0.001:
            progress_reward = 2.0 * progress
    
    # Combine rewards
    total_reward = base_reward + delta_reward + progress_reward
    
    return total_reward

def calculate_action_smoothness_reward(action, previous_action):
    """
    Calculate reward to encourage smooth actions.
    
    Args:
        action: Current action
        previous_action: Previous action
        
    Returns:
        Smoothness reward component (negative for large changes)
    """
    if previous_action is None:
        return 0.0
        
    # Calculate L2 norm of action difference
    diff = np.linalg.norm(action - previous_action)
    
    # Penalize large changes in action
    smoothness_reward = -0.1 * diff
    
    return smoothness_reward

def calculate_joint_limit_reward(joint_positions, joint_limits):
    """
    Calculate penalty for being close to joint limits.
    
    Args:
        joint_positions: Current joint positions
        joint_limits: Dictionary mapping joint indices to [min, max] limits
        
    Returns:
        Joint limit reward component (negative when close to limits)
    """
    total_penalty = 0.0
    
    for i, pos in enumerate(joint_positions):
        if i in joint_limits:
            min_limit, max_limit = joint_limits[i]
            
            # Calculate distance to closest limit as a fraction of total range
            range_size = max_limit - min_limit
            dist_to_min = (pos - min_limit) / range_size
            dist_to_max = (max_limit - pos) / range_size
            
            # Distance to closest limit
            min_dist = min(dist_to_min, dist_to_max)
            
            # Apply penalty that increases as joints get closer to limits
            # Only penalize when within 10% of limits
            if min_dist < 0.1:
                # Exponential penalty that grows rapidly as joint approaches limit
                penalty_factor = math.exp(5.0 * (0.1 - min_dist)) - 1.0
                total_penalty -= 0.2 * penalty_factor
    
    return total_penalty

def calculate_success_reward(distance, accuracy_threshold):
    """
    Calculate reward for successfully reaching the target.
    
    Args:
        distance: Current distance to target
        accuracy_threshold: Distance threshold for success
        
    Returns:
        Success reward (large positive value if successful)
    """
    if distance <= accuracy_threshold:
        return 100.0  # Large reward for success
    return 0.0

def calculate_timeout_reward(steps, timeout_steps, best_distance):
    """
    Calculate penalty for timeout.
    
    Args:
        steps: Current step count
        timeout_steps: Maximum steps before timeout
        best_distance: Best distance achieved during episode
        
    Returns:
        Timeout penalty (negative) or 0 if not timed out
    """
    if steps >= timeout_steps:
        # Penalty scaled by best distance (less penalty if got close)
        return -50.0 * (best_distance ** 2)
    return 0.0

def calculate_combined_reward(
    distance, 
    previous_distance,
    initial_distance,
    action,
    previous_action,
    joint_positions,
    joint_limits,
    steps,
    timeout_steps,
    best_distance,
    accuracy_threshold,
    positions_within_limits=True
):
    """
    Calculate combined reward from all components.
    
    Args:
        distance: Current distance to target
        previous_distance: Previous distance to target
        initial_distance: Initial distance to target
        action: Current action
        previous_action: Previous action
        joint_positions: Current joint positions
        joint_limits: Dictionary of joint limits
        steps: Current step count
        timeout_steps: Maximum steps before timeout
        best_distance: Best distance achieved during episode
        accuracy_threshold: Distance threshold for success
        positions_within_limits: Whether positions are within joint limits
        
    Returns:
        Combined reward and dictionary of reward components
    """
    # Calculate individual reward components
    distance_reward = calculate_distance_reward(
        distance, previous_distance, initial_distance
    )
    
    action_reward = calculate_action_smoothness_reward(
        action, previous_action
    )
    
    joint_limit_reward = calculate_joint_limit_reward(
        joint_positions, joint_limits
    )
    
    success_reward = calculate_success_reward(
        distance, accuracy_threshold
    )
    
    timeout_reward = calculate_timeout_reward(
        steps, timeout_steps, best_distance
    )
    
    # Apply large penalty if joint limits were exceeded (should never happen with JointLimitedBox)
    limit_violation_penalty = 0.0
    if not positions_within_limits:
        limit_violation_penalty = -50.0
    
    # Combine all rewards
    total_reward = (
        distance_reward +
        action_reward +
        joint_limit_reward +
        success_reward +
        timeout_reward +
        limit_violation_penalty
    )
    
    # Create dictionary of reward components for analysis
    reward_info = {
        'distance_reward': distance_reward,
        'action_reward': action_reward,
        'joint_limit_reward': joint_limit_reward,
        'success_reward': success_reward,
        'timeout_reward': timeout_reward,
        'limit_violation_penalty': limit_violation_penalty,
        'total_reward': total_reward
    }
    
    return total_reward, reward_info 