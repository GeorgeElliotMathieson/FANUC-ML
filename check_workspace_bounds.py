#!/usr/bin/env python3
# A script to check the robot's reachable workspace bounds

from src.utils.pybullet_utils import get_pybullet_client, determine_reachable_workspace
from src.envs.robot_sim import FANUCRobotEnv
import numpy as np
import time

# Initialize PyBullet client
client = get_pybullet_client(gui=False)

# Create FANUC robot environment
env = FANUCRobotEnv(client=client, render=False, verbose=False)

# Determine the reachable workspace bounds
max_reach, workspace_bounds = determine_reachable_workspace(env.robot_id, n_samples=1000, client_id=client)

# Print the results
print("\nWorkspace determination complete:")
print(f"Maximum reach: {max_reach:.4f} meters")
print("Workspace bounds:")
print(f"  X: {workspace_bounds['x_min']:.4f} to {workspace_bounds['x_max']:.4f} meters")
print(f"  Y: {workspace_bounds['y_min']:.4f} to {workspace_bounds['y_max']:.4f} meters")
print(f"  Z: {workspace_bounds['z_min']:.4f} to {workspace_bounds['z_max']:.4f} meters")

# Sample a few random end-effector positions
print("\nSampling 5 random end-effector positions:")
for i in range(5):
    # Generate random joint configuration
    joint_positions = []
    for joint_idx in range(env.dof):
        if joint_idx in env.joint_limits:
            limit_low, limit_high = env.joint_limits[joint_idx]
            joint_positions.append(np.random.uniform(limit_low, limit_high))
        else:
            joint_positions.append(0.0)
    
    # Set the robot to this configuration
    env.step(joint_positions)
    
    # Get end-effector position
    state = env._get_state()
    ee_position = state[-7:-4]
    
    # Calculate distance from origin
    distance = np.linalg.norm(ee_position)
    
    # Print the result
    print(f"Sample {i+1}: {ee_position}, Distance: {distance:.4f} meters")

# Clean up
env.close()
print("\nDone!") 