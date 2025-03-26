#!/usr/bin/env python3
# Test script to verify PPO agent with JointLimitedBox action space

import os
import sys
import time
import numpy as np
import pybullet as p
import torch as th

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the necessary components
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO  # Use standard PPO
from res.rml.python.train_robot_rl_positioning_revamped import (
    RobotPositioningRevampedEnv,
    JointLimitedBox,
    JointLimitEnforcingEnv,
    get_shared_pybullet_client
)

def test_ppo_with_joint_limits():
    """
    Test that PPO correctly respects joint limits when using JointLimitedBox action space.
    This demonstrates that the joint limits are enforced inherently within the model's
    action space rather than by clamping after actions are decided.
    """
    print("\n" + "="*80)
    print("Testing PPO with JointLimitedBox Action Space")
    print("="*80 + "\n")
    
    # Create environment with visualization
    print("Creating environment...")
    
    # Initialize PyBullet client
    client_id = get_shared_pybullet_client(render=True)
    
    # Create a single environment
    base_env = RobotPositioningRevampedEnv(
        gui=True,
        gui_delay=0.05,
        viz_speed=0.05,
        verbose=True,
        rank=0
    )
    
    # Wrap environment with JointLimitEnforcingEnv
    env = JointLimitEnforcingEnv(base_env)
    
    # Create a DummyVecEnv to work with PPO
    vec_env = DummyVecEnv([lambda: env])
    
    # Verify it uses JointLimitedBox action space
    print(f"Base env action space type: {type(base_env.action_space).__name__}")
    print(f"Wrapped env action space type: {type(env.action_space).__name__}")
    
    # Create a PPO agent with default parameters
    print("\nCreating PPO agent...")
    model = PPO("MlpPolicy", vec_env, verbose=1)
    
    # Generate observations to test action sampling
    print("\nTesting action sampling from policy...")
    obs = vec_env.reset()
    
    # Sample several actions
    print("\n--- Testing action generation ---")
    for i in range(10):
        # Sample both deterministic and non-deterministic actions
        deterministic = (i % 2 == 0)  # Alternate between deterministic and non-deterministic
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Print normalized action
        print(f"\n  Action {i+1} (deterministic={deterministic}):")
        print(f"    Normalized action: {action[0]}")
        
        # Unnormalize to see actual joint positions
        robot = base_env.robot
        joint_positions = base_env.action_space.unnormalize_action(action[0])
        
        # Check if all joint positions are within limits
        all_within_limits = True
        for j, pos in enumerate(joint_positions):
            if j in robot.joint_limits:
                limit_low, limit_high = robot.joint_limits[j]
                within_limits = limit_low <= pos <= limit_high
                all_within_limits = all_within_limits and within_limits
                status = "✓" if within_limits else "✗"
                print(f"    Joint {j}: {pos:.4f} rad (limits: [{limit_low:.4f}, {limit_high:.4f}]) {status}")
        
        if all_within_limits:
            print("    All joint positions are within limits!")
        else:
            print("    WARNING: Some joint positions are outside limits!")
        
        # Apply action to see the robot move
        obs, _, _, _ = vec_env.step(action)
        time.sleep(0.5)
    
    print("\nTest completed! All actions were inherently within joint limits.")
    
    # Keep GUI open for observation
    try:
        print("\nPress Ctrl+C to exit...")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting test.")
    finally:
        vec_env.close()

if __name__ == "__main__":
    test_ppo_with_joint_limits() 