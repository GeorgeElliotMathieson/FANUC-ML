#!/usr/bin/env python3
# Test script to verify the JointLimitedBox action space implementation

import os
import sys
import time
import numpy as np

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the robot environment and JointLimitedBox
from res.rml.python.robot_sim import FANUCRobotEnv
from res.rml.python.train_robot_rl_positioning_revamped import JointLimitedBox

def test_joint_limited_action_space():
    """
    Test the JointLimitedBox action space to verify it correctly maps
    normalized actions to actual joint positions within limits.
    """
    print("\n" + "="*80)
    print("Testing JointLimitedBox Action Space")
    print("="*80 + "\n")
    
    # Create robot environment with verbose output
    env = FANUCRobotEnv(render=True, verbose=True)
    
    # Create JointLimitedBox action space
    action_space = JointLimitedBox(env, shape=(env.dof,), dtype=np.float32)
    
    print("\nJoint limits from URDF:")
    for joint_idx, limits in env.joint_limits.items():
        print(f"Joint {joint_idx}: Limits = [{limits[0]:.6f}, {limits[1]:.6f}] radians")
    
    print("\nTesting action space mapping:")
    
    # Test extreme actions [-1, 1] for each joint
    test_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for test_value in test_values:
        print(f"\n--- Testing normalized action value: {test_value:.1f} ---")
        
        # Create test action with the same value for all joints
        test_action = np.ones(env.dof) * test_value
        
        # Map to actual joint positions
        joint_positions = action_space.unnormalize_action(test_action)
        
        # Print mapped positions
        print("Mapped joint positions:")
        for joint_idx, pos in enumerate(joint_positions):
            if joint_idx in env.joint_limits:
                limit_low, limit_high = env.joint_limits[joint_idx]
                mid = (limit_high + limit_low) / 2.0
                rng = (limit_high - limit_low) / 2.0
                expected = mid + test_value * rng
                print(f"  Joint {joint_idx}: {pos:.6f} radians (expected: {expected:.6f}, limits: [{limit_low:.6f}, {limit_high:.6f}])")
            else:
                print(f"  Joint {joint_idx}: {pos:.6f} radians (no limits defined)")
        
        # Apply action to robot
        env.step(joint_positions)
        time.sleep(0.5)
    
    print("\nTest completed!")
    
    # Reset robot to home position
    env.reset()
    
    try:
        # Keep running until user interrupts
        print("\nPress Ctrl+C to exit...")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting test.")
    finally:
        env.close()

if __name__ == "__main__":
    test_joint_limited_action_space() 