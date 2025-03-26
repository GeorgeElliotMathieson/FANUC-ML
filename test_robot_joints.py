#!/usr/bin/env python3
# Test script to verify joint limits enforcement from the URDF file

import os
import sys
import time
import numpy as np

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the robot environment
from res.rml.python.robot_sim import FANUCRobotEnv

def main():
    print("\n" + "="*80)
    print("Testing FANUC Robot Joint Limits Enforcement")
    print("="*80 + "\n")
    
    # Create environment with verbose flag enabled to see detailed joint limit information
    env = FANUCRobotEnv(render=True, verbose=True)
    
    print("\nRobot loaded. Joint limits read from URDF file:")
    for joint_idx, limits in env.joint_limits.items():
        print(f"Joint {joint_idx}: Limits = [{limits[0]:.6f}, {limits[1]:.6f}] radians")
    
    print("\nTesting joint limits enforcement:")
    
    # Reset to home position
    print("\nResetting to home position...")
    env.reset()
    time.sleep(1.0)
    
    # Test each joint by moving beyond limits
    for joint_idx in range(env.dof):
        if joint_idx in env.joint_limits:
            limit_low, limit_high = env.joint_limits[joint_idx]
            
            # Test exceeding lower limit
            print(f"\nTesting Joint {joint_idx} - Exceeding Lower Limit ({limit_low:.4f}):")
            action = [0.0] * env.dof
            action[joint_idx] = limit_low - 0.5  # 0.5 radians below the lower limit
            print(f"Attempting to set joint {joint_idx} to {action[joint_idx]:.4f} (below limit)")
            env.step(action)
            time.sleep(1.0)
            
            # Test valid position
            print(f"\nTesting Joint {joint_idx} - Valid Position:")
            action = [0.0] * env.dof
            # Set to 25% of the way between low and high
            action[joint_idx] = limit_low + (limit_high - limit_low) * 0.25
            print(f"Setting joint {joint_idx} to {action[joint_idx]:.4f} (valid position)")
            env.step(action)
            time.sleep(1.0)
            
            # Test exceeding upper limit
            print(f"\nTesting Joint {joint_idx} - Exceeding Upper Limit ({limit_high:.4f}):")
            action = [0.0] * env.dof
            action[joint_idx] = limit_high + 0.5  # 0.5 radians above the upper limit
            print(f"Attempting to set joint {joint_idx} to {action[joint_idx]:.4f} (above limit)")
            env.step(action)
            time.sleep(1.0)
            
            # Reset joint back to zero
            action = [0.0] * env.dof
            env.step(action)
            time.sleep(1.0)
    
    print("\nJoint limits test completed!")
    
    try:
        # Keep running until user interrupts
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting joint limits test.")
    finally:
        env.close()

if __name__ == "__main__":
    main() 