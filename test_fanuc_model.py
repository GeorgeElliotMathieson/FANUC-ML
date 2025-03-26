#!/usr/bin/env python3
# Test script to verify the 6-axis FANUC robot model is loading correctly

import os
import sys
import time
import numpy as np
import pybullet as p

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the robot environment
from res.rml.python.robot_sim import FANUCRobotEnv

def main():
    print("\n" + "="*80)
    print("Testing 6-Axis FANUC Robot (without gripper) Visualization")
    print("="*80 + "\n")
    
    # Create environment with GUI rendering
    env = FANUCRobotEnv(render=True, verbose=True)
    
    # Wait for user to observe the robot
    print("\nRobot loaded. Testing joint movements...")
    
    try:
        # Move joints to different positions to demonstrate the robot
        for i in range(12):
            print(f"Movement sequence {i+1}/12...")
            
            # Create some simple joint movements
            action = [0.0] * env.dof
            
            # Move different joints in sequence
            joint_idx = i % env.dof
            action[joint_idx] = np.sin(i/5.0) * np.pi/4  # ~45 degree movement
            
            # Apply the movement
            env.step(action)
            
            # Wait to observe
            time.sleep(1.0)
        
        # Reset to home position
        env.reset()
        print("Demonstration complete. Press Ctrl+C to exit.")
        
        # Keep running until user interrupts
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting robot visualization test.")
    finally:
        env.close()

if __name__ == "__main__":
    main() 