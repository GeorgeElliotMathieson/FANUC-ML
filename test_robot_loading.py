import os
import sys
import time

# Add the current directory to the path
sys.path.append(os.getcwd())

# Set up paths for imports from the correct location
from res.rml.python.pybullet_utils import get_pybullet_client, configure_visualization

# Now import the robot environment
from res.rml.python.train_robot_rl_positioning import FANUCRobotEnv

print("Testing robot loading...")

# Create an environment instance
env = FANUCRobotEnv(render=False, verbose=True)

print("Robot loaded successfully!")
print(f"Robot ID: {env.robot_id}")
print(f"Number of joints: {env.num_joints}")

# Try to reset the robot
print("Resetting robot...")
env.reset()
print("Reset completed!")

# Close the environment
env.close()
print("Test completed successfully!") 