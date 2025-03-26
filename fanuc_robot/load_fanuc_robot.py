#!/usr/bin/env python3
# Load FANUC robot in PyBullet

import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data

# Configuration
SHOW_GUI = True           # Set to False for headless mode
GRAVITY = -9.81           # Gravity constant
TIME_STEP = 1/240.0       # Physics update rate
DEMO_DURATION = 15.0      # Duration of the demo in seconds

def setup_environment():
    """Set up the simulation environment"""
    # Initialize PyBullet
    if SHOW_GUI:
        client_id = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    else:
        client_id = p.connect(p.DIRECT)
    
    # Configure camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.3,
        cameraYaw=50,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    # Add data path and load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Set gravity and time step
    p.setGravity(0, 0, GRAVITY)
    p.setTimeStep(TIME_STEP)
    p.setRealTimeSimulation(0)  # 0 = non-realtime simulation
    
    return client_id

def load_fanuc_robot():
    """Load the FANUC robot URDF"""
    # Get the path to the URDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "fanuc.urdf")
    
    # Load the robot URDF
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION
    )
    
    # Print robot information
    print(f"Loaded FANUC robot with ID: {robot_id}")
    print(f"Number of joints: {p.getNumJoints(robot_id)}")
    
    return robot_id

def print_joint_info(robot_id):
    """Print information about robot joints"""
    num_joints = p.getNumJoints(robot_id)
    print("\nJoint Information:")
    print("-------------------")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
        print(f"  Type: {joint_info[2]}")
        print(f"  Lower limit: {joint_info[8]}")
        print(f"  Upper limit: {joint_info[9]}")
        print(f"  Max force: {joint_info[10]}")
        print(f"  Max velocity: {joint_info[11]}")
        print("-------------------")

def get_joint_by_name(robot_id, name):
    """Get joint index by name"""
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') == name:
            return i
    return None

def get_arm_joint_indices(robot_id):
    """Get arm joint indices (the 6 main revolute joints)"""
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    indices = []
    
    for name in joint_names:
        idx = get_joint_by_name(robot_id, name)
        if idx is not None:
            indices.append(idx)
    
    return indices

def reset_robot_positions(robot_id):
    """Reset robot to home position"""
    # Get joint indices
    arm_joints = get_arm_joint_indices(robot_id)
    
    # Reset arm joints to zero position
    for idx in arm_joints:
        p.resetJointState(robot_id, idx, 0)
    
    # Step simulation to apply changes
    for _ in range(10):
        p.stepSimulation()

def move_to_joint_position(robot_id, joint_positions, duration=1.0):
    """Move to specified joint positions over a duration"""
    arm_joints = get_arm_joint_indices(robot_id)
    
    # Check if we have the correct number of positions
    if len(joint_positions) != len(arm_joints):
        print(f"Error: Expected {len(arm_joints)} joint positions, got {len(joint_positions)}")
        return
    
    # Store initial positions for interpolation
    initial_positions = []
    for idx in arm_joints:
        joint_state = p.getJointState(robot_id, idx)
        initial_positions.append(joint_state[0])
    
    # Calculate number of steps based on duration and time step
    steps = int(duration / TIME_STEP)
    
    # Interpolate and move
    for step in range(steps):
        t = step / steps  # Interpolation factor [0, 1]
        
        # Linear interpolation for each joint
        for i, joint_idx in enumerate(arm_joints):
            current_pos = initial_positions[i] + t * (joint_positions[i] - initial_positions[i])
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=current_pos,
                force=300
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Sleep if in GUI mode to make motion visible
        if SHOW_GUI:
            time.sleep(TIME_STEP)

def demo_robot_motion(robot_id):
    """Demonstrate robot motion with a series of poses"""
    print("\nStarting robot motion demo...")
    
    # Get joint indices
    arm_joints = get_arm_joint_indices(robot_id)
    
    # Reset to home position
    reset_robot_positions(robot_id)
    time.sleep(1.0) if SHOW_GUI else None
    
    # Define a series of joint positions for the demo
    positions = [
        # Joint positions format: [joint1, joint2, joint3, joint4, joint5, joint6]
        [0, 0, 0, 0, 0, 0],                   # Home position
        [math.pi/4, 0, 0, 0, 0, 0],           # Rotate base 45 degrees
        [math.pi/4, math.pi/6, 0, 0, 0, 0],   # Lift arm slightly
        [math.pi/4, math.pi/6, -math.pi/4, 0, 0, 0],  # Bend elbow
        [math.pi/4, math.pi/6, -math.pi/4, math.pi/2, 0, 0],  # Rotate wrist
        [math.pi/4, math.pi/6, -math.pi/4, math.pi/2, math.pi/4, 0],  # Tilt wrist
        [math.pi/4, math.pi/6, -math.pi/4, math.pi/2, math.pi/4, math.pi],  # Rotate tool
        [math.pi/4, math.pi/6, -math.pi/4, math.pi/2, math.pi/4, 0],  # Return tool
        [0, 0, 0, 0, 0, 0],                   # Return to home
    ]
    
    # Execute each position in sequence
    for i, pos in enumerate(positions):
        print(f"Moving to position {i+1}/{len(positions)}")
        move_to_joint_position(robot_id, pos, duration=1.5)
        time.sleep(0.5) if SHOW_GUI else None
    
    print("Demo completed!")

def optimize_robot_parameters(robot_id):
    """
    Optimize robot parameters for stable simulation with the original inertia values.
    This helps prevent instability with the precise (but potentially challenging for physics)
    inertia tensors from the original specifications.
    """
    print("Optimizing robot parameters for stable simulation...")
    
    # Get all joints
    num_joints = p.getNumJoints(robot_id)
    
    # Increase joint damping for stability
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        
        # Only adjust movable joints (revolute, prismatic)
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            # Add additional damping beyond what's in the URDF
            p.changeDynamics(
                robot_id, 
                i, 
                linearDamping=0.1,
                angularDamping=0.9,
                jointDamping=0.5
            )
            
            # For inertia stability when original values are used
            p.changeDynamics(
                robot_id,
                i,
                maxJointVelocity=10,  # Limit max velocity
                contactStiffness=10000,
                contactDamping=1,
                restitution=0.1,
                lateralFriction=0.5
            )
    
    # Increase base stability
    p.changeDynamics(
        robot_id,
        -1,  # Base link
        linearDamping=0.04,
        angularDamping=0.04,
        contactStiffness=10000,
        contactDamping=1
    )
    
    print("Robot parameters optimized for stability with original inertia values")

def main():
    """Main function"""
    # Setup environment
    client_id = setup_environment()
    
    # Load robot
    robot_id = load_fanuc_robot()
    
    # Optimize robot parameters for stability
    optimize_robot_parameters(robot_id)
    
    # Print joint info
    print_joint_info(robot_id)
    
    # Run demo
    demo_robot_motion(robot_id)
    
    # Keep simulation running if in GUI mode
    if SHOW_GUI:
        print("\nSimulation running. Press Ctrl+C to exit.")
        while True:
            p.stepSimulation()
            time.sleep(TIME_STEP)
    
    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main() 