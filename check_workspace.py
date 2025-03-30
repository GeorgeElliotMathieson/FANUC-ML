import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import json # Import json

# --- Constants (Adapted from fanuc_env.py) ---
# Ensure these match your fanuc_env.py
JOINT_LIMITS_LOWER = np.array([-2.96, -1.74, -2.37, -3.31, -2.18]) # J1-J5 radians
JOINT_LIMITS_UPPER = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18]) # J1-J5 radians
NUM_CONTROLLABLE_JOINTS = 5
NUM_SAMPLES = 20000 # Number of random configurations to sample
END_EFFECTOR_LINK_NAME = 'Part6' # Ensure this matches the link name in your URDF/env

CONFIG_FILENAME = "workspace_config.json" # Define config filename

def main():
    """Samples joint configurations to estimate workspace reach."""
    physics_client = -1
    robot_id = -1
    max_reach = 0.0
    max_reach_config = None
    max_reach_pos = None
    # Add variables for minimum reach tracking
    min_reach = float('inf')
    min_reach_config = None
    min_reach_pos = None

    try:
        # --- PyBullet Setup (Direct Mode) ---
        print("Connecting to PyBullet in DIRECT mode...")
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # No gravity needed for pure FK check
        # p.setGravity(0, 0, -9.81)

        # --- Load Robot ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file_path = os.path.join(script_dir, "Fanuc", "urdf", "Fanuc.urdf")
        mesh_path = os.path.join(script_dir, "Fanuc")

        if not os.path.exists(urdf_file_path):
            print(f"Error: Cannot find URDF file at {urdf_file_path}")
            return

        p.setAdditionalSearchPath(mesh_path)
        print(f"Loading robot from: {urdf_file_path}")
        robot_id = p.loadURDF(
            urdf_file_path,
            basePosition=[0, 0, 0],
            useFixedBase=True
            # No flags needed like self-collision for FK check
        )
        print(f"Robot loaded with ID: {robot_id}")

        # --- Get Robot Info ---
        num_joints_total = p.getNumJoints(robot_id)
        joint_indices = []
        end_effector_link_index = -1

        for i in range(num_joints_total):
            info = p.getJointInfo(robot_id, i)
            joint_type = info[2]
            link_name = info[12].decode('UTF-8')

            if joint_type == p.JOINT_REVOLUTE and len(joint_indices) < NUM_CONTROLLABLE_JOINTS:
                joint_indices.append(i)

            if link_name == END_EFFECTOR_LINK_NAME:
                end_effector_link_index = i

        if len(joint_indices) != NUM_CONTROLLABLE_JOINTS:
            print(f"Error: Found {len(joint_indices)} controllable revolute joints, expected {NUM_CONTROLLABLE_JOINTS}.")
            return
        if end_effector_link_index == -1:
            print(f"Error: Could not find end effector link named '{END_EFFECTOR_LINK_NAME}'. Using last joint index as fallback.")
            # Fallback: Use the link associated with the last *controllable* joint index found
            if joint_indices:
                 end_effector_link_index = joint_indices[-1]
            else: # Or ultimately the last joint if no revolute found (unlikely)
                 end_effector_link_index = num_joints_total - 1
            print(f"Using end effector link index: {end_effector_link_index}")


        print(f"Using End Effector Link Index: {end_effector_link_index}")
        print(f"Sampling {NUM_SAMPLES} random joint configurations...")

        start_time = time.time()
        # --- Sampling Loop ---
        for i in range(NUM_SAMPLES):
            # Generate random joint angles within limits
            random_joint_angles = np.random.uniform(low=JOINT_LIMITS_LOWER, high=JOINT_LIMITS_UPPER)

            # Set the robot to this configuration
            # Note: Using resetJointState is slightly inefficient but simple for FK check.
            for joint_idx, angle in zip(joint_indices, random_joint_angles):
                 # p.resetJointState(robot_id, joint_idx, targetValue=angle) # This resets physics state, maybe too slow/heavy
                 # Using direct joint position setting via calculateInverseKinematics's null space feature is complex.
                 # Let's try setting joint positions directly if possible, or stick to reset for simplicity.
                 # Unfortunately, there's no direct 'setJointPositions' in PyBullet without physics.
                 # We *must* involve the physics state, so resetJointState or applying a fast position control step is needed.
                 # Using resetJointState is the most direct way to set a pose for FK.
                 p.resetJointState(robot_id, joint_idx, targetValue=angle, targetVelocity=0)


            # Perform a simulation step to ensure the pose is updated internally? Sometimes needed.
            # p.stepSimulation() # Let's see if it works without this first. Might not be needed for resetJointState.

            # Get End Effector Position using Forward Kinematics
            # We need computeForwardKinematics=True IF we didn't reset the state but used controls.
            # Since we used resetJointState, the base calculation should be correct.
            link_state = p.getLinkState(robot_id, end_effector_link_index) # computeForwardKinematics=0 by default
            ee_position = np.array(link_state[0]) # World position of the link's CoM

            # Calculate distance from base (origin)
            distance_from_base = np.linalg.norm(ee_position)

            # Update max reach
            if distance_from_base > max_reach:
                max_reach = distance_from_base
                max_reach_config = random_joint_angles
                max_reach_pos = ee_position
            
            # Update min reach
            if distance_from_base < min_reach:
                min_reach = distance_from_base
                min_reach_config = random_joint_angles
                min_reach_pos = ee_position

            # Progress indicator
            if (i + 1) % (NUM_SAMPLES // 10) == 0:
                print(f"  Processed {i+1}/{NUM_SAMPLES} samples...")

        end_time = time.time()
        print(f"Sampling finished in {end_time - start_time:.2f} seconds.")

        # --- Output and Save Results ---
        workspace_data = {}
        # Max Reach
        if max_reach_pos is not None:
            print(f"\nEstimated Maximum Reach: {max_reach:.4f} meters")
            print(f"  Found at position: [{max_reach_pos[0]:.4f}, {max_reach_pos[1]:.4f}, {max_reach_pos[2]:.4f}]")
            workspace_data['max_reach'] = max_reach
        else:
            print("\nCould not estimate maximum reach (no samples processed?).")
        
        # Min Reach
        if min_reach_pos is not None:
            print(f"\nEstimated Minimum Reach: {min_reach:.4f} meters")
            print(f"  Found at position: [{min_reach_pos[0]:.4f}, {min_reach_pos[1]:.4f}, {min_reach_pos[2]:.4f}]")
            workspace_data['min_reach'] = min_reach
        else:
            print("\nCould not estimate minimum reach (no samples processed?).")

        # Save to JSON file if data exists
        if workspace_data:
            try:
                with open(CONFIG_FILENAME, 'w') as f:
                    json.dump(workspace_data, f, indent=4)
                print(f"\nWorkspace dimensions saved to {CONFIG_FILENAME}")
            except IOError as e:
                print(f"\nError saving workspace config to {CONFIG_FILENAME}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        if physics_client >= 0 and p.isConnected(physics_client):
            print("Disconnecting PyBullet.")
            p.disconnect(physics_client)

if __name__ == "__main__":
    main() 