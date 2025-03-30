import pybullet as p # type: ignore
import pybullet_data # type: ignore
import numpy as np
import os
import time
import json # Import json
import logging # Import logging
import traceback # Import traceback
import sys

# --- Constants (Adapted from fanuc_env.py) ---
# Ensure these match your fanuc_env.py
JOINT_LIMITS_LOWER = np.array([-2.96, -1.74, -2.37, -3.31, -2.18]) # J1-J5 radians
JOINT_LIMITS_UPPER = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18]) # J1-J5 radians
NUM_CONTROLLABLE_JOINTS = 5
NUM_SAMPLES = 100000 # Increased from 20000
END_EFFECTOR_LINK_NAME = 'Part6' # Ensure this matches the link name in your URDF/env

# Define paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
CONFIG_FILENAME = os.path.join(PROJECT_ROOT, "workspace_config.json")
FANUC_DIR = os.path.join(PROJECT_ROOT, "Fanuc")
URDF_FILENAME = os.path.join(FANUC_DIR, "urdf", "Fanuc.urdf")

# --- Configure Logging --- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        logger.info("Connecting to PyBullet in DIRECT mode...")
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # No gravity needed for pure FK check
        # p.setGravity(0, 0, -9.81)

        # --- Load Robot ---
        if not os.path.exists(URDF_FILENAME):
            logger.error(f"Cannot find URDF file at {URDF_FILENAME}")
            return

        p.setAdditionalSearchPath(FANUC_DIR)
        logger.info(f"Loading robot from: {URDF_FILENAME}")
        robot_id = p.loadURDF(
            URDF_FILENAME,
            basePosition=[0, 0, 0],
            useFixedBase=True
            # No flags needed like self-collision for FK check
        )
        logger.info(f"Robot loaded with ID: {robot_id}")

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
            logger.error(f"Found {len(joint_indices)} controllable revolute joints, expected {NUM_CONTROLLABLE_JOINTS}.")
            return
        if end_effector_link_index == -1:
            logger.error(f"Could not find end effector link named '{END_EFFECTOR_LINK_NAME}'. Using last joint index as fallback.")
            # Fallback: Use the link associated with the last *controllable* joint index found
            if joint_indices:
                 end_effector_link_index = joint_indices[-1]
            else: # Or ultimately the last joint if no revolute found (unlikely)
                 end_effector_link_index = num_joints_total - 1
            logger.info(f"Using end effector link index: {end_effector_link_index}")


        logger.info(f"Using End Effector Link Index: {end_effector_link_index}")
        logger.info(f"Sampling {NUM_SAMPLES} random joint configurations...")

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
            # Use link origin [4] instead of CoM [0] for EE position
            link_state = p.getLinkState(robot_id, end_effector_link_index) # computeForwardKinematics=0 by default
            ee_position = np.array(link_state[4]) # World position of the link origin

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
                logger.info(f"  Processed {i+1}/{NUM_SAMPLES} samples...")

        end_time = time.time()
        logger.info(f"Sampling finished in {end_time - start_time:.2f} seconds.")

        # --- Output and Save Results ---
        workspace_data = {}
        # Max Reach
        if max_reach_pos is not None:
            logger.info(f"\nEstimated Maximum Reach: {max_reach:.4f} meters")
            logger.info(f"  Found at position: [{max_reach_pos[0]:.4f}, {max_reach_pos[1]:.4f}, {max_reach_pos[2]:.4f}]")
            workspace_data['max_reach'] = max_reach
        else:
            logger.warning("\nCould not estimate maximum reach (no samples processed?).")
        
        # Min Reach
        if min_reach_pos is not None:
            logger.info(f"\nEstimated Minimum Reach: {min_reach:.4f} meters")
            logger.info(f"  Found at position: [{min_reach_pos[0]:.4f}, {min_reach_pos[1]:.4f}, {min_reach_pos[2]:.4f}]")
            workspace_data['min_reach'] = min_reach
        else:
            logger.warning("\nCould not estimate minimum reach (no samples processed?).")

        # Save to JSON file if data exists
        if workspace_data:
            try:
                with open(CONFIG_FILENAME, 'w') as f:
                    json.dump(workspace_data, f, indent=4)
                logger.info(f"\nWorkspace dimensions saved to {CONFIG_FILENAME}")
            except IOError as e:
                logger.error(f"\nError saving workspace config to {CONFIG_FILENAME}: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        # --- Cleanup ---
        if physics_client >= 0 and p.isConnected(physics_client):
            logger.info("Disconnecting PyBullet.")
            p.disconnect(physics_client)

if __name__ == "__main__":
    # REMOVE sys.path adjustment - not needed when run from scripts/
    # SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
    # if SRC_DIR not in sys.path:
    #      sys.path.insert(0, SRC_DIR)

    main() 