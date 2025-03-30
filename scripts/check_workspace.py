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
NUM_SAMPLES_PASS_1 = 100000 # Samples for initial max reach and Z
NUM_SAMPLES_PASS_2 = 50000  # Samples specifically for reach at midpoint Z (can be less)
MIDPOINT_Z_TOLERANCE = 0.05 # +/- tolerance in meters for Z midpoint slice
END_EFFECTOR_LINK_NAME = 'Part6' # Ensure this matches the link name in your URDF/env

# Define paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
# Point to the new config directory
CONFIG_FILENAME = os.path.join(PROJECT_ROOT, "config", "workspace_config.json")
FANUC_DIR = os.path.join(PROJECT_ROOT, "Fanuc")
URDF_FILENAME = os.path.join(FANUC_DIR, "urdf", "Fanuc.urdf")

# --- Configure Logging --- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def sample_reach(robot_id, joint_indices, end_effector_link_index, num_samples, z_target_slice=None, z_tolerance=0.0):
    """
    Helper function to sample joint configurations and find min/max reach.
    Can optionally focus on finding max reach within a specific Z-slice.

    Args:
        robot_id: PyBullet robot ID.
        joint_indices: List of controllable joint indices.
        end_effector_link_index: Index of the end-effector link.
        num_samples: Number of random configurations to sample.
        z_target_slice (float, optional): Target Z height to focus on. If None, finds overall min/max. Defaults to None.
        z_tolerance (float, optional): Tolerance (+/-) for the Z-slice. Defaults to 0.0.

    Returns:
        dict: Dictionary containing reach results ('min_reach', 'max_reach', 'z_at_max_reach', etc.).
    """
    max_r = 0.0
    z_at_max_r = 0.0
    min_r = float('inf')
    # Specific max reach within the Z slice
    max_reach_in_slice = 0.0
    samples_in_slice = 0

    logger.info(f"Starting sampling pass: {num_samples} samples...")
    if z_target_slice is not None:
        logger.info(f"  Targeting Z-slice: {z_target_slice:.4f} +/- {z_tolerance:.4f} meters")

    start_time = time.time()
    for i in range(num_samples):
        # Generate random joint angles within limits
        random_joint_angles = np.random.uniform(low=JOINT_LIMITS_LOWER, high=JOINT_LIMITS_UPPER)

        # Set the robot to this configuration using resetJointState
        for joint_idx, angle in zip(joint_indices, random_joint_angles):
            p.resetJointState(robot_id, joint_idx, targetValue=angle, targetVelocity=0)

        # Get End Effector Position using Forward Kinematics
        link_state = p.getLinkState(robot_id, end_effector_link_index)
        ee_position = np.array(link_state[4]) # World position of the link origin
        ee_x, ee_y, ee_z = ee_position

        # Calculate distance from base (origin)
        radial_distance = np.linalg.norm(ee_position)

        # Update overall max reach and its Z coordinate (always track this)
        if radial_distance > max_r:
            max_r = radial_distance
            z_at_max_r = ee_z

        # Update overall min reach (always track this)
        if radial_distance < min_r:
            min_r = radial_distance

        # If targeting a Z-slice, check if this sample is within it
        if z_target_slice is not None:
            if abs(ee_z - z_target_slice) <= z_tolerance:
                samples_in_slice += 1
                # Update the max radial reach found *within this slice*
                if radial_distance > max_reach_in_slice:
                    max_reach_in_slice = radial_distance

        # Progress indicator
        if (i + 1) % (num_samples // 10 if num_samples >= 10 else 1) == 0:
            logger.info(f"  ...processed {i+1}/{num_samples} samples...")

    end_time = time.time()
    logger.info(f"Sampling pass finished in {end_time - start_time:.2f} seconds.")
    if z_target_slice is not None:
        logger.info(f"  Found {samples_in_slice} samples within the Z-slice.")

    results = {
        'min_reach': min_r if min_r != float('inf') else 0.0,
        'max_reach': max_r,
        'z_at_max_reach': z_at_max_r,
        'reach_at_midpoint_z': max_reach_in_slice if z_target_slice is not None else 0.0 # Rename for clarity later if needed
    }
    return results


def main():
    """Samples joint configurations to estimate workspace reach."""
    physics_client = -1
    robot_id = -1

    try:
        # --- PyBullet Setup (Direct Mode) ---
        logger.info("Connecting to PyBullet in DIRECT mode...")
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

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
            logger.warning(f"Could not find end effector link named '{END_EFFECTOR_LINK_NAME}'. Using last joint index as fallback.")
            end_effector_link_index = joint_indices[-1] if joint_indices else num_joints_total - 1

        logger.info(f"Using End Effector Link Index: {end_effector_link_index}")

        # --- Sampling Pass 1: Find Overall Min/Max Reach and Z at Max ---
        logger.info("\n--- Starting Sampling Pass 1: Finding Overall Reach ---")
        results_pass1 = sample_reach(robot_id, joint_indices, end_effector_link_index, NUM_SAMPLES_PASS_1)

        min_reach_overall = results_pass1['min_reach']
        max_reach_overall = results_pass1['max_reach']
        z_at_max_reach = results_pass1['z_at_max_reach']

        if max_reach_overall <= 0:
            logger.error("Could not determine maximum reach in Pass 1. Aborting.")
            return

        logger.info(f"Pass 1 Results: Min Reach={min_reach_overall:.4f}, Max Reach={max_reach_overall:.4f} at Z={z_at_max_reach:.4f}")

        # --- Calculate Midpoint Z ---
        z_midpoint = z_at_max_reach / 2.0
        logger.info(f"\nCalculated Midpoint Z: {z_midpoint:.4f} (based on Z at max reach)")

        # --- Sampling Pass 2: Find Max Reach near Midpoint Z ---
        logger.info(f"\n--- Starting Sampling Pass 2: Finding Reach near Midpoint Z ({z_midpoint:.4f}) ---")
        results_pass2 = sample_reach(
            robot_id,
            joint_indices,
            end_effector_link_index,
            NUM_SAMPLES_PASS_2,
            z_target_slice=z_midpoint,
            z_tolerance=MIDPOINT_Z_TOLERANCE
        )

        # The 'reach_at_midpoint_z' key holds the result from this pass
        reach_at_midpoint = results_pass2['reach_at_midpoint_z']

        if reach_at_midpoint <= 0:
            logger.warning(f"Could not find samples near Z midpoint or reach was zero in Pass 2.")
            # Fallback: use overall max reach? Or keep it 0? Let's keep it 0 and handle in env.
            reach_at_midpoint = 0.0 # Explicitly set to 0 if not found
        else:
             logger.info(f"Pass 2 Results: Max reach near Z midpoint ({z_midpoint:.4f} +/- {MIDPOINT_Z_TOLERANCE:.4f}) = {reach_at_midpoint:.4f}")


        # --- Output and Save Combined Results ---
        logger.info("\n--- Final Workspace Estimation ---")
        logger.info(f"  Absolute Minimum Reach: {min_reach_overall:.4f} meters")
        logger.info(f"  Absolute Maximum Reach: {max_reach_overall:.4f} meters (at Z={z_at_max_reach:.4f})")
        logger.info(f"  Estimated Reach near Midpoint Z ({z_midpoint:.4f}): {reach_at_midpoint:.4f} meters")

        workspace_data = {
            'min_reach': min_reach_overall,
            'max_reach': max_reach_overall, # Keep the absolute max for reference
            'z_at_max_reach': z_at_max_reach, # Store the Z where absolute max occurred
            'reach_at_midpoint_z': reach_at_midpoint # Store the reach found near the midpoint
        }

        # Save to JSON file
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