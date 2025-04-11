import pybullet as p # type: ignore
import pybullet_data # type: ignore
import numpy as np
import os
import time
import json
import logging
import traceback
import sys

# Import robot config
from config import robot_config

# Constants from robot_config
JOINT_LIMITS_LOWER = robot_config.JOINT_LIMITS_LOWER_RAD
JOINT_LIMITS_UPPER = robot_config.JOINT_LIMITS_UPPER_RAD
NUM_CONTROLLABLE_JOINTS = robot_config.NUM_CONTROLLED_JOINTS
NUM_SAMPLES_PASS_1 = 100000
NUM_SAMPLES_PASS_2 = 50000
MIDPOINT_Z_TOLERANCE = 0.05
END_EFFECTOR_LINK_NAME = robot_config.END_EFFECTOR_LINK_NAME

# Project paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
CONFIG_FILENAME = os.path.join(PROJECT_ROOT, "config", "workspace_config.json")
ROBOT_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "robot_model")
URDF_FILENAME = os.path.join(ROBOT_MODEL_DIR, "urdf", "Fanuc.urdf")

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def sample_reach(robot_id, joint_indices, end_effector_link_index, num_samples, z_target_slice=None, z_tolerance=0.0):
    """
    Samples joint configurations and finds min/max reach.

    Args:
        robot_id: Robot ID.
        joint_indices: Controllable joint indices.
        end_effector_link_index: End-effector link index.
        num_samples: Number of samples.
        z_target_slice (float, optional): Target Z height (m). Defaults to None.
        z_tolerance (float, optional): Z-slice tolerance (+/- m). Defaults to 0.0.

    Returns:
        dict: Reach results.
    """
    max_r = 0.0
    z_at_max_r = 0.0
    min_r = float('inf')
    max_reach_in_slice = 0.0
    samples_in_slice = 0

    logger.info(f"Starting sampling pass: {num_samples} samples...")
    if z_target_slice is not None:
        logger.info(f"  Targeting Z-slice: {z_target_slice:.4f} +/- {z_tolerance:.4f} meters")

    start_time = time.time()
    for i in range(num_samples):
        # Random joint angles
        random_joint_angles = np.random.uniform(low=JOINT_LIMITS_LOWER, high=JOINT_LIMITS_UPPER)

        # Set joint states
        for joint_idx, angle in zip(joint_indices, random_joint_angles):
            p.resetJointState(robot_id, joint_idx, targetValue=angle, targetVelocity=0)

        # Get EE Position
        link_state = p.getLinkState(robot_id, end_effector_link_index)
        ee_position = np.array(link_state[4])
        ee_x, ee_y, ee_z = ee_position

        # Radial distance
        radial_distance = np.linalg.norm(ee_position)

        # Update overall max reach
        if radial_distance > max_r:
            max_r = radial_distance
            z_at_max_r = ee_z

        # Update overall min reach
        if radial_distance < min_r:
            min_r = radial_distance

        # Check Z-slice
        if z_target_slice is not None:
            if abs(ee_z - z_target_slice) <= z_tolerance:
                samples_in_slice += 1
                # Update max reach within slice
                if radial_distance > max_reach_in_slice:
                    max_reach_in_slice = radial_distance

        # Progress
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
        'reach_at_midpoint_z': max_reach_in_slice if z_target_slice is not None else 0.0
    }
    return results


def main():
    """Samples joint configurations to estimate workspace reach."""
    physics_client = -1
    robot_id = -1

    try:
        # PyBullet setup
        logger.info("Connecting to PyBullet in DIRECT mode...")
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot
        if not os.path.exists(URDF_FILENAME):
            logger.error(f"Cannot find URDF file at {URDF_FILENAME}")
            return

        p.setAdditionalSearchPath(ROBOT_MODEL_DIR)
        logger.info(f"Loading robot from: {URDF_FILENAME}")
        robot_id = p.loadURDF(
            URDF_FILENAME,
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        logger.info(f"Robot loaded with ID: {robot_id}")

        # Get robot info
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

        # Sampling Pass 1
        logger.info("\n--- Starting Sampling Pass 1: Finding Overall Reach ---")
        results_pass1 = sample_reach(robot_id, joint_indices, end_effector_link_index, NUM_SAMPLES_PASS_1)

        min_reach_overall = results_pass1['min_reach']
        max_reach_overall = results_pass1['max_reach']
        z_at_max_reach = results_pass1['z_at_max_reach']

        if max_reach_overall <= 0:
            logger.error("Could not determine maximum reach in Pass 1. Aborting.")
            return

        logger.info(f"Pass 1 Results: Min Reach={min_reach_overall:.4f}, Max Reach={max_reach_overall:.4f} at Z={z_at_max_reach:.4f}")

        # Calculate Midpoint Z
        z_midpoint = z_at_max_reach / 2.0
        logger.info(f"\nCalculated Midpoint Z: {z_midpoint:.4f} (based on Z at max reach)")

        # Sampling Pass 2
        logger.info(f"\n--- Starting Sampling Pass 2: Finding Reach near Midpoint Z ({z_midpoint:.4f}) ---")
        results_pass2 = sample_reach(
            robot_id,
            joint_indices,
            end_effector_link_index,
            NUM_SAMPLES_PASS_2,
            z_target_slice=z_midpoint,
            z_tolerance=MIDPOINT_Z_TOLERANCE
        )

        reach_at_midpoint = results_pass2['reach_at_midpoint_z']

        if reach_at_midpoint <= 0:
            logger.warning(f"Could not find samples near Z midpoint or reach was zero in Pass 2.")
            reach_at_midpoint = 0.0
        else:
             logger.info(f"Pass 2 Results: Max reach near Z midpoint ({z_midpoint:.4f} +/- {MIDPOINT_Z_TOLERANCE:.4f}) = {reach_at_midpoint:.4f}")


        # Output and Save Results
        logger.info("\n--- Final Workspace Estimation ---")
        logger.info(f"  Absolute Minimum Reach: {min_reach_overall:.4f} meters")
        logger.info(f"  Absolute Maximum Reach: {max_reach_overall:.4f} meters (at Z={z_at_max_reach:.4f})")
        logger.info(f"  Estimated Reach near Midpoint Z ({z_midpoint:.4f}): {reach_at_midpoint:.4f} meters")

        workspace_data = {
            'min_reach': min_reach_overall,
            'max_reach': max_reach_overall,
            'z_at_max_reach': z_at_max_reach,
            'reach_at_midpoint_z': reach_at_midpoint
        }

        # Save to JSON
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
        # Cleanup
        if physics_client >= 0 and p.isConnected(physics_client):
            logger.info("Disconnecting PyBullet.")
            p.disconnect(physics_client)

if __name__ == "__main__":
    # REMOVED sys.path adjustment
    main() 