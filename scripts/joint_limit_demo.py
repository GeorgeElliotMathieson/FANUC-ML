import pybullet as p # type: ignore
import pybullet_data # type: ignore
import time
import os
import numpy as np
import logging
import sys

# Import robot config constants
from config import robot_config

# Constants
JOINT_LIMITS_LOWER = robot_config.JOINT_LIMITS_LOWER_RAD
JOINT_LIMITS_UPPER = robot_config.JOINT_LIMITS_UPPER_RAD
NUM_CONTROLLABLE_JOINTS = robot_config.NUM_CONTROLLED_JOINTS
SIMULATION_TIME_STEP = 1.0/240.0
# Movement duration (steps)
MOVEMENT_DURATION_STEPS = 240
# Pause duration (s)
PAUSE_DURATION = 0.5

# Project root
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
# Robot model directory
ROBOT_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "robot_model")
URDF_FILENAME = os.path.join(ROBOT_MODEL_DIR, "urdf", "Fanuc.urdf")

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Runs the joint limit demonstration."""
    physics_client = -1
    try:
        # PyBullet Setup
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-15, cameraTargetPosition=[0,0,0.4])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIME_STEP)

        # Load plane
        p.loadURDF("plane.urdf")

        # Load Robot
        if not os.path.exists(URDF_FILENAME):
             logger.error(f"Error: Cannot find URDF file at {URDF_FILENAME}")
             return

        # Add search path
        p.setAdditionalSearchPath(ROBOT_MODEL_DIR)

        logger.info(f"Loading robot from: {URDF_FILENAME}")
        robot_id = p.loadURDF(
            URDF_FILENAME,
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )
        logger.info(f"Robot loaded with ID: {robot_id}")

        # Robot Info
        num_joints = p.getNumJoints(robot_id)
        joint_indices = []
        joint_names = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_type = info[2]
            joint_name = info[1].decode('UTF-8')
            # Identify controllable joints
            if joint_type == p.JOINT_REVOLUTE:
                if len(joint_indices) < NUM_CONTROLLABLE_JOINTS:
                    joint_indices.append(i)
                    joint_names.append(joint_name)

        if len(joint_indices) != NUM_CONTROLLABLE_JOINTS:
            logger.error(f"Error: Found {len(joint_indices)} controllable revolute joints, expected {NUM_CONTROLLABLE_JOINTS}.")
            logger.error(f"Found joints: {joint_names}")
            return

        logger.info(f"Found {NUM_CONTROLLABLE_JOINTS} controllable joints:")
        for i in range(NUM_CONTROLLABLE_JOINTS):
             logger.info(f"  - {joint_names[i]} (Index: {joint_indices[i]}) - Limits: [{JOINT_LIMITS_LOWER[i]:.2f}, {JOINT_LIMITS_UPPER[i]:.2f}] rad")

        logger.info("Resetting robot to home position...")
        home_position = [0.0] * NUM_CONTROLLABLE_JOINTS
        for i, joint_index in enumerate(joint_indices):
             p.resetJointState(robot_id, joint_index, targetValue=home_position[i], targetVelocity=0)
        # Settle simulation
        for _ in range(50):
             p.stepSimulation()
             time.sleep(SIMULATION_TIME_STEP)

        logger.info("Starting joint limit demonstration...")

        # Animation Loop
        for i in range(NUM_CONTROLLABLE_JOINTS):
            current_joint_pybullet_index = joint_indices[i]
            current_joint_name = joint_names[i]
            logger.info(f"--- Testing Joint {i+1}: {current_joint_name} (Index: {current_joint_pybullet_index}) ---")

            # Move to Lower Limit
            logger.info(f"  -> Moving to Lower Limit ({JOINT_LIMITS_LOWER[i]:.2f} rad)")
            target_pos = list(home_position)
            target_pos[i] = JOINT_LIMITS_LOWER[i]
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION)

            # Move to Upper Limit
            logger.info(f"  -> Moving to Upper Limit ({JOINT_LIMITS_UPPER[i]:.2f} rad)")
            target_pos = list(home_position)
            target_pos[i] = JOINT_LIMITS_UPPER[i]
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION)

            # Return to Home Position
            logger.info("  -> Returning to Home Position")
            target_pos = list(home_position)
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION)


        logger.info("Demonstration finished.")
        time.sleep(3)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Cleanup
        if physics_client >= 0 and p.isConnected(physics_client):
             logger.info("Disconnecting PyBullet.")
             p.disconnect(physics_client)

if __name__ == "__main__":
    main() 