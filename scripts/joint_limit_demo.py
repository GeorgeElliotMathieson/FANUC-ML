import pybullet as p # type: ignore
import pybullet_data # type: ignore
import time
import os
import numpy as np
import logging # <-- Add logging import
import sys # <-- Add sys import

# Import robot config constants
from config import robot_config

# --- Constants (adapted from fanuc_env.py) ---
# Use constants imported from robot_config
JOINT_LIMITS_LOWER = robot_config.JOINT_LIMITS_LOWER_RAD
JOINT_LIMITS_UPPER = robot_config.JOINT_LIMITS_UPPER_RAD
NUM_CONTROLLABLE_JOINTS = robot_config.NUM_CONTROLLED_JOINTS
SIMULATION_TIME_STEP = 1.0/240.0
MOVEMENT_DURATION_STEPS = 240 # Number of sim steps to move (e.g., 1 second at 240Hz)
PAUSE_DURATION = 0.5 # Seconds to pause at limits/home

# Define paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
# Point to the new assets directory
ROBOT_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "robot_model")
URDF_FILENAME = os.path.join(ROBOT_MODEL_DIR, "urdf", "Fanuc.urdf")

# --- Configure Logging --- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger for this module

def main():
    """Runs the joint limit demonstration."""
    physics_client = -1
    try:
        # --- PyBullet Setup ---
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Set camera position for better view
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-15, cameraTargetPosition=[0,0,0.4])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIME_STEP)

        # Load plane
        p.loadURDF("plane.urdf")

        # --- Load Robot --- 
        # Use paths defined globally
        if not os.path.exists(URDF_FILENAME):
             logger.error(f"Error: Cannot find URDF file at {URDF_FILENAME}")
             return

        # Add Fanuc directory to search path *before* loading URDF
        p.setAdditionalSearchPath(ROBOT_MODEL_DIR)

        logger.info(f"Loading robot from: {URDF_FILENAME}")
        robot_id = p.loadURDF(
            URDF_FILENAME,
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )
        logger.info(f"Robot loaded with ID: {robot_id}")

        # --- Get Robot Info ---
        num_joints = p.getNumJoints(robot_id)
        joint_indices = []
        joint_names = [] # Store names for printing
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_type = info[2]
            joint_name = info[1].decode('UTF-8')
            # Identify controllable joints (assuming REVOLUTE type in URDF corresponds to controllable axes)
            if joint_type == p.JOINT_REVOLUTE:
                if len(joint_indices) < NUM_CONTROLLABLE_JOINTS:
                    joint_indices.append(i)
                    joint_names.append(joint_name)

        if len(joint_indices) != NUM_CONTROLLABLE_JOINTS:
            # Use logger.error
            logger.error(f"Error: Found {len(joint_indices)} controllable revolute joints, expected {NUM_CONTROLLABLE_JOINTS}.")
            logger.error(f"Found joints: {joint_names}")
            return

        # Use logger.info
        logger.info(f"Found {NUM_CONTROLLABLE_JOINTS} controllable joints:")
        for i in range(NUM_CONTROLLABLE_JOINTS):
             # Use logger.info
             logger.info(f"  - {joint_names[i]} (Index: {joint_indices[i]}) - Limits: [{JOINT_LIMITS_LOWER[i]:.2f}, {JOINT_LIMITS_UPPER[i]:.2f}] rad")

        # Use logger.info
        logger.info("\nResetting robot to home position...")
        home_position = [0.0] * NUM_CONTROLLABLE_JOINTS
        for i, joint_index in enumerate(joint_indices):
             p.resetJointState(robot_id, joint_index, targetValue=home_position[i], targetVelocity=0)
        # Let simulation settle briefly
        for _ in range(50):
             p.stepSimulation()
             time.sleep(SIMULATION_TIME_STEP)

        # Use logger.info
        logger.info("Starting joint limit demonstration...")

        # --- Animation Loop ---
        for i in range(NUM_CONTROLLABLE_JOINTS):
            current_joint_pybullet_index = joint_indices[i]
            current_joint_name = joint_names[i]
            # Use logger.info
            logger.info(f"\n--- Testing Joint {i+1}: {current_joint_name} (Index: {current_joint_pybullet_index}) ---")

            # --- Move to Lower Limit ---
            # Use logger.info
            logger.info(f"  -> Moving to Lower Limit ({JOINT_LIMITS_LOWER[i]:.2f} rad)")
            target_pos = list(home_position) # Start from home for other joints
            target_pos[i] = JOINT_LIMITS_LOWER[i]
            # Use setJointMotorControlArray for controlling multiple joints simultaneously
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
                # Optional: Add forces/maxVelocity if needed for smooth control
                # forces=[100]*NUM_CONTROLLABLE_JOINTS
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at the limit

            # --- Move to Upper Limit ---
            # Use logger.info
            logger.info(f"  -> Moving to Upper Limit ({JOINT_LIMITS_UPPER[i]:.2f} rad)")
            target_pos = list(home_position) # Start from home for other joints
            target_pos[i] = JOINT_LIMITS_UPPER[i]
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at the limit

            # --- Return to Home Position ---
            # Use logger.info
            logger.info("  -> Returning to Home Position")
            target_pos = list(home_position)
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at home


        # Use logger.info
        logger.info("\nDemonstration finished.")
        time.sleep(3) # Keep window open briefly

    except Exception as e:
        # Use logger.error
        logger.error(f"An error occurred: {e}")
    finally:
        # --- Cleanup ---
        if physics_client >= 0 and p.isConnected(physics_client):
             # Use logger.info
             logger.info("Disconnecting PyBullet.")
             p.disconnect(physics_client)

if __name__ == "__main__":
    # REMOVE sys.path adjustment - not needed when run from scripts/
    # SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
    # if SRC_DIR not in sys.path:
    #      sys.path.insert(0, SRC_DIR)

    main() 