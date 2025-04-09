# config/robot_config.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Core Robot Setup ---
# IMPORTANT: Double-check these values against your specific robot.

# Joints controlled by the policy (action space size).
NUM_CONTROLLED_JOINTS: int = 5

# Joints reported by the robot's feedback (e.g., RDJPOS).
NUM_REPORTED_JOINTS: int = 6 # Typically 6 for FANUC LRMate

# Joint limits [rad]. Order: J1-J5.
# Based on LRMate 200iD approximation (see src/fanuc_env.py).
# Order MUST match PyBullet and robot API joint indices.
JOINT_LIMITS_LOWER_RAD: np.ndarray = np.array([-2.96, -1.74, -2.37, -3.31, -2.18], dtype=np.float32)
JOINT_LIMITS_UPPER_RAD: np.ndarray = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18], dtype=np.float32)

# Velocity limits [rad/s]. Order: J1-J5.
# Estimated values, maybe scaled down for safety.
# Used for action clipping and safety checks.
VELOCITY_LIMITS_RAD_S: np.ndarray = np.array([3.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32) # Matches scaled-down values from fanuc_env

# End effector link name (for simulation/FK).
END_EFFECTOR_LINK_NAME: str = 'Part6' # Matches URDF

# --- Derived Parameters (Convenience) ---
# Joint limits [deg] (derived for convenience).
JOINT_LIMITS_LOWER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_LOWER_RAD)
JOINT_LIMITS_UPPER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_UPPER_RAD)

# Define limits for the 6th joint [deg] if used - **VERIFY THESE**.
JOINT_6_LIMIT_DEG_LOW: float = -720.0
JOINT_6_LIMIT_DEG_HIGH: float = 720.0

# Combine limits for all reported joints [deg].
REPORTED_JOINT_LIMITS_DEG = {
    i: [JOINT_LIMITS_LOWER_DEG[i], JOINT_LIMITS_UPPER_DEG[i]] for i in range(NUM_CONTROLLED_JOINTS)
}
# Add J6 limits if needed.
if NUM_REPORTED_JOINTS > NUM_CONTROLLED_JOINTS:
     # Assumes J6 is index 5.
     REPORTED_JOINT_LIMITS_DEG[NUM_CONTROLLED_JOINTS] = [JOINT_6_LIMIT_DEG_LOW, JOINT_6_LIMIT_DEG_HIGH]

logger.debug("Robot configuration loaded.") 