# config/robot_config.py
import numpy as np
import logging # Import logging

logger = logging.getLogger(__name__)

# --- Core Robot Parameters ---
# **CRITICAL**: Verify these values match your specific robot model and setup.

# Number of joints controlled by the RL policy/action space
NUM_CONTROLLED_JOINTS: int = 5

# Number of joints reported by the real robot's position feedback (e.g., RDJPOS)
NUM_REPORTED_JOINTS: int = 6 # Typically 6 for FANUC LRMate

# Joint Limits (Radians) - Order: J1, J2, J3, J4, J5
# Based on FANUC LRMate 200iD approximation used in src/fanuc_env.py
# Ensure this order matches the joint indices used in PyBullet and the real robot API.
JOINT_LIMITS_LOWER_RAD: np.ndarray = np.array([-2.96, -1.74, -2.37, -3.31, -2.18], dtype=np.float32)
JOINT_LIMITS_UPPER_RAD: np.ndarray = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18], dtype=np.float32)

# Velocity Limits (Radians per Second) - Order: J1, J2, J3, J4, J5
# Estimated reasonable values, potentially scaled down for safety/smoothness.
# Used for clipping actions in transfer_learning and deploy_real safety checks.
VELOCITY_LIMITS_RAD_S: np.ndarray = np.array([3.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32) # Matches scaled-down values from fanuc_env

# End Effector Link Name (used in simulation and potentially for FK/safety checks)
END_EFFECTOR_LINK_NAME: str = 'Part6' # Matches URDF

# --- Optional: Derived Parameters ---
# Joint Limits in Degrees (for convenience, e.g., in robot_api)
JOINT_LIMITS_LOWER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_LOWER_RAD)
JOINT_LIMITS_UPPER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_UPPER_RAD)

# Add limits for the 6th joint (degrees) for robot_api - **VERIFY THESE LIMITS**
JOINT_6_LIMIT_DEG_LOW: float = -720.0 # Example: +/- 720 deg
JOINT_6_LIMIT_DEG_HIGH: float = 720.0

# Combine limits for all reported joints (degrees)
REPORTED_JOINT_LIMITS_DEG = {
    i: [JOINT_LIMITS_LOWER_DEG[i], JOINT_LIMITS_UPPER_DEG[i]] for i in range(NUM_CONTROLLED_JOINTS)
}
# Add J6 limit if applicable
if NUM_REPORTED_JOINTS > NUM_CONTROLLED_JOINTS:
     # Assuming the 6th joint is index 5
     REPORTED_JOINT_LIMITS_DEG[NUM_CONTROLLED_JOINTS] = [JOINT_6_LIMIT_DEG_LOW, JOINT_6_LIMIT_DEG_HIGH]

logger.debug("Robot configuration loaded.") 