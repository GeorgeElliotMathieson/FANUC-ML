# config/robot_config.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Number of joints for control
NUM_CONTROLLED_JOINTS: int = 5

# Number of joints reported
NUM_REPORTED_JOINTS: int = 6

# Joint limits (rad)
JOINT_LIMITS_LOWER_RAD: np.ndarray = np.array([-2.96, -1.74, -2.37, -3.31, -2.18], dtype=np.float32)
JOINT_LIMITS_UPPER_RAD: np.ndarray = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18], dtype=np.float32)

# Velocity limits (rad/s)
VELOCITY_LIMITS_RAD_S: np.ndarray = np.array([3.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32)

# End effector link name (sim/FK)
END_EFFECTOR_LINK_NAME: str = 'Part6'

# Joint limits (deg)
JOINT_LIMITS_LOWER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_LOWER_RAD)
JOINT_LIMITS_UPPER_DEG: np.ndarray = np.rad2deg(JOINT_LIMITS_UPPER_RAD)

# J6 limits (deg)
JOINT_6_LIMIT_DEG_LOW: float = -720.0
JOINT_6_LIMIT_DEG_HIGH: float = 720.0

# All reported joint limits (deg)
REPORTED_JOINT_LIMITS_DEG = {
    i: [JOINT_LIMITS_LOWER_DEG[i], JOINT_LIMITS_UPPER_DEG[i]] for i in range(NUM_CONTROLLED_JOINTS)
}
# Add J6 limits if needed
if NUM_REPORTED_JOINTS > NUM_CONTROLLED_JOINTS:
     REPORTED_JOINT_LIMITS_DEG[NUM_CONTROLLED_JOINTS] = [JOINT_6_LIMIT_DEG_LOW, JOINT_6_LIMIT_DEG_HIGH]

logger.debug("Robot configuration loaded.") 