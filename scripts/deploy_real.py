# scripts/deploy_real.py
import time
import argparse
import logging
import numpy as np
import os
import sys
import traceback
import json # Import json for loading workspace config
import math # Import math for spherical coords
import collections # <-- Import collections for deque
from typing import List, Optional, Any, Tuple # Import List, Optional, Any, Tuple for type hinting
import pybullet as p # type: ignore # Import pybullet for FK calculation
import pybullet_data # type: ignore

# Import robot config constants
from config import robot_config

# Import custom modules (ensure src is in PYTHONPATH)
# Add src directory to path temporarily if running script directly

# --- Project Setup ---
# Ensure the src directory is in the Python path
# Allows importing modules from src when running scripts/deploy_real.py
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    # Now we can import from src
    from src.deployment.robot_api import FANUCRobotAPI
    from src.deployment.transfer_learning import RobotTransfer
except ImportError as e:
     print(f"Error importing src modules: {e}")
     print("Ensure deploy_real.py is in the 'scripts' directory and src is accessible.")
     sys.exit(1)

# --- Constants and Configuration ---
# Define defaults locally, matching robot_api.py
DEFAULT_ROBOT_IP = '192.168.1.10' # Default IP from instructions
DEFAULT_ROBOT_PORT = 6000       # Default Port from instructions

# Default path assumes model saved by src/train.py
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'output', 'ppo_logs', 'ppo_fanuc_model.zip')
DEFAULT_LOOP_RATE_HZ = 10.0 # Target control loop frequency
DEFAULT_CONTROL_MODE = 'position' # 'position' or 'velocity'
DEFAULT_POS_SPEED_PERCENT = 20 # Speed for position moves

# SAFETY: Define safe operational boundaries (Units: METERS and RADIANS)
# ** CRITICAL: DEFINE ACCURATE SAFETY LIMITS FOR YOUR SETUP **
WORKSPACE_LIMITS_XYZ_MIN = np.array([0.1, -0.5, 0.05]) # Min X, Y, Z (meters)
WORKSPACE_LIMITS_XYZ_MAX = np.array([0.8, 0.5, 0.8])  # Max X, Y, Z (meters)

# Load velocity limits from config
MAX_JOINT_VEL_RAD_S = robot_config.VELOCITY_LIMITS_RAD_S

# Workspace boundaries (Example - **VERIFY AND UPDATE**)
# Used for safety checks if FK is available

# Calibration settings
CALIBRATION_POSES = {
    "home": np.zeros(5, dtype=np.float32),
    "pose1": np.deg2rad(np.array([30, 0, 30, 0, 0], dtype=np.float32)),
    "pose2": np.deg2rad(np.array([-30, 45, 0, 0, 30], dtype=np.float32)),
    # Add more representative poses as needed
}
CALIBRATION_SPEED = 15 # Speed percentage for calibration moves
CALIBRATION_PAUSE_S = 8 # Seconds to pause at each calibration pose for observation

# Config file for manual calibration params
PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'transfer_params.json')

# Config file for workspace limits (needed for target generation)
WORKSPACE_CONFIG_FILENAME = os.path.join(PROJECT_ROOT, "workspace_config.json")

# Target generation parameters (should match fanuc_env.py)
WORKSPACE_SAFETY_FACTOR = 0.95 # Apply to max radius
TARGET_MIN_Z_HEIGHT = 0.05 # Minimum allowed target Z coord

# Load workspace limits for target generation
MIN_BASE_RADIUS_DEFAULT = 0.02
MAX_REACH_DEFAULT = 1.26
MIDPOINT_REACH_DEFAULT = 0.0
try:
    with open(WORKSPACE_CONFIG_FILENAME, 'r') as f:
        ws_config = json.load(f)
    MIN_BASE_RADIUS = ws_config.get('min_reach', MIN_BASE_RADIUS_DEFAULT)
    MAX_REACH_ABS = ws_config.get('max_reach', MAX_REACH_DEFAULT)
    REACH_AT_MIDPOINT = ws_config.get('reach_at_midpoint_z', MIDPOINT_REACH_DEFAULT)
    logging.info(f"Loaded workspace config for target gen: MinR={MIN_BASE_RADIUS:.3f}, MaxAbsR={MAX_REACH_ABS:.3f}, MidZ_R={REACH_AT_MIDPOINT:.3f}")
except Exception as e:
    logging.warning(f"Could not load {WORKSPACE_CONFIG_FILENAME} for target generation: {e}. Using defaults.")
    MIN_BASE_RADIUS = MIN_BASE_RADIUS_DEFAULT
    MAX_REACH_ABS = MAX_REACH_DEFAULT
    REACH_AT_MIDPOINT = MIDPOINT_REACH_DEFAULT

# Determine the practical max radius for sampling
if REACH_AT_MIDPOINT > MIN_BASE_RADIUS: # Check if midpoint reach is valid
    TARGET_SAMPLING_MAX_RADIUS = REACH_AT_MIDPOINT * WORKSPACE_SAFETY_FACTOR
    logging.info(f"Using Midpoint Reach for Target Sampling Max Radius: {TARGET_SAMPLING_MAX_RADIUS:.3f}")
else:
    TARGET_SAMPLING_MAX_RADIUS = MAX_REACH_ABS * WORKSPACE_SAFETY_FACTOR
    logging.warning(f"Using Absolute Reach for Target Sampling Max Radius: {TARGET_SAMPLING_MAX_RADIUS:.3f}")
# Ensure max radius is greater than min base radius
TARGET_SAMPLING_MAX_RADIUS = max(TARGET_SAMPLING_MAX_RADIUS, MIN_BASE_RADIUS + 0.1) # Ensure some range
TARGET_SAMPLING_MIN_RADIUS = MIN_BASE_RADIUS * 5.0 # Use a smaller multiplier than env for potentially easier real targets
TARGET_SAMPLING_MIN_RADIUS = min(TARGET_SAMPLING_MIN_RADIUS, TARGET_SAMPLING_MAX_RADIUS * 0.5) # Ensure min is not too close to max
logging.info(f"Target Sampling Radius Range: [{TARGET_SAMPLING_MIN_RADIUS:.3f} - {TARGET_SAMPLING_MAX_RADIUS:.3f}] meters")

# --- Configure Logging ---
# Setup basic logging to console
# Consider adding a FileHandler for persistent logs
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger() # Get root logger
logger.setLevel(logging.INFO) # Set level (DEBUG, INFO, WARNING, ERROR)
logger.addHandler(log_handler)
# Optional: Add file handler
# file_handler = logging.FileHandler(os.path.join(PROJECT_ROOT, 'output', 'deploy_real.log'))
# file_handler.setFormatter(log_formatter)
# logger.addHandler(file_handler)

# ==========================================================================
# --- Helper Functions ---
# ==========================================================================

# Remove global cache and setup/cleanup for separate FK client
# _fk_client: Optional[int] = None
# _fk_robot_id: int = -1
# _fk_joint_indices: List[int] = []
# _fk_ee_link_index: int = -1

# def setup_fk_calculator():
#     """Initializes a PyBullet client in DIRECT mode for FK calculations."""
#     ...

# def cleanup_fk_calculator():
#     """Disconnects the PyBullet client used for FK."""
#     ...

# Modify FK function to accept client ID and robot ID
def calculate_forward_kinematics(joint_angles_rad: np.ndarray, physics_client_id: int, robot_id: int, joint_indices: List[int], ee_link_index: int) -> np.ndarray | None:
    """
    Calculates the End Effector position using PyBullet based on joint angles.
    Uses the provided physics client and robot ID.

    Args:
        joint_angles_rad: Array of 5 joint angles in radians.
        physics_client_id: The PyBullet physics client ID.
        robot_id: The PyBullet robot ID.
        joint_indices: List of controllable joint indices for the robot.
        ee_link_index: The end-effector link index for the robot.

    Returns:
        np.ndarray | None: The EE position [x, y, z] in meters (world frame),
                           or None if calculation fails.
    """
    # Removed check for global _fk_client
    if physics_client_id < 0 or robot_id == -1:
        logger.error("Invalid physics client or robot ID for FK calculation.")
        return None
    if len(joint_angles_rad) != len(joint_indices):
        logger.error(f"FK input dimension mismatch: Expected {len(joint_indices)}, got {len(joint_angles_rad)}")
        return None

    try:
        # Reset joint states in the specified simulation
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_index,
                              targetValue=joint_angles_rad[i],
                              targetVelocity=0,
                              physicsClientId=physics_client_id)

        # Get the link state (world position of the link origin)
        # Use computeForwardKinematics=True just to be sure
        link_state = p.getLinkState(robot_id, ee_link_index,
                                  computeForwardKinematics=True,
                                  physicsClientId=physics_client_id)

        ee_position_m = np.array(link_state[4], dtype=np.float32)
        return ee_position_m

    except Exception as e:
        logger.error(f"Error during FK calculation: {e}")
        logger.error(traceback.format_exc())
        return None

# ==========================================================================
# --- Placeholder Functions (Implement or Adapt) ---
# ==========================================================================

def run_calibration_observation_routine(robot: FANUCRobotAPI):
    """
    Guides the user through observing the robot at predefined poses.

    This function commands the robot to move to several static configurations
    and pauses, allowing the user to visually inspect the robot and mentally
    note discrepancies or required adjustments for sim-to-real transfer.

    It does NOT collect data automatically. The user is expected to manually
    create/edit the 'config/transfer_params.json' file afterwards.
    """
    logger.info("--- Starting Calibration Observation Routine ---")
    logger.info("The robot will move sequentially to several poses.")
    logger.info(f"At each pose, it will pause for {CALIBRATION_PAUSE_S} seconds.")
    logger.info("Observe the robot's posture and end-effector position carefully.")
    logger.info("Note any consistent deviations from expected simulation behavior.")
    logger.info("Use these observations to manually determine the correction parameters")
    logger.info(f"(state_mean, state_std, action_scale, action_offset) in {PARAMS_FILE_PATH}")
    logger.info("Press Ctrl+C to stop the routine early if needed.")
    time.sleep(5) # Give user time to read

    initial_pos_deg = robot.get_joint_positions() # Get current pos (degrees)
    if initial_pos_deg is None:
         logger.error("Failed to get initial robot position. Aborting calibration routine.")
         return
    # Note: CALIBRATION_POSES are already in radians, but move command takes degrees
    # We keep the dictionary keys for reference, but will convert pose_rad to deg for command

    # Add current pose (in degrees) to the list of poses to visit first
    poses_to_visit_deg = {"current": initial_pos_deg}
    # Add other poses, converting them to degrees for the move command
    for name, pose_rad in CALIBRATION_POSES.items():
        poses_to_visit_deg[name] = np.rad2deg(pose_rad)

    try:
        for name, pose_deg in poses_to_visit_deg.items():
            # The API now expects 6 joints based on RDJPOS parsing
            # Need to pad the 5-joint pose commands if robot reports 6
            if robot.num_reported_joints == 6 and len(pose_deg) == 5:
                pose_cmd_deg = np.append(pose_deg, 0.0) # Append 0 for 6th joint
            elif len(pose_deg) == robot.num_reported_joints:
                 pose_cmd_deg = pose_deg
            else:
                logger.warning(f"Skipping invalid calibration pose '{name}'. Dimension mismatch (expected {robot.num_reported_joints}).")
                continue

            logger.info(f"\nMoving to calibration pose: '{name}' ({np.round(pose_cmd_deg, 1)} deg)...")
            # Use move_to_joint_positions with degrees
            success = robot.move_to_joint_positions(
                positions_deg=pose_cmd_deg,
                speed_percent=CALIBRATION_SPEED
            )

            if not success:
                logger.error(f"Failed to send command for pose '{name}'. Skipping pause.")
                continue

            # --- PAUSE FOR OBSERVATION --- 
            logger.info(f"Pausing for {CALIBRATION_PAUSE_S} seconds. OBSERVE THE ROBOT at pose '{name}'.")
            # **TODO**: Add status check here to wait for move completion before pausing
            # e.g., while robot.is_moving(): time.sleep(0.1)
            time.sleep(CALIBRATION_PAUSE_S)
            logger.info(f"... observation pause finished for pose '{name}'.")

        # Optionally move back to home at the end?
        logger.info("Returning to home position...")
        home_pose_deg = np.rad2deg(CALIBRATION_POSES["home"])
        if robot.num_reported_joints == 6:
             home_cmd_deg = np.append(home_pose_deg, 0.0)
        else:
             home_cmd_deg = home_pose_deg

        robot.move_to_joint_positions(home_cmd_deg, speed_percent=CALIBRATION_SPEED)
        # **TODO**: Add status check here to wait for move completion
        time.sleep(CALIBRATION_PAUSE_S / 2) # Shorter pause after returning home

    except KeyboardInterrupt:
        logger.warning("Calibration routine interrupted by user.")
        logger.info("Attempting to stop motion...")
        robot.stop_motion()

    except Exception as e:
         logger.error(f"An error occurred during the calibration routine: {e}")
         logger.error(traceback.format_exc())
         logger.info("Attempting to stop motion due to error...")
         robot.stop_motion()

    logger.info("--- Calibration Observation Routine Finished ---")
    logger.info("Remember to manually create/update the correction parameters in:")
    logger.info(f"  {PARAMS_FILE_PATH}")

# Removed the old collect_calibration_data function

def get_current_task_target() -> np.ndarray:
    """
    Generates a random target position within the robot's workspace,
    similar to the training environment's sampling.

    Uses spherical coordinates (radius, theta, phi) and loaded workspace limits.

    Returns:
        np.ndarray: The current target position in METERS (e.g., [x, y, z]).
    """
    # Get logger instance locally to help linter
    func_logger = logging.getLogger(__name__) 

    # Use loaded/calculated radius range
    min_radius = TARGET_SAMPLING_MIN_RADIUS
    max_radius = TARGET_SAMPLING_MAX_RADIUS

    # Sample radius and angles
    radius = min_radius + np.random.rand() * (max_radius - min_radius)
    theta = np.random.rand() * 2 * math.pi # Azimuthal angle (0 to 2pi)
    # Polar angle (restricted to upper hemisphere mostly, similar to env)
    # Adjust range if needed for real robot workspace
    phi_range = 0.6 # 0.6 * pi range
    phi_offset = 0.1 # Start 0.1*pi from vertical
    phi = (np.random.rand() * phi_range + phi_offset) * math.pi

    # Spherical to Cartesian conversion (Origin at robot base)
    x = radius * math.sin(phi) * math.sin(theta)
    y = radius * math.sin(phi) * math.cos(theta)
    z = radius * math.cos(phi)

    # Ensure target is above minimum height
    if z < TARGET_MIN_Z_HEIGHT:
        # If too low, shift it up slightly or regenerate? Let's shift.
        func_logger.debug(f"Generated target Z ({z:.3f}) below minimum {TARGET_MIN_Z_HEIGHT:.3f}. Adjusting Z.")
        z = TARGET_MIN_Z_HEIGHT
        # Recalculate radius based on new Z? Simpler to just use adjusted Z for now.

    target_pos_m = np.array([x, y, z], dtype=np.float32)
    func_logger.info(f"Generated Random Target: [{x:.3f}, {y:.3f}, {z:.3f}] (R={radius:.3f}, Phi={phi:.3f}, Theta={theta:.3f})")
    return target_pos_m


def get_current_obstacles() -> list[np.ndarray]:
    """
    **NOTE: Obstacle handling disabled.**
    Returns an empty list. If known static obstacles exist, this function
    should be updated to return their coordinates relative to the robot base.
    """
    # Known static obstacles are currently ignored.
    # Example (if implemented): obstacles_m: List[np.ndarray] = [np.array([0.5, 0.1, 0.2])]
    obstacles_m: List[np.ndarray] = [] # Return empty list
    return obstacles_m


def perform_safety_checks(robot_api: FANUCRobotAPI, current_pos_rad: np.ndarray, command_rad: np.ndarray, is_velocity_cmd: bool, skip_safety: bool, physics_client_id: int, gui_robot_id: int, gui_joint_indices: List[int], gui_ee_link_index: int) -> bool:
    """
    **PLACEHOLDER: Implement CRITICAL safety checks BEFORE sending a command.**

    Args:
        robot_api: The FANUCRobotAPI instance.
        current_pos_rad: Current joint positions (radians).
        command_rad: The intended command (position in rad or velocity in rad/s).
        is_velocity_cmd: True if the command is velocity.
        skip_safety: If True, bypass all safety checks.
        physics_client_id: PyBullet client ID for checks.
        gui_robot_id: PyBullet robot ID for checks.
        gui_joint_indices: Joint indices for the PyBullet robot.
        gui_ee_link_index: EE link index for the PyBullet robot.

    Returns:
        bool: True if the command is deemed SAFE, False otherwise.
    """
    # --- Bypass Check ---
    if skip_safety:
        # logger.warning("SAFETY CHECKS BYPASSED by flag.") # Logged in main loop entry
        return True

    safe = True
    # --- 1. Joint Limit Checks --- #
    if not is_velocity_cmd:
        # Check if joint_limits_rad is available and is a dictionary
        # Corrected attribute access based on robot_api.py
        if hasattr(robot_api, 'joint_limits_rad') and isinstance(robot_api.joint_limits_rad, dict):
            for i, pos_cmd in enumerate(command_rad):
                 # Check if the index i exists as a key in the dictionary
                 if i in robot_api.joint_limits_rad:
                     low, high = robot_api.joint_limits_rad[i] # Use robot_api.joint_limits_rad
                     # Use radians for comparison
                     if not (low - 0.01 <= pos_cmd <= high + 0.01): # Allow small tolerance
                          logger.critical(f"SAFETY VIOLATION: J{i+1} position command {pos_cmd:.3f} rad exceeds limits [{low:.3f}, {high:.3f}] rad")
                          safe = False
                 else:
                      # Log if a joint index expected is missing from the limits dict
                      logger.warning(f"Safety Check: Joint index {i} not found in robot_api.joint_limits_rad dictionary.")
        else:
            logger.warning("Safety Check: Robot joint_limits_rad attribute not found or not a dict. Skipping position limit check.")
            # safe = False # Consider making this unsafe if limits are critical
    else:
        # --- 2. Velocity Limit Checks --- #
        for i, vel_cmd in enumerate(command_rad):
             if i < len(MAX_JOINT_VEL_RAD_S):
                 limit = MAX_JOINT_VEL_RAD_S[i]
                 if abs(vel_cmd) > limit:
                     logger.critical(f"SAFETY VIOLATION: J{i+1} velocity command {vel_cmd:.3f} rad/s exceeds limit +/-{limit:.3f}")
                     safe = False

    # --- 3. Workspace Boundary Checks (Requires FK) --- #
    # This check only applies to position commands
    if not is_velocity_cmd:
        # Use modified FK function
        commanded_ee_pos_m = calculate_forward_kinematics(command_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
        if commanded_ee_pos_m is not None:
             if not np.all((WORKSPACE_LIMITS_XYZ_MIN <= commanded_ee_pos_m) & (commanded_ee_pos_m <= WORKSPACE_LIMITS_XYZ_MAX)):
                  logger.critical(f"SAFETY VIOLATION: Commanded EE position {commanded_ee_pos_m} outside workspace limits.")
                  logger.critical(f"  Limits MIN: {WORKSPACE_LIMITS_XYZ_MIN}, MAX: {WORKSPACE_LIMITS_XYZ_MAX}")
                  safe = False
        else:
             logger.warning("Safety Check: Could not calculate FK for command. Skipping workspace boundary check.")
             # safe = False

    # --- 4. Excessive Displacement/Speed Check --- #
    # This check only applies to position commands
    if not is_velocity_cmd:
        # Calculate EE pos for current joint angles using modified FK
        current_ee_pos_m = calculate_forward_kinematics(current_pos_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
        # Commanded EE pos calculated above (if successful)
        commanded_ee_pos_m_for_disp = commanded_ee_pos_m # Reuse from workspace check

        if current_ee_pos_m is not None and commanded_ee_pos_m_for_disp is not None:
            ee_displacement = np.linalg.norm(commanded_ee_pos_m_for_disp - current_ee_pos_m)
            if ee_displacement > MAX_STEP_DISPLACEMENT_M:
                logger.critical(f"SAFETY VIOLATION: Commanded position results in large step displacement ({ee_displacement:.4f}m). Limit: {MAX_STEP_DISPLACEMENT_M:.3f}m.")
                safe = False
        else:
             logger.warning("Safety Check: Could not calculate FK for current or commanded pose. Skipping displacement check.")
             # safe = False

    # --- 5. Collision Checks (Requires Collision Model) --- #
    # 5.a Self-Collision Check (using GUI simulation instance)
    if not is_velocity_cmd:
        if physics_client_id >= 0 and gui_robot_id != -1:
            try:
                # Set robot pose in GUI sim (already done by FK calculation above)
                # for i, joint_index in enumerate(gui_joint_indices):
                #     p.resetJointState(gui_robot_id, joint_index,
                #                       targetValue=command_rad[i],
                #                       targetVelocity=0,
                #                       physicsClientId=physics_client_id)

                # Check for self-collisions
                p.performCollisionDetection(physicsClientId=physics_client_id)
                self_contacts = p.getContactPoints(bodyA=gui_robot_id, bodyB=gui_robot_id, physicsClientId=physics_client_id)

                # Filter contacts: Ignore contacts between adjacent links
                num_joints_fk = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id)
                adjacent_link_pairs = set()
                for i in range(num_joints_fk):
                    joint_info = p.getJointInfo(gui_robot_id, i, physicsClientId=physics_client_id)
                    parent_link_index = joint_info[16]
                    child_link_index = i # In PyBullet, link index often matches joint index
                    # Store pairs in both orders for easy checking
                    adjacent_link_pairs.add(tuple(sorted((parent_link_index, child_link_index))))

                dangerous_contacts = []
                for contact in self_contacts:
                    link_a = contact[3]
                    link_b = contact[4]
                    # Ignore contact with self and adjacent links
                    if link_a != link_b and tuple(sorted((link_a, link_b))) not in adjacent_link_pairs:
                        dangerous_contacts.append((link_a, link_b))

                if dangerous_contacts:
                    logger.critical(f"SAFETY VIOLATION: Predicted self-collision for command!")
                    logger.debug(f"  Detected non-adjacent contact pairs: {dangerous_contacts}")
                    safe = False

            except Exception as e:
                 logger.warning(f"Safety Check: Error during self-collision check: {e}")
                 # safe = False
        else:
             logger.warning("Safety Check: GUI client/robot not available. Skipping self-collision check.")
             # safe = False # Consider making this unsafe

    # 5.b Environment Collision Check (Omitted)
    # Assuming a closed-off environment clear of unknown/unmodeled static obstacles.
    # Self-collision checks are still performed above.
    # If static obstacles ARE present, they should be modeled and checked here.
    # logger.warning("Safety Check: Environment collision check against known static obstacles is NOT implemented.")

    if not safe:
        logger.critical("Command deemed UNSAFE. Skipping command.")

    return safe

# ==========================================================================
# --- Main Deployment Logic ---
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description='Deploy trained model to a real FANUC robot.')
    parser.add_argument('--ip', type=str, default=DEFAULT_ROBOT_IP,
                        help=f'IP address of the FANUC robot controller (default: {DEFAULT_ROBOT_IP}).')
    parser.add_argument('--port', type=int, default=DEFAULT_ROBOT_PORT,
                        help=f'Port number for the robot communication interface (default: {DEFAULT_ROBOT_PORT}).')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to the trained PPO model .zip file (default: {DEFAULT_MODEL_PATH}).')
    parser.add_argument('--loop_rate_hz', type=float, default=DEFAULT_LOOP_RATE_HZ,
                        help=f'Frequency (Hz) of the main control loop (default: {DEFAULT_LOOP_RATE_HZ}).')
    parser.add_argument('--control_mode', type=str, default=DEFAULT_CONTROL_MODE, choices=['position', 'velocity'],
                        help=f'Control mode: whether model outputs target positions or velocities (default: {DEFAULT_CONTROL_MODE}).')
    parser.add_argument('--pos_speed', type=int, default=DEFAULT_POS_SPEED_PERCENT,
                         help=f'Speed percentage (1-100) for position control moves (default: {DEFAULT_POS_SPEED_PERCENT}).')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run the calibration observation routine before the main loop.') # Updated help text
    parser.add_argument('--skip_safety', action='store_true',
                        help='DANGEROUS: Skip the safety checks (USE WITH EXTREME CAUTION). Requires explicit flag.')

    args = parser.parse_args()

    logger.info("--- Starting Real Robot Deployment Script with GUI Visualisation ---") # Updated title
    logger.info(f"Arguments: {vars(args)}")

    if args.skip_safety:
        logger.critical("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical("!!! SAFETY CHECKS ARE DISABLED via --skip_safety !!!")
        logger.critical("!!!      USE WITH EXTREME CAUTION - RISK       !!!")
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        time.sleep(5.0) # Give user time to see the warning

    # --- Initialize PyBullet GUI --- #
    physics_client_id = -1
    gui_robot_id = -1
    gui_joint_indices = []
    gui_ee_link_index = -1
    plane_id = -1
    target_marker_id = -1
    obstacle_marker_ids = []

    try:
        logger.info("Initializing PyBullet GUI for visualisation...")
        physics_client_id = p.connect(p.GUI)
        if physics_client_id < 0:
            raise ConnectionError("Failed to connect to PyBullet GUI.")

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physics_client_id)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-15,
                                   cameraTargetPosition=[0,0,0.4], physicsClientId=physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)

        # Load plane
        plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client_id)

        # Load robot URDF (ensure paths are correct relative to PROJECT_ROOT)
        fanuc_dir = os.path.join(PROJECT_ROOT, "Fanuc")
        urdf_path = os.path.join(fanuc_dir, "urdf", "Fanuc.urdf")
        if not os.path.exists(urdf_path):
             raise FileNotFoundError(f"GUI URDF not found at {urdf_path}")
        p.setAdditionalSearchPath(fanuc_dir) # For meshes

        gui_robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=physics_client_id)

        # Find joint indices and EE link index for GUI robot
        num_joints_total = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id)
        for i in range(num_joints_total):
            info = p.getJointInfo(gui_robot_id, i, physicsClientId=physics_client_id)
            joint_type = info[2]
            # Assuming first 5 revolute joints are the controlled ones
            if joint_type == p.JOINT_REVOLUTE and len(gui_joint_indices) < 5:
                gui_joint_indices.append(i)
            link_name = info[12].decode('UTF-8')
            if link_name == 'Part6': # Make sure 'Part6' is your EE link name
                 gui_ee_link_index = i

        if gui_ee_link_index == -1:
            logger.warning("GUI Robot: Could not find EE link 'Part6', using last joint index.")
            gui_ee_link_index = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id) - 1
        if len(gui_joint_indices) != 5:
             raise ValueError(f"GUI Robot: Found {len(gui_joint_indices)} controllable joints, expected 5.")

        logger.info("PyBullet GUI initialized successfully.")

        # --- Initialize Modules --- #
        logger.info("Initialising Robot API...")
        robot_api = FANUCRobotAPI(ip=args.ip, port=args.port) # Renamed variable

        logger.info("Initialising Transfer Learning Module...")
        if not os.path.exists(args.model):
            logger.error(f"Model file not found at: {args.model}")
            raise FileNotFoundError(args.model)
        # Pass velocity limits for action clipping
        transfer = RobotTransfer(model_path=args.model,
                                 params_file=PARAMS_FILE_PATH,
                                 velocity_limits=MAX_JOINT_VEL_RAD_S)

        if transfer.model is None:
             logger.error("Failed to load the RL model from the transfer module. Exiting.")
             raise ValueError("RL Model load failed")

        # --- Connect to Robot --- #
        if not robot_api.connect():
            logger.error("Failed to connect to the robot. Please check IP, port, network, and robot state.")
            raise ConnectionError("Robot connection failed")

        # --- Calibration Observation Phase --- #
        if args.calibrate:
            logger.info("--- Running Calibration Observation Routine (before main loop) --- ")
            run_calibration_observation_routine(robot_api)
            logger.info("--- Calibration Observation Finished --- ")
            logger.info("Please ensure you have created/updated config/transfer_params.json before proceeding.")
            input("Press Enter to continue with the main control loop (after updating JSON)...")

        # --- Check transfer parameters --- #
        transfer.load_calibration_params(PARAMS_FILE_PATH)
        if not transfer.is_calibrated:
             logger.warning("Transfer module using default/uncalibrated parameters.")

        # --- Main Control Loop Setup --- #
        logger.info("Starting main control loop. Press Ctrl+C to stop.")
        loop_delay = 1.0 / args.loop_rate_hz
        is_velocity_control = (args.control_mode == 'velocity')
        target_reached_threshold = 0.03 # Meters
        current_target_m = get_current_task_target()

        # Variables for velocity estimation
        prev_joint_pos_rad = None
        prev_timestamp = None

        # --- Velocity Filtering Setup --- #
        VELOCITY_FILTER_WINDOW = 4 # Number of samples for moving average
        # Initialize deque for each joint, assuming 5 controlled joints
        # Get num_controlled_joints from robot_api instance if possible after init
        # For now, assume 5 based on constants
        num_joints_assumed = 5
        velocity_history = [collections.deque(maxlen=VELOCITY_FILTER_WINDOW) for _ in range(num_joints_assumed)]

        # Get initial state
        initial_obs = robot_api.get_robot_state_observation()
        if initial_obs is None:
             logger.error("Failed to get initial robot observation. Aborting loop.")
             raise ConnectionError("Cannot get initial state")
        # Use the correct number of joints from the robot_api instance
        num_controlled_joints = robot_api.num_controlled_joints
        if num_joints_assumed != num_controlled_joints:
             logger.warning(f"Initial joint number assumption ({num_joints_assumed}) != robot_api value ({num_controlled_joints}). Re-initializing velocity filter.")
             velocity_history = [collections.deque(maxlen=VELOCITY_FILTER_WINDOW) for _ in range(num_controlled_joints)]

        prev_joint_pos_rad = initial_obs[0:num_controlled_joints]
        prev_timestamp = time.monotonic()
        current_joint_pos_rad = prev_joint_pos_rad # Use initial for first safety check

        # Pre-fill velocity history with zeros to avoid issues on first loops
        initial_zero_vel = np.zeros(num_controlled_joints)
        for i in range(VELOCITY_FILTER_WINDOW):
            for joint_idx in range(num_controlled_joints):
                velocity_history[joint_idx].append(initial_zero_vel[joint_idx])

        while True:
            loop_start_time = time.monotonic()
            current_timestamp = loop_start_time # Use loop start time for consistency

            # --- Connection Check --- #
            if not robot_api.check_connection():
                logger.error("Robot connection lost. Attempting to reconnect...")
                time.sleep(2.0)
                if not robot_api.connect():
                     logger.error("Reconnect failed. Exiting.")
                     break
                else:
                     logger.info("Reconnected successfully.")
                     initial_obs = robot_api.get_robot_state_observation() # Get fresh state
                     if initial_obs is None: raise ConnectionError("Cannot get state after reconnect")
                     prev_joint_pos_rad = initial_obs[0:robot_api.num_controlled_joints]
                     prev_timestamp = time.monotonic()
                     current_joint_pos_rad = prev_joint_pos_rad
                     # Reset velocity filter on reconnect
                     for joint_hist in velocity_history:
                         joint_hist.clear()
                         for _ in range(VELOCITY_FILTER_WINDOW):
                             joint_hist.append(0.0) # Fill with zeros
                     continue # Skip rest of loop iteration to get fresh data next time

            # 1. Get Real Robot Joint Positions (Degrees)
            current_joint_pos_deg_all = robot_api.get_joint_positions()
            if current_joint_pos_deg_all is None:
                logger.warning("Failed to get current joint positions. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue
            current_joint_pos_rad = np.deg2rad(current_joint_pos_deg_all[:robot_api.num_controlled_joints])

            # --- Update GUI Visualisation --- #
            for i, joint_index in enumerate(gui_joint_indices):
                p.resetJointState(gui_robot_id, joint_index,
                                  targetValue=current_joint_pos_rad[i],
                                  targetVelocity=0,
                                  physicsClientId=physics_client_id)

            # --- Calculate Velocity --- #
            raw_velocity_rad_s = np.zeros_like(current_joint_pos_rad) # Default to zero
            if prev_joint_pos_rad is not None and prev_timestamp is not None:
                delta_time = current_timestamp - prev_timestamp
                if delta_time > 1e-6: # Avoid division by zero
                    delta_pos = current_joint_pos_rad - prev_joint_pos_rad
                    # Simple Euler estimate, could use filtering
                    raw_velocity_rad_s = delta_pos / delta_time
                else:
                    # If no previous data, raw velocity is zero
                    pass # raw_velocity_rad_s already initialized to zero

            # --- Filter Velocity --- #
            filtered_velocity_rad_s = np.zeros_like(current_joint_pos_rad)
            for i in range(len(current_joint_pos_rad)):
                velocity_history[i].append(raw_velocity_rad_s[i])
                # Calculate moving average
                filtered_velocity_rad_s[i] = np.mean(velocity_history[i])

            # --- Calculate EE Position using FK (from GUI robot state) --- #
            # Note: FK function now implicitly uses the state set by resetJointState above
            current_ee_pos_m = calculate_forward_kinematics(current_joint_pos_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
            if current_ee_pos_m is None:
                 logger.warning("Failed to calculate FK for current pose. Using origin as fallback.")
                 current_ee_pos_m = np.zeros(3)

            # --- Update Target/Obstacle Markers --- #
            obstacles_m = get_current_obstacles()
            # Remove old markers
            if target_marker_id >= 0:
                p.removeBody(target_marker_id, physicsClientId=physics_client_id)
                target_marker_id = -1
            for obs_id in obstacle_marker_ids:
                 if obs_id >= 0:
                      try:
                           p.removeBody(obs_id, physicsClientId=physics_client_id)
                      except p.error:
                           pass # Ignore error if body already removed
            obstacle_marker_ids.clear()

            # Draw new target marker
            tgt_col = [0, 1, 0, 0.8] # Green
            target_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=tgt_col, physicsClientId=physics_client_id)
            target_marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape_id, basePosition=current_target_m, physicsClientId=physics_client_id)

            # Draw new obstacle markers
            obs_col = [1, 0, 0, 0.7] # Red
            for obs_pos in obstacles_m:
                 obs_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=obs_col, physicsClientId=physics_client_id)
                 obs_marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=obs_shape_id, basePosition=obs_pos, physicsClientId=physics_client_id)
                 obstacle_marker_ids.append(obs_marker_id)

            # --- Target Reached Check --- #
            distance_to_target = np.linalg.norm(current_target_m - current_ee_pos_m)
            if distance_to_target < target_reached_threshold:
                logger.info(f"Target {current_target_m} reached (Dist: {distance_to_target:.4f}m). Generating new target.")
                current_target_m = get_current_task_target()
                prev_joint_pos_rad = None # Force zero velocity estimate next step

            # --- Calculate Relative Vectors --- #
            relative_target_vec = current_target_m - current_ee_pos_m
            if obstacles_m:
                nearest_dist = float('inf')
                relative_obstacle_vec = np.zeros(3)
                for obs_pos_m in obstacles_m:
                    dist = np.linalg.norm(obs_pos_m - current_ee_pos_m)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        relative_obstacle_vec = obs_pos_m - current_ee_pos_m
            else:
                relative_obstacle_vec = np.zeros(3)

            # --- Construct Full Observation Vector --- #
            norm_joint_pos = np.zeros(robot_api.num_controlled_joints)
            for i in range(robot_api.num_controlled_joints):
                 if i in robot_api.joint_limits_rad:
                     low, high = robot_api.joint_limits_rad[i]
                     joint_range = high - low
                     if joint_range > 1e-6:
                         norm_joint_pos[i] = 2 * (current_joint_pos_rad[i] - low) / joint_range - 1.0
            norm_joint_pos = np.clip(norm_joint_pos, -1.0, 1.0)

            try:
                real_obs = np.concatenate([
                    current_joint_pos_rad,       # 5 dims
                    filtered_velocity_rad_s,     # 5 dims (Now Filtered)
                    relative_target_vec,         # 3 dims
                    norm_joint_pos,              # 5 dims
                    relative_obstacle_vec        # 3 dims
                ]).astype(np.float32)
            except ValueError as e:
                logger.error(f"Error concatenating observation: {e}. Skipping cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # --- Predict Action --- #
            action_for_robot_rad = transfer.predict(real_obs)
            if action_for_robot_rad is None:
                logger.warning("Failed to predict action. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # --- Safety Checks --- #
            is_safe = True
            if not args.skip_safety:
                # Pass GUI client/robot IDs for checks
                is_safe = perform_safety_checks(robot_api, current_joint_pos_rad, action_for_robot_rad, is_velocity_control, args.skip_safety, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
            else:
                is_safe = True

            # --- Send Command --- #
            if is_safe:
                if is_velocity_control:
                     logger.error("Velocity control mode not supported by current API.")
                     success = False
                else:
                    action_for_robot_deg = np.rad2deg(action_for_robot_rad)
                    if robot_api.num_reported_joints == 6 and len(action_for_robot_deg) == 5:
                         command_deg = np.append(action_for_robot_deg, 0.0)
                    elif len(action_for_robot_deg) == robot_api.num_reported_joints:
                         command_deg = action_for_robot_deg
                    else:
                         logger.error(f"Action dimension mismatch for command.")
                         command_deg = None

                    if command_deg is not None:
                        success = robot_api.move_to_joint_positions(
                            positions_deg=command_deg, speed_percent=args.pos_speed)
                        # --- Add estimated wait after successful command send --- #
                        if success:
                             # WARNING: This is NOT a guarantee of move completion.
                             # It\'s a fixed pause assuming the move starts quickly.
                             # A robust solution requires robot status feedback.
                             estimated_wait = loop_delay * 0.8 # Wait 80% of loop time
                             time.sleep(estimated_wait)
                    else:
                        success = False
                if not success:
                    logger.warning("Failed to send command to robot.")

            # --- Loop Timing --- #
            loop_end_time = time.monotonic()
            elapsed_time = loop_end_time - loop_start_time
            wait_time = loop_delay - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                logger.warning(f"Control loop duration ({elapsed_time:.4f}s) exceeded rate ({loop_delay:.4f}s).")

    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Stopping...")
    except ConnectionError as e:
         logger.error(f"Connection error: {e}. Stopping...")
    except Exception as e:
        logger.error(f"Unexpected error in control loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        # --- Cleanup --- #
        logger.info("Initiating shutdown...")
        if 'robot_api' in locals() and robot_api.connected:
            logger.info("Attempting to stop robot motion...")
            robot_api.stop_motion()
            time.sleep(0.5) # Short pause before disconnecting
            logger.info("Disconnecting from robot...")
            robot_api.disconnect()
        # cleanup_fk_calculator() # Removed - using GUI client
        if physics_client_id >= 0:
             logger.info("Disconnecting PyBullet GUI...")
             try:
                  p.disconnect(physicsClientId=physics_client_id)
             except Exception as e:
                  logger.error(f"Error disconnecting PyBullet GUI: {e}")
        logger.info("Deployment script finished.")

if __name__ == "__main__":
    main() 