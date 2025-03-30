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
from typing import List, Optional, Any # Import List, Optional, Any for type hinting
import pybullet as p # Import pybullet for FK calculation
import pybullet_data

# --- Project Setup ---
# Ensure the src directory is in the Python path
# Allows importing modules from src when running scripts/deploy_real.py
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    # Now we can import from src
    from robot_api import FANUCRobotAPI # Removed DEFAULT imports
    from transfer_learning import RobotTransfer
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
MAX_JOINT_VEL_RAD_S = np.array([np.pi, np.pi, np.pi, np.pi*2, np.pi*2]) # Max velocity per joint (rad/s)
MAX_STEP_DISPLACEMENT_M = 0.05 # Max allowed EE movement per step (meters)

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

# Global cache for PyBullet FK calculation to avoid reloading URDF repeatedly
_fk_client: Optional[int] = None
_fk_robot_id: int = -1
_fk_joint_indices: List[int] = []
_fk_ee_link_index: int = -1

def setup_fk_calculator():
    """Initializes a PyBullet client in DIRECT mode for FK calculations."""
    global _fk_client, _fk_robot_id, _fk_joint_indices, _fk_ee_link_index
    if _fk_client is not None:
        return # Already initialized

    try:
        logger.info("Initializing PyBullet client for Forward Kinematics...")
        _fk_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot URDF (ensure paths are correct relative to PROJECT_ROOT)
        fanuc_dir = os.path.join(PROJECT_ROOT, "Fanuc")
        urdf_path = os.path.join(fanuc_dir, "urdf", "Fanuc.urdf")
        if not os.path.exists(urdf_path):
             raise FileNotFoundError(f"FK URDF not found at {urdf_path}")
        p.setAdditionalSearchPath(fanuc_dir) # For meshes

        _fk_robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=_fk_client)

        # Find joint indices and EE link index (only needs to be done once)
        num_joints_total = p.getNumJoints(_fk_robot_id, physicsClientId=_fk_client)
        _fk_joint_indices = []
        _fk_ee_link_index = -1
        link_name_to_index = {p.getJointInfo(_fk_robot_id, i, physicsClientId=_fk_client)[12].decode('UTF-8'): i for i in range(num_joints_total)}

        for i in range(num_joints_total):
            info = p.getJointInfo(_fk_robot_id, i, physicsClientId=_fk_client)
            joint_type = info[2]
            # Assuming first 5 revolute joints are the controlled ones
            if joint_type == p.JOINT_REVOLUTE and len(_fk_joint_indices) < 5:
                _fk_joint_indices.append(i)
            link_name = info[12].decode('UTF-8')
            if link_name == 'Part6': # Make sure 'Part6' is your EE link name
                 _fk_ee_link_index = i

        if _fk_ee_link_index == -1:
            logger.warning("FK Calculator: Could not find EE link 'Part6', using last joint index.")
            _fk_ee_link_index = p.getNumJoints(_fk_robot_id, physicsClientId=_fk_client) - 1
        if len(_fk_joint_indices) != 5:
             raise ValueError(f"FK Calculator: Found {len(_fk_joint_indices)} controllable joints, expected 5.")

        logger.info("PyBullet FK calculator initialized successfully.")

    except Exception as e:
        logger.error(f"Error initializing PyBullet FK calculator: {e}")
        logger.error(traceback.format_exc())
        if _fk_client is not None:
            p.disconnect(physicsClientId=_fk_client)
        _fk_client = None # Ensure it's None on failure

def cleanup_fk_calculator():
    """Disconnects the PyBullet client used for FK."""
    global _fk_client
    if _fk_client is not None:
        logger.info("Cleaning up PyBullet FK calculator...")
        try:
            p.disconnect(physicsClientId=_fk_client)
        except Exception as e:
             logger.error(f"Error disconnecting FK PyBullet client: {e}")
        _fk_client = None

def calculate_forward_kinematics(joint_angles_rad: np.ndarray) -> np.ndarray | None:
    """
    Calculates the End Effector position using PyBullet based on joint angles.

    Args:
        joint_angles_rad: Array of 5 joint angles in radians.

    Returns:
        np.ndarray | None: The EE position [x, y, z] in meters (world frame),
                           or None if calculation fails.
    """
    if _fk_client is None or _fk_robot_id == -1 or not _fk_joint_indices or _fk_ee_link_index == -1:
        logger.error("FK calculator not initialized. Cannot calculate FK.")
        return None
    if len(joint_angles_rad) != len(_fk_joint_indices):
        logger.error(f"FK input dimension mismatch: Expected {len(_fk_joint_indices)}, got {len(joint_angles_rad)}")
        return None

    try:
        # Reset joint states in the FK simulation
        for i, joint_index in enumerate(_fk_joint_indices):
            p.resetJointState(_fk_robot_id, joint_index,
                              targetValue=joint_angles_rad[i],
                              targetVelocity=0,
                              physicsClientId=_fk_client)

        # Get the link state (world position of the link origin)
        # Use computeForwardKinematics=True just to be sure
        link_state = p.getLinkState(_fk_robot_id, _fk_ee_link_index,
                                  computeForwardKinematics=True,
                                  physicsClientId=_fk_client)

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
    **PLACEHOLDER: Implement logic to determine current obstacle positions.**
    Returns a list of obstacle positions in METERS.
    """
    # Example: One fixed obstacle for testing FK-based relative obs vector
    obstacles_m: List[np.ndarray] = [np.array([0.5, 0.1, 0.2])] 
    # logger.debug(f"Current Obstacles: {obstacles_m}")
    return obstacles_m


def perform_safety_checks(robot: FANUCRobotAPI, current_pos_rad: np.ndarray, command_rad: np.ndarray, is_velocity_cmd: bool) -> bool:
    """
    **PLACEHOLDER: Implement CRITICAL safety checks BEFORE sending a command.**

    Checks to implement:
    1.  **Joint Limit Violation:** Check if `command_rad` (if position) or the projected next position
        (if velocity) exceeds `robot.joint_limits`.
    2.  **Velocity Limit Violation:** Check if `command_rad` (if velocity) exceeds `MAX_JOINT_VEL_RAD_S`.
    3.  **Workspace Boundary Violation:** Calculate the forward kinematics for the commanded position/velocity
        and check if the end-effector position stays within `WORKSPACE_LIMITS_XYZ_MIN/MAX`.
        (Requires FK function - PyBullet or external library needed).
    4.  **Excessive Speed/Displacement:** Check if the commanded velocity or the difference between
        current and commanded position is too large (`MAX_STEP_DISPLACEMENT_M`).
    5.  **Collision Check:** (Advanced) If you have a collision model, check for self-collision or
        collision with known obstacles for the commanded move.

    Args:
        robot: The FANUCRobotAPI instance.
        current_pos_rad: Current joint positions (radians).
        command_rad: The intended command (position in rad or velocity in rad/s).
        is_velocity_cmd: True if the command is velocity.

    Returns:
        bool: True if the command is deemed SAFE, False otherwise.
    """
    safe = True
    # --- 1. Joint Limit Checks ---
    if not is_velocity_cmd:
        # Check if joint_limits is available and is a dictionary
        if hasattr(robot, 'joint_limits') and isinstance(robot.joint_limits, dict):
            for i, pos_cmd in enumerate(command_rad):
                 if i in robot.joint_limits: # Use robot.joint_limits
                     low, high = robot.joint_limits[i] # Use robot.joint_limits
                     if not (low - 0.01 <= pos_cmd <= high + 0.01): # Allow small tolerance
                          logger.critical(f"SAFETY VIOLATION: J{i+1} position command {np.rad2deg(pos_cmd):.2f} deg exceeds limits [{np.rad2deg(low):.2f}, {np.rad2deg(high):.2f}]")
                          safe = False
        else:
            logger.warning("Safety Check: Robot joint_limits attribute not found or not a dict. Skipping position limit check.")
            # Decide if this should be treated as unsafe
            # safe = False
    else:
        # --- 2. Velocity Limit Checks ---
        for i, vel_cmd in enumerate(command_rad):
             if i < len(MAX_JOINT_VEL_RAD_S):
                 limit = MAX_JOINT_VEL_RAD_S[i]
                 if abs(vel_cmd) > limit:
                     logger.critical(f"SAFETY VIOLATION: J{i+1} velocity command {vel_cmd:.3f} rad/s exceeds limit +/-{limit:.3f}")
                     safe = False

    # --- 3. Workspace Boundary Checks (Requires FK) ---
    # **TODO:** Implement FK check
    # commanded_ee_pos_m = calculate_fk(command_rad) # Your FK function here
    # if not np.all((WORKSPACE_LIMITS_XYZ_MIN <= commanded_ee_pos_m) & (commanded_ee_pos_m <= WORKSPACE_LIMITS_XYZ_MAX)):
    #      logger.critical(f"SAFETY VIOLATION: Commanded EE position {commanded_ee_pos_m} outside workspace limits.")
    #      safe = False
    logger.warning("Safety Check: Workspace boundary check is NOT implemented.")

    # --- 4. Excessive Displacement/Speed Check ---
    if not is_velocity_cmd:
        delta_joint = command_rad - current_pos_rad
        # **TODO:** Convert joint delta to EE displacement (needs Jacobian or FK diff)
        # approx_ee_displacement = estimate_ee_displacement(delta_joint)
        # if approx_ee_displacement > MAX_STEP_DISPLACEMENT_M:
        #     logger.critical(f"SAFETY VIOLATION: Commanded position results in large step displacement (~{approx_ee_displacement:.3f}m). Limit: {MAX_STEP_DISPLACEMENT_M:.3f}m.")
        #     safe = False
        logger.warning("Safety Check: Excessive step displacement check is NOT implemented.")

    # --- 5. Collision Checks (Requires Collision Model) ---
    # **TODO:** Implement collision check if feasible
    # if check_for_collisions(command_rad):
    #     logger.critical("SAFETY VIOLATION: Predicted collision for command.")
    #     safe = False
    logger.warning("Safety Check: Collision check is NOT implemented.")

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

    logger.info("--- Starting Real Robot Deployment Script ---")
    logger.info(f"Arguments: {vars(args)}")

    if args.skip_safety:
        logger.critical("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical("!!! SAFETY CHECKS ARE DISABLED via --skip_safety !!!")
        logger.critical("!!!      USE WITH EXTREME CAUTION - RISK       !!!")
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        time.sleep(5.0) # Give user time to see the warning

    # --- Initialize FK Calculator --- 
    setup_fk_calculator()
    if _fk_client is None:
        logger.critical("Failed to initialize FK calculator. Cannot proceed.")
        return

    # --- Initialize Modules ---
    logger.info("Initialising Robot API...")
    robot = FANUCRobotAPI(ip=args.ip, port=args.port)

    logger.info("Initialising Transfer Learning Module...")
    if not os.path.exists(args.model):
        logger.error(f"Model file not found at: {args.model}")
        cleanup_fk_calculator()
        return
    try:
        # Pass the expected path for calibration params to transfer module
        transfer = RobotTransfer(model_path=args.model, params_file=PARAMS_FILE_PATH)
    except Exception as e:
        logger.error(f"Error initializing Transfer Learning module: {e}")
        logger.error(traceback.format_exc())
        cleanup_fk_calculator()
        return

    if transfer.model is None:
         logger.error("Failed to load the RL model from the transfer module. Exiting.")
         cleanup_fk_calculator()
         return

    # --- Connect to Robot ---
    if not robot.connect():
        logger.error("Failed to connect to the robot. Please check IP, port, network, and robot state.")
        cleanup_fk_calculator()
        return

    # --- Calibration Observation Phase --- 
    if args.calibrate:
        logger.info("--- Running Calibration Observation Routine (before main loop) --- ")
        run_calibration_observation_routine(robot)
        logger.info("--- Calibration Observation Finished --- ")
        logger.info("Please ensure you have created/updated config/transfer_params.json before proceeding.")
        input("Press Enter to continue with the main control loop (after updating JSON)...")

    # --- Check transfer parameters --- 
    transfer.load_calibration_params(PARAMS_FILE_PATH)
    if not transfer.is_calibrated:
         logger.warning("Transfer module using default/uncalibrated parameters.")

    # --- Main Control Loop Setup --- 
    logger.info("Starting main control loop. Press Ctrl+C to stop.")
    loop_delay = 1.0 / args.loop_rate_hz
    is_velocity_control = (args.control_mode == 'velocity')
    target_reached_threshold = 0.03 # Meters
    current_target_m = get_current_task_target()

    # Variables for velocity estimation
    prev_joint_pos_rad = None
    prev_timestamp = None

    try:
        # Get initial state
        initial_obs = robot.get_robot_state_observation()
        if initial_obs is None:
             logger.error("Failed to get initial robot observation. Aborting loop.")
             raise ConnectionError("Cannot get initial state")
        prev_joint_pos_rad = initial_obs[0:robot.num_controlled_joints]
        prev_timestamp = time.monotonic()
        current_joint_pos_rad = prev_joint_pos_rad # Use initial for first safety check

        while True:
            loop_start_time = time.monotonic()
            current_timestamp = loop_start_time # Use loop start time for consistency

            # --- Connection Check --- 
            if not robot.check_connection():
                logger.error("Robot connection lost. Attempting to reconnect...")
                time.sleep(2.0)
                if not robot.connect():
                     logger.error("Reconnect failed. Exiting.")
                     break
                else:
                     logger.info("Reconnected successfully.")
                     initial_obs = robot.get_robot_state_observation() # Get fresh state
                     if initial_obs is None: raise ConnectionError("Cannot get state after reconnect")
                     prev_joint_pos_rad = initial_obs[0:robot.num_controlled_joints]
                     prev_timestamp = time.monotonic()
                     current_joint_pos_rad = prev_joint_pos_rad
                     continue # Skip rest of loop iteration to get fresh data next time

            # 1. Get Real Robot Joint Positions (Degrees)
            # We get the full observation later, but need current pos now for FK and velocity
            current_joint_pos_deg_all = robot.get_joint_positions()
            if current_joint_pos_deg_all is None:
                logger.warning("Failed to get current joint positions. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue
            current_joint_pos_rad = np.deg2rad(current_joint_pos_deg_all[:robot.num_controlled_joints])

            # --- Calculate Velocity --- 
            if prev_joint_pos_rad is not None and prev_timestamp is not None:
                delta_time = current_timestamp - prev_timestamp
                if delta_time > 1e-6: # Avoid division by zero
                    delta_pos = current_joint_pos_rad - prev_joint_pos_rad
                    # Simple Euler estimate, could use filtering
                    estimated_velocity_rad_s = delta_pos / delta_time
                else:
                    estimated_velocity_rad_s = np.zeros_like(current_joint_pos_rad)
            else:
                estimated_velocity_rad_s = np.zeros_like(current_joint_pos_rad)
            # Update history for next iteration
            prev_joint_pos_rad = current_joint_pos_rad
            prev_timestamp = current_timestamp

            # --- Calculate EE Position using FK --- 
            current_ee_pos_m = calculate_forward_kinematics(current_joint_pos_rad)
            if current_ee_pos_m is None:
                 logger.warning("Failed to calculate FK for current pose. Using origin as fallback.")
                 current_ee_pos_m = np.zeros(3)

            # --- Target Reached Check --- 
            distance_to_target = np.linalg.norm(current_target_m - current_ee_pos_m)
            if distance_to_target < target_reached_threshold:
                logger.info(f"Target {current_target_m} reached (Dist: {distance_to_target:.4f}m). Generating new target.")
                current_target_m = get_current_task_target()
                # Maybe reset velocity estimate history here?
                prev_joint_pos_rad = None # Force zero velocity estimate next step

            # --- Calculate Relative Vectors --- 
            relative_target_vec = current_target_m - current_ee_pos_m
            obstacles_m = get_current_obstacles()
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

            # --- Construct Full Observation Vector --- 
            # Use calculated/estimated values instead of placeholders
            norm_joint_pos = np.zeros(robot.num_controlled_joints)
            for i in range(robot.num_controlled_joints):
                 if i in robot.joint_limits_rad:
                     low, high = robot.joint_limits_rad[i]
                     joint_range = high - low
                     if joint_range > 1e-6:
                         norm_joint_pos[i] = 2 * (current_joint_pos_rad[i] - low) / joint_range - 1.0
            norm_joint_pos = np.clip(norm_joint_pos, -1.0, 1.0)

            try:
                real_obs = np.concatenate([
                    current_joint_pos_rad,       # 5 dims
                    estimated_velocity_rad_s,    # 5 dims
                    relative_target_vec,         # 3 dims
                    norm_joint_pos,              # 5 dims
                    relative_obstacle_vec        # 3 dims
                ]).astype(np.float32)
            except ValueError as e:
                logger.error(f"Error concatenating observation: {e}. Skipping cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # --- Predict Action --- 
            action_for_robot_rad = transfer.predict(real_obs)
            if action_for_robot_rad is None:
                logger.warning("Failed to predict action. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # --- Safety Checks --- 
            is_safe = True
            if not args.skip_safety:
                # Pass current rad position and predicted rad command (pos or vel)
                is_safe = perform_safety_checks(robot, current_joint_pos_rad, action_for_robot_rad, is_velocity_control)

            # --- Send Command --- 
            if is_safe:
                if is_velocity_control:
                     logger.error("Velocity control mode not supported by current API.")
                     success = False
                else:
                    action_for_robot_deg = np.rad2deg(action_for_robot_rad)
                    if robot.num_reported_joints == 6 and len(action_for_robot_deg) == 5:
                         command_deg = np.append(action_for_robot_deg, 0.0)
                    elif len(action_for_robot_deg) == robot.num_reported_joints:
                         command_deg = action_for_robot_deg
                    else:
                         logger.error(f"Action dimension mismatch for command.")
                         command_deg = None

                    if command_deg is not None:
                        success = robot.move_to_joint_positions(
                            positions_deg=command_deg, speed_percent=args.pos_speed)
                    else:
                        success = False
                if not success:
                    logger.warning("Failed to send command to robot.")

            # --- Loop Timing --- 
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
        # --- Cleanup --- 
        logger.info("Initiating shutdown...")
        if robot.connected:
            logger.info("Attempting to stop robot motion...")
            robot.stop_motion()
            time.sleep(0.5) # Short pause before disconnecting
            logger.info("Disconnecting from robot...")
            robot.disconnect()
        cleanup_fk_calculator() # Disconnect FK pybullet client
        logger.info("Deployment script finished.")

if __name__ == "__main__":
    main() 