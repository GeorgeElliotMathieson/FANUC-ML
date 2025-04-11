# scripts/deploy_real.py
import time
import argparse
import logging
import numpy as np
import os
import sys
import traceback
import json
import math
import collections
from typing import List, Optional, Any, Tuple
import pybullet as p # type: ignore
import pybullet_data # type: ignore

from config import robot_config

# Add src directory to path
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    from src.deployment.robot_api import FANUCRobotAPI
    from src.deployment.transfer_learning import RobotTransfer
except ImportError as e:
     print(f"Error importing src modules: {e}")
     print("Ensure deploy_real.py is in the 'scripts' directory and src is accessible.")
     sys.exit(1)

# Robot IP
DEFAULT_ROBOT_IP = '192.168.1.10'
# Robot Port
DEFAULT_ROBOT_PORT = 6000

# Model path
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'output', 'ppo_logs', 'ppo_fanuc_model.zip')
# Loop rate (Hz)
DEFAULT_LOOP_RATE_HZ = 10.0
# Control mode
DEFAULT_CONTROL_MODE = 'position'
# Position speed (%)
DEFAULT_POS_SPEED_PERCENT = 20

# Min workspace limits (m)
WORKSPACE_LIMITS_XYZ_MIN = np.array([0.1, -0.5, 0.05])
# Max workspace limits (m)
WORKSPACE_LIMITS_XYZ_MAX = np.array([0.8, 0.5, 0.8])

# Max EE step displacement (m)
MAX_STEP_DISPLACEMENT_M = 0.1

# Max joint velocity (rad/s)
MAX_JOINT_VEL_RAD_S = robot_config.VELOCITY_LIMITS_RAD_S

# Calibration poses (rad)
CALIBRATION_POSES = {
    "home": np.zeros(5, dtype=np.float32),
    "pose1": np.deg2rad(np.array([30, 0, 30, 0, 0], dtype=np.float32)),
    "pose2": np.deg2rad(np.array([-30, 45, 0, 0, 30], dtype=np.float32)),
}
# Calibration speed (%)
CALIBRATION_SPEED = 15
# Calibration pause (s)
CALIBRATION_PAUSE_S = 8

# Transfer params file path
PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'transfer_params.json')

# Workspace config file path
WORKSPACE_CONFIG_FILENAME = os.path.join(PROJECT_ROOT, "workspace_config.json")

# Workspace safety factor
WORKSPACE_SAFETY_FACTOR = 0.95
# Min target Z height (m)
TARGET_MIN_Z_HEIGHT = 0.05

# Target sampling defaults
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

if REACH_AT_MIDPOINT > MIN_BASE_RADIUS:
    TARGET_SAMPLING_MAX_RADIUS = REACH_AT_MIDPOINT * WORKSPACE_SAFETY_FACTOR
    logging.info(f"Using Midpoint Reach for Target Sampling Max Radius: {TARGET_SAMPLING_MAX_RADIUS:.3f}")
else:
    TARGET_SAMPLING_MAX_RADIUS = MAX_REACH_ABS * WORKSPACE_SAFETY_FACTOR
    logging.warning(f"Using Absolute Reach for Target Sampling Max Radius: {TARGET_SAMPLING_MAX_RADIUS:.3f}")
# Target sampling max radius (m)
TARGET_SAMPLING_MAX_RADIUS = max(TARGET_SAMPLING_MAX_RADIUS, MIN_BASE_RADIUS + 0.1)
# Target sampling min radius (m)
TARGET_SAMPLING_MIN_RADIUS = MIN_BASE_RADIUS * 5.0
TARGET_SAMPLING_MIN_RADIUS = min(TARGET_SAMPLING_MIN_RADIUS, TARGET_SAMPLING_MAX_RADIUS * 0.5)
logging.info(f"Target Sampling Radius Range: [{TARGET_SAMPLING_MIN_RADIUS:.3f} - {TARGET_SAMPLING_MAX_RADIUS:.3f}] meters")

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def calculate_forward_kinematics(joint_angles_rad: np.ndarray, physics_client_id: int, robot_id: int, joint_indices: List[int], ee_link_index: int) -> np.ndarray | None:
    """Calculate EE position via PyBullet FK."""
    if physics_client_id < 0 or robot_id == -1:
        logger.error("Invalid physics client or robot ID for FK calculation.")
        return None
    if len(joint_angles_rad) != len(joint_indices):
        logger.error(f"FK input dimension mismatch: Expected {len(joint_indices)}, got {len(joint_angles_rad)}")
        return None

    try:
        # Reset joint states
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_index,
                              targetValue=joint_angles_rad[i],
                              targetVelocity=0,
                              physicsClientId=physics_client_id)

        # Get link state
        link_state = p.getLinkState(robot_id, ee_link_index,
                                  computeForwardKinematics=True,
                                  physicsClientId=physics_client_id)

        ee_position_m = np.array(link_state[4], dtype=np.float32)
        return ee_position_m

    except Exception as e:
        logger.error(f"Error during FK calculation: {e}")
        logger.error(traceback.format_exc())
        return None

def run_calibration_observation_routine(robot: FANUCRobotAPI):
    """Guide user through observing robot calibration poses."""
    logger.info("Starting Calibration Observation Routine...")
    logger.info("The robot will move sequentially to several poses.")
    logger.info(f"At each pose, it will pause for {CALIBRATION_PAUSE_S} seconds.")
    logger.info("Observe the robot's posture and end-effector position carefully.")
    logger.info("Note any consistent deviations from expected simulation behavior.")
    logger.info("Use these observations to manually determine the correction parameters")
    logger.info(f"(state_mean, state_std, action_scale, action_offset) in {PARAMS_FILE_PATH}")
    logger.info("Press Ctrl+C to stop the routine early if needed.")
    time.sleep(5)

    initial_pos_deg = robot.get_joint_positions()
    if initial_pos_deg is None:
         logger.error("Failed to get initial robot position. Aborting calibration routine.")
         return
    poses_to_visit_deg = {"current": initial_pos_deg}
    for name, pose_rad in CALIBRATION_POSES.items():
        poses_to_visit_deg[name] = np.rad2deg(pose_rad)

    try:
        for name, pose_deg in poses_to_visit_deg.items():
            if robot.num_reported_joints == 6 and len(pose_deg) == 5:
                pose_cmd_deg = np.append(pose_deg, 0.0)
            elif len(pose_deg) == robot.num_reported_joints:
                 pose_cmd_deg = pose_deg
            else:
                logger.warning(f"Skipping invalid calibration pose '{name}'. Dimension mismatch (expected {robot.num_reported_joints}).")
                continue

            logger.info(f"\nMoving to calibration pose: '{name}' ({np.round(pose_cmd_deg, 1)} deg)...")
            success = robot.move_to_joint_positions(
                positions_deg=pose_cmd_deg,
                speed_percent=CALIBRATION_SPEED
            )

            if not success:
                logger.error(f"Failed to send command for pose '{name}'. Skipping pause.")
                continue

            logger.info(f"Pausing for {CALIBRATION_PAUSE_S} seconds. OBSERVE THE ROBOT at pose '{name}'.")
            time.sleep(CALIBRATION_PAUSE_S)
            logger.info(f"... observation pause finished for pose '{name}'.")

        logger.info("Returning to home position...")
        home_pose_deg = np.rad2deg(CALIBRATION_POSES["home"])
        if robot.num_reported_joints == 6:
             home_cmd_deg = np.append(home_pose_deg, 0.0)
        else:
             home_cmd_deg = home_pose_deg

        robot.move_to_joint_positions(home_cmd_deg, speed_percent=CALIBRATION_SPEED)
        time.sleep(CALIBRATION_PAUSE_S / 2)

    except KeyboardInterrupt:
        logger.warning("Calibration routine interrupted by user.")
        logger.info("Attempting to stop motion...")
        robot.stop_motion()

    except Exception as e:
         logger.error(f"An error occurred during the calibration routine: {e}")
         logger.error(traceback.format_exc())
         logger.info("Attempting to stop motion due to error...")
         robot.stop_motion()

    logger.info("Starting Calibration Observation Routine Finished...")
    logger.info("Remember to manually create/update the correction parameters in:")
    logger.info(f"  {PARAMS_FILE_PATH}")

def get_current_task_target() -> np.ndarray:
    """Generate random target position (m)."""
    func_logger = logging.getLogger(__name__)

    min_radius = TARGET_SAMPLING_MIN_RADIUS
    max_radius = TARGET_SAMPLING_MAX_RADIUS

    radius = min_radius + np.random.rand() * (max_radius - min_radius)
    theta = np.random.rand() * 2 * math.pi
    phi_range = 0.6
    phi_offset = 0.1
    phi = (np.random.rand() * phi_range + phi_offset) * math.pi

    x = radius * math.sin(phi) * math.sin(theta)
    y = radius * math.sin(phi) * math.cos(theta)
    z = radius * math.cos(phi)

    if z < TARGET_MIN_Z_HEIGHT:
        func_logger.debug(f"Generated target Z ({z:.3f}) below minimum {TARGET_MIN_Z_HEIGHT:.3f}. Adjusting Z.")
        z = TARGET_MIN_Z_HEIGHT

    target_pos_m = np.array([x, y, z], dtype=np.float32)
    func_logger.info(f"Generated Random Target: [{x:.3f}, {y:.3f}, {z:.3f}] (R={radius:.3f}, Phi={phi:.3f}, Theta={theta:.3f})")
    return target_pos_m

def get_current_obstacles() -> list[np.ndarray]:
    """Get obstacle coordinates (m) (currently empty)."""
    obstacles_m: List[np.ndarray] = []
    return obstacles_m

def perform_safety_checks(robot_api: FANUCRobotAPI, current_pos_rad: np.ndarray, command_rad: np.ndarray, is_velocity_cmd: bool, skip_safety: bool, physics_client_id: int, gui_robot_id: int, gui_joint_indices: List[int], gui_ee_link_index: int) -> bool:
    """Perform safety checks before sending command."""
    if skip_safety:
        return True

    safe = True
    # Position limit check
    if not is_velocity_cmd:
        if hasattr(robot_api, 'joint_limits_rad') and isinstance(robot_api.joint_limits_rad, dict):
            for i, pos_cmd in enumerate(command_rad):
                 if i in robot_api.joint_limits_rad:
                     low, high = robot_api.joint_limits_rad[i]
                     if not (low - 0.01 <= pos_cmd <= high + 0.01):
                          logger.critical(f"SAFETY VIOLATION: J{i+1} position command {pos_cmd:.3f} rad exceeds limits ({low:.3f}, {high:.3f}) rad")
                          safe = False
                 else:
                      logger.warning(f"Safety Check: Joint index {i} not found in robot_api.joint_limits_rad dictionary.")
        else:
            logger.warning("Safety Check: Robot joint_limits_rad attribute not found or not a dict. Skipping position limit check.")
    # Velocity limit check
    else:
        for i, vel_cmd in enumerate(command_rad):
             if i < len(MAX_JOINT_VEL_RAD_S):
                 limit = MAX_JOINT_VEL_RAD_S[i]
                 if abs(vel_cmd) > limit:
                     logger.critical(f"SAFETY VIOLATION: J{i+1} velocity command {vel_cmd:.3f} rad/s exceeds limit +/-{limit:.3f}")
                     safe = False

    # Workspace limit check
    if not is_velocity_cmd:
        commanded_ee_pos_m = calculate_forward_kinematics(command_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
        if commanded_ee_pos_m is not None:
             if not np.all((WORKSPACE_LIMITS_XYZ_MIN <= commanded_ee_pos_m) & (commanded_ee_pos_m <= WORKSPACE_LIMITS_XYZ_MAX)):
                  logger.critical(f"SAFETY VIOLATION: Commanded EE position {commanded_ee_pos_m} outside workspace limits.")
                  logger.critical(f"  Limits MIN: {WORKSPACE_LIMITS_XYZ_MIN}, MAX: {WORKSPACE_LIMITS_XYZ_MAX}")
                  safe = False
        else:
             logger.warning("Safety Check: Could not calculate FK for command. Skipping workspace boundary check.")

    # Displacement check
    if not is_velocity_cmd:
        current_ee_pos_m = calculate_forward_kinematics(current_pos_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
        commanded_ee_pos_m_for_disp = commanded_ee_pos_m

        if current_ee_pos_m is not None and commanded_ee_pos_m_for_disp is not None:
            ee_displacement = np.linalg.norm(commanded_ee_pos_m_for_disp - current_ee_pos_m)
            if ee_displacement > MAX_STEP_DISPLACEMENT_M:
                logger.critical(f"SAFETY VIOLATION: Commanded position results in large step displacement ({ee_displacement:.4f}m). Limit: {MAX_STEP_DISPLACEMENT_M:.3f}m.")
                safe = False
        else:
             logger.warning("Safety Check: Could not calculate FK for current or commanded pose. Skipping displacement check.")

    # Self-collision check
    if not is_velocity_cmd:
        if physics_client_id >= 0 and gui_robot_id != -1:
            try:
                p.performCollisionDetection(physicsClientId=physics_client_id)
                self_contacts = p.getContactPoints(bodyA=gui_robot_id, bodyB=gui_robot_id, physicsClientId=physics_client_id)

                num_joints_fk = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id)
                adjacent_link_pairs = set()
                for i in range(num_joints_fk):
                    joint_info = p.getJointInfo(gui_robot_id, i, physicsClientId=physics_client_id)
                    parent_link_index = joint_info[16]
                    child_link_index = i
                    adjacent_link_pairs.add(tuple(sorted((parent_link_index, child_link_index))))

                dangerous_contacts = []
                for contact in self_contacts:
                    link_a = contact[3]
                    link_b = contact[4]
                    if link_a != link_b and tuple(sorted((link_a, link_b))) not in adjacent_link_pairs:
                        dangerous_contacts.append((link_a, link_b))

                if dangerous_contacts:
                    logger.critical(f"SAFETY VIOLATION: Predicted self-collision for command!")
                    logger.debug(f"  Detected non-adjacent contact pairs: {dangerous_contacts}")
                    safe = False

            except Exception as e:
                 logger.warning(f"Safety Check: Error during self-collision check: {e}")
        else:
             logger.warning("Safety Check: GUI client/robot not available. Skipping self-collision check.")

    if not safe:
        logger.critical("Command deemed UNSAFE. Skipping command.")

    return safe

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
                        help='Run the calibration observation routine before the main loop.')
    parser.add_argument('--skip_safety', action='store_true',
                        help='DANGEROUS: Skip the safety checks (USE WITH EXTREME CAUTION). Requires explicit flag.')

    args = parser.parse_args()

    logger.info("Starting Real Robot Deployment Script with GUI Visualisation...")
    logger.info(f"Arguments: {vars(args)}")

    if args.skip_safety:
        logger.critical("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical("!!! SAFETY CHECKS ARE DISABLED via --skip_safety !!!")
        logger.critical("!!!      USE WITH EXTREME CAUTION - RISK       !!!")
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        time.sleep(5.0)

    # PyBullet GUI variables
    physics_client_id = -1
    gui_robot_id = -1
    gui_joint_indices = []
    plane_id = -1
    target_marker_id = -1
    obstacle_marker_ids = []

    try:
        logger.info("Initialising PyBullet GUI for visualisation...")
        physics_client_id = p.connect(p.GUI)
        if physics_client_id < 0:
            raise ConnectionError("Failed to connect to PyBullet GUI.")

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physics_client_id)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-15,
                                   cameraTargetPosition=[0,0,0.4], physicsClientId=physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)

        plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client_id)

        fanuc_dir = os.path.join(PROJECT_ROOT, "Fanuc")
        urdf_path = os.path.join(fanuc_dir, "urdf", "Fanuc.urdf")
        if not os.path.exists(urdf_path):
             raise FileNotFoundError(f"GUI URDF not found at {urdf_path}")
        p.setAdditionalSearchPath(fanuc_dir)

        gui_robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=physics_client_id)

        # Get GUI joint/link indices
        num_joints_total = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id)
        for i in range(num_joints_total):
            info = p.getJointInfo(gui_robot_id, i, physicsClientId=physics_client_id)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE and len(gui_joint_indices) < 5:
                gui_joint_indices.append(i)
            link_name = info[12].decode('UTF-8')
            if link_name == 'Part6':
                 gui_ee_link_index = i

        if gui_ee_link_index == -1:
            logger.warning("GUI Robot: Could not find EE link 'Part6', using last joint index.")
            gui_ee_link_index = p.getNumJoints(gui_robot_id, physicsClientId=physics_client_id) - 1
        if len(gui_joint_indices) != 5:
             raise ValueError(f"GUI Robot: Found {len(gui_joint_indices)} controllable joints, expected 5.")

        logger.info("PyBullet GUI initialised successfully.")

        logger.info("Initialising Robot API...")
        robot_api = FANUCRobotAPI(ip=args.ip, port=args.port)

        logger.info("Initialising Transfer Learning Module...")
        if not os.path.exists(args.model):
            logger.error(f"Model file not found at: {args.model}")
            raise FileNotFoundError(args.model)
        transfer = RobotTransfer(model_path=args.model,
                                 params_file=PARAMS_FILE_PATH,
                                 velocity_limits=MAX_JOINT_VEL_RAD_S)

        if transfer.model is None:
             logger.error("Failed to load the RL model from the transfer module. Exiting.")
             raise ValueError("RL Model load failed")

        if not robot_api.connect():
            logger.error("Failed to connect to the robot. Please check IP, port, network, and robot state.")
            raise ConnectionError("Robot connection failed")

        if args.calibrate:
            logger.info("Running Calibration Observation Routine (before main loop)...")
            run_calibration_observation_routine(robot_api)
            logger.info("Calibration Observation Finished.")
            logger.info("Ensure config/transfer_params.json is updated.")
            input("Press Enter to continue (after updating JSON)...")

        # Load transfer parameters
        transfer.load_calibration_params(PARAMS_FILE_PATH)
        if not transfer.is_calibrated:
             logger.warning("Transfer module using default/uncalibrated parameters.")

        # Main control loop setup
        logger.info("Starting main control loop. Press Ctrl+C to stop.")
        loop_delay = 1.0 / args.loop_rate_hz
        is_velocity_control = (args.control_mode == 'velocity')
        target_reached_threshold = 0.03
        current_target_m = get_current_task_target()

        prev_joint_pos_rad = None
        prev_timestamp = None

        VELOCITY_FILTER_WINDOW = 4
        num_joints_assumed = 5
        velocity_history = [collections.deque(maxlen=VELOCITY_FILTER_WINDOW) for _ in range(num_joints_assumed)]

        initial_obs = robot_api.get_robot_state_observation()
        if initial_obs is None:
             logger.error("Failed to get initial robot observation. Aborting loop.")
             raise ConnectionError("Cannot get initial state")
        # Use actual number of joints from API
        num_controlled_joints = robot_api.num_controlled_joints
        if num_joints_assumed != num_controlled_joints:
             logger.warning(f"Initial joint number assumption ({num_joints_assumed}) != robot_api value ({num_controlled_joints}). Re-initialising velocity filter.")
             velocity_history = [collections.deque(maxlen=VELOCITY_FILTER_WINDOW) for _ in range(num_controlled_joints)]

        prev_joint_pos_rad = initial_obs[0:num_controlled_joints]
        prev_timestamp = time.monotonic()
        current_joint_pos_rad = prev_joint_pos_rad

        # Pre-fill velocity history
        initial_zero_vel = np.zeros(num_controlled_joints)
        for i in range(VELOCITY_FILTER_WINDOW):
            for joint_idx in range(num_controlled_joints):
                velocity_history[joint_idx].append(initial_zero_vel[joint_idx])

        while True:
            loop_start_time = time.monotonic()
            current_timestamp = loop_start_time

            # Check connection
            if not robot_api.check_connection():
                logger.error("Robot connection lost. Attempting to reconnect...")
                time.sleep(2.0)
                if not robot_api.connect():
                     logger.error("Reconnect failed. Exiting.")
                     break
                else:
                     logger.info("Reconnected successfully.")
                     initial_obs = robot_api.get_robot_state_observation()
                     if initial_obs is None: raise ConnectionError("Cannot get state after reconnect")
                     prev_joint_pos_rad = initial_obs[0:robot_api.num_controlled_joints]
                     prev_timestamp = time.monotonic()
                     current_joint_pos_rad = prev_joint_pos_rad
                     for joint_hist in velocity_history:
                         joint_hist.clear()
                         for _ in range(VELOCITY_FILTER_WINDOW):
                             joint_hist.append(0.0)
                     continue

            # Get joint positions (deg)
            current_joint_pos_deg_all = robot_api.get_joint_positions()
            if current_joint_pos_deg_all is None:
                logger.warning("Failed to get current joint positions. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue
            current_joint_pos_rad = np.deg2rad(current_joint_pos_deg_all[:robot_api.num_controlled_joints])

            # Update GUI Visualisation
            for i, joint_index in enumerate(gui_joint_indices):
                p.resetJointState(gui_robot_id, joint_index,
                                  targetValue=current_joint_pos_rad[i],
                                  targetVelocity=0,
                                  physicsClientId=physics_client_id)

            # Calculate raw velocity (rad/s)
            raw_velocity_rad_s = np.zeros_like(current_joint_pos_rad)
            if prev_joint_pos_rad is not None and prev_timestamp is not None:
                delta_time = current_timestamp - prev_timestamp
                if delta_time > 1e-6:
                    delta_pos = current_joint_pos_rad - prev_joint_pos_rad
                    raw_velocity_rad_s = delta_pos / delta_time
                else:
                    pass

            # Filter velocity
            filtered_velocity_rad_s = np.zeros_like(current_joint_pos_rad)
            for i in range(len(current_joint_pos_rad)):
                velocity_history[i].append(raw_velocity_rad_s[i])
                filtered_velocity_rad_s[i] = np.mean(velocity_history[i])

            # Calculate EE Position (m) via FK
            current_ee_pos_m = calculate_forward_kinematics(current_joint_pos_rad, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
            if current_ee_pos_m is None:
                 logger.warning("Failed to calculate FK for current pose. Using origin as fallback.")
                 current_ee_pos_m = np.zeros(3)

            # Update Target/Obstacle Markers
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
                           pass
            obstacle_marker_ids.clear()

            # Draw new target marker
            tgt_col = [0, 1, 0, 0.8]
            target_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=tgt_col, physicsClientId=physics_client_id)
            target_marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape_id, basePosition=current_target_m, physicsClientId=physics_client_id)

            # Draw new obstacle markers
            obs_col = [1, 0, 0, 0.7]
            for obs_pos in obstacles_m:
                 obs_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=obs_col, physicsClientId=physics_client_id)
                 obs_marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=obs_shape_id, basePosition=obs_pos, physicsClientId=physics_client_id)
                 obstacle_marker_ids.append(obs_marker_id)

            # Check if target reached
            distance_to_target = np.linalg.norm(current_target_m - current_ee_pos_m)
            if distance_to_target < target_reached_threshold:
                logger.info(f"Target {current_target_m} reached (Dist: {distance_to_target:.4f}m). Generating new target.")
                current_target_m = get_current_task_target()
                prev_joint_pos_rad = None

            # Calculate relative vectors
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

            # Construct observation vector
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
                    current_joint_pos_rad,
                    filtered_velocity_rad_s,
                    relative_target_vec,
                    norm_joint_pos,
                    relative_obstacle_vec
                ]).astype(np.float32)
            except ValueError as e:
                logger.error(f"Error concatenating observation: {e}. Skipping cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # Predict action
            action_for_robot_rad = transfer.predict(real_obs)
            if action_for_robot_rad is None:
                logger.warning("Failed to predict action. Skipping control cycle.")
                time.sleep(max(0, loop_delay - (time.monotonic() - loop_start_time)))
                continue

            # Perform safety checks
            is_safe = True
            if not args.skip_safety:
                is_safe = perform_safety_checks(robot_api, current_joint_pos_rad, action_for_robot_rad, is_velocity_control, args.skip_safety, physics_client_id, gui_robot_id, gui_joint_indices, gui_ee_link_index)
            else:
                is_safe = True

            # Send command if safe
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
                        if success:
                             estimated_wait = loop_delay * 0.8
                             time.sleep(estimated_wait)
                    else:
                        success = False
                if not success:
                    logger.warning("Failed to send command to robot.")

            # Maintain loop rate
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
        # Cleanup
        logger.info("Initiating shutdown...")
        if 'robot_api' in locals() and robot_api.connected:
            logger.info("Attempting to stop robot motion...")
            robot_api.stop_motion()
            time.sleep(0.5)
            logger.info("Disconnecting from robot...")
            robot_api.disconnect()
        if physics_client_id >= 0:
             logger.info("Disconnecting PyBullet GUI...")
             try:
                  p.disconnect(physicsClientId=physics_client_id)
             except Exception as e:
                  logger.error(f"Error disconnecting PyBullet GUI: {e}")
        logger.info("Deployment script finished.")

if __name__ == "__main__":
    main() 