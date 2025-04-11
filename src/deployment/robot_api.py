# src/robot_api.py
import socket
import struct
import time
import numpy as np
import logging
import traceback
import threading

from config import robot_config

logger = logging.getLogger(__name__)

# Default robot IP
DEFAULT_ROBOT_IP = "192.168.1.10"
# Default robot port
DEFAULT_ROBOT_PORT = 6000

# Socket timeout (s)
DEFAULT_TIMEOUT = 5.0

class FANUCRobotAPI:
    # FANUC socket communication.
    # WARNING: Assumes SM config. Verify commands/parsing.
    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT):
        self.ip: str = ip
        self.port: int = port
        self.socket: socket.socket | None = None
        self.connected: bool = False
        self.timeout: float = DEFAULT_TIMEOUT
        self.lock = threading.Lock()

        # Controlled joints count
        self.num_controlled_joints: int = robot_config.NUM_CONTROLLED_JOINTS
        # Reported joints count
        self.num_reported_joints: int = robot_config.NUM_REPORTED_JOINTS
        # Reported joint limits (deg)
        self.joint_limits_deg = robot_config.REPORTED_JOINT_LIMITS_DEG

        # Controlled joint limits (rad)
        self.joint_limits_rad = {
            i: [np.deg2rad(self.joint_limits_deg[i][0]), np.deg2rad(self.joint_limits_deg[i][1])]
            for i in range(self.num_controlled_joints)
            if i in self.joint_limits_deg
        }

        logger.info(f"Initialised FANUCRobotAPI for {self.num_controlled_joints} controlled joints.")
        logger.info(f"  Target IP: {self.ip}, Port: {self.port}")
        logger.info("  Joint Limits (deg, Approx - VERIFY):")
        for i in range(self.num_controlled_joints):
             if i in self.joint_limits_deg:
                 logger.info(f"    J{i+1}: {self.joint_limits_deg[i]}")
             else:
                 logger.warning(f"    J{i+1}: Limits not defined in robot_config.")


    def connect(self) -> bool:
        # Establish connection.
        if self.connected:
            logger.warning("Already connected.")
            return True
        try:
            logger.info(f"Attempting to connect to robot at {self.ip}:{self.port} (Timeout: {self.timeout}s)...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            logger.info(f"Successfully connected to robot controller.")
            return True
        except socket.timeout:
            logger.error(f"Connection timed out to {self.ip}:{self.port}. Check IP, port, network, and controller program/state.")
            self.socket = None
            self.connected = False
            return False
        except ConnectionRefusedError:
             logger.error(f"Connection refused by {self.ip}:{self.port}. Is the controller running the correct program/server?")
             self.socket = None
             self.connected = False
             return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to robot: {e}")
            logger.error(traceback.format_exc())
            if self.socket:
                self.socket.close()
            self.socket = None
            self.connected = False
            return False

    def disconnect(self):
        # Close connection.
        if not self.connected:
            return
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
                logger.info("Socket closed.")
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        self.socket = None
        self.connected = False
        logger.info("Disconnected from FANUC robot.")

    def _send_command(self, command: str) -> str | None:
        # Send command, get response.
        if not self.check_connection() or self.socket is None:
            logger.warning("Not connected to robot. Cannot send command.")
            return None

        with self.lock:
            try:
                if not command.endswith('\r'):
                    command += '\r'

                logger.debug(f"Sending command: '{command.strip()}'")
                self.socket.sendall(command.encode('ascii'))

                response_bytes = b''
                self.socket.settimeout(self.timeout)
                while True:
                     chunk = self.socket.recv(64)
                     if not chunk:
                          logger.error("Connection closed by robot while receiving response.")
                          self._handle_connection_error()
                          return None
                     response_bytes += chunk
                     if b'\r' in response_bytes:
                          break

                response_str = response_bytes.decode('ascii').strip()
                logger.debug(f"Received response: '{response_str}'")
                return response_str

            except socket.timeout:
                 logger.warning(f"Timeout waiting for robot response after sending '{command.strip()}'.")
                 return None
            except Exception as e:
                logger.error(f"Error sending/receiving for command '{command.strip()}': {e}")
                logger.error(traceback.format_exc())
                self._handle_connection_error()
                return None

    def _handle_connection_error(self):
        # Centralised handling for connection loss.
        logger.error("Connection error detected. Marking as disconnected.")
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.socket = None
        self.connected = False

    def get_joint_positions(self) -> np.ndarray | None:
        # Get current joint positions (deg) via RDJPOS.
        # Returns: Array (deg) of reported joints, or None.
        command = "RDJPOS"
        response = self._send_command(command)
        if response and isinstance(response, str):
            try:
                if response.startswith("JOINT:"):
                    parts = response.split(" ")[1:]
                    if len(parts) >= self.num_reported_joints:
                        joint_positions_deg = [float(p.split("=")[1]) for p in parts[:self.num_reported_joints]]
                        return np.array(joint_positions_deg)
                    else:
                        logger.error(f"Could not parse {self.num_reported_joints} joint positions from response: '{response}'")
                else:
                     logger.error(f"Robot response indicates error or unexpected format for RDJPOS: '{response}'")
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing RDJPOS response '{response}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing RDJPOS response '{response}': {e}")
        else:
            logger.warning(f"No valid response received for command '{command}'.")
        return None

    def move_to_joint_positions(self, positions_deg: list | np.ndarray, speed_percent: int = 20) -> bool:
        # Move robot via MOVJ (deg). Returns ack status.
        if positions_deg is None or len(positions_deg) != self.num_reported_joints:
            logger.error(f"Invalid joint positions array length (expected {self.num_reported_joints}): {positions_deg}")
            return False

        self._log_limit_warnings(np.array(positions_deg))

        pos_str = ' '.join([f"{pos:.3f}" for pos in positions_deg])
        command = f"MOVJ {pos_str}"

        logger.debug(f"Sending command: '{command}'")
        response = self._send_command(command)

        if response and "OK" in response:
            logger.info(f"MOVJ command acknowledged successfully.")
            return True
        else:
            logger.error(f"MOVJ command failed or NACKed: {response}")
            return False

    def stop_motion(self) -> bool:
        # Emergency stop via STOP.
        logger.warning("Sending STOP command!")
        response = self._send_command("STOP")
        if response and "OK" in response:
             logger.info("STOP command acknowledged.")
             return True
        else:
             logger.error(f"STOP command failed or NACKed: {response}")
             return False

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        # Attempts RDPOS, logs raw response. Does NOT parse.
        # Returns: None. Implement parsing based on logged data.
        command = "RDPOS"
        logger.info(f"Attempting to get End-Effector pose using command: '{command}'")
        response = self._send_command(command)

        if response is not None:
            logger.info(f"Received potential pose data (raw): '{response}'")
            logger.warning("Pose data received, but parsing is NOT implemented.")
            logger.warning("Please analyse the raw response above and update the parsing logic in get_end_effector_pose.")
        else:
            logger.warning(f"No response received for '{command}'. Robot may not support this command or is not sending pose data.")

        return None

    def get_robot_state_observation(self) -> np.ndarray | None:
        # Get RL observation (21 dims).
        # Returns: Observation vector (float32) or None.

        joint_pos_deg_all = self.get_joint_positions()
        if joint_pos_deg_all is None:
            logger.error("Failed to get joint positions for observation.")
            return None
        joint_pos_deg = joint_pos_deg_all[:self.num_controlled_joints]
        joint_pos_rad = np.deg2rad(joint_pos_deg)

        joint_vel_rad_s = np.zeros(self.num_controlled_joints)

        relative_target = np.zeros(3)

        norm_joint_pos = np.zeros(self.num_controlled_joints)
        for i in range(self.num_controlled_joints):
             if i in self.joint_limits_rad:
                 low, high = self.joint_limits_rad[i]
                 joint_range = high - low
                 if joint_range > 1e-6:
                     norm_joint_pos[i] = 2 * (joint_pos_rad[i] - low) / joint_range - 1.0
             else:
                  logger.error(f"Joint index {i} not found in radian limits for normalisation!")
        norm_joint_pos = np.clip(norm_joint_pos, -1.0, 1.0)

        relative_obstacle = np.zeros(3)

        try:
            obs = np.concatenate([
                joint_pos_rad,
                joint_vel_rad_s,
                relative_target,
                norm_joint_pos,
                relative_obstacle
            ]).astype(np.float32)
        except ValueError as e:
             logger.error(f"Error concatenating observation components: {e}")
             logger.error(f"  Shapes: pos={joint_pos_rad.shape}, vel={joint_vel_rad_s.shape}, "
                          f"tgt={relative_target.shape}, norm={norm_joint_pos.shape}, obs={relative_obstacle.shape}")
             return None

        expected_dim = 21
        if obs.shape != (expected_dim,):
            logger.error(f"Constructed observation has wrong shape: {obs.shape}, expected ({expected_dim},).")
            return None

        return obs

    def check_connection(self) -> bool:
        # Check if connection is likely active.
        if not self.connected or self.socket is None:
            return False
        try:
             err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
             if err == 0:
                 return True
             else:
                  logger.warning(f"Socket check failed with error code: {err}")
                  self._handle_connection_error()
                  return False
        except socket.timeout:
             logger.warning("Socket check timed out unexpectedly.")
             self._handle_connection_error()
             return False
        except Exception as e:
             logger.warning(f"Socket check failed with exception: {e}")
             self._handle_connection_error()
             return False

    def _log_limit_warnings(self, joint_target_deg: np.ndarray):
        # Log warnings if commanded positions exceed client-side limits.
        num_to_check = min(len(joint_target_deg), self.num_reported_joints)
        for i in range(num_to_check):
            pos = joint_target_deg[i]
            if i in self.joint_limits_deg:
                low, high = self.joint_limits_deg[i]
                if not (low <= pos <= high):
                    logger.warning(f"Commanded position {pos:.2f} for J{i+1} exceeds client-side limits [{low}, {high}]. Relying on controller limits.")
            else:
                 logger.warning(f"No client-side limits defined for commanded J{i+1}.")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    robot_ip = DEFAULT_ROBOT_IP
    robot_port = DEFAULT_ROBOT_PORT
    logger.info(f"--- Robot API Test Script ---")
    logger.info(f"Attempting connection to {robot_ip}:{robot_port}")

    robot = FANUCRobotAPI(ip=robot_ip, port=robot_port)

    if robot.connect():
        logger.info("Connection successful.")

        logger.info("Attempting to get joint positions (RDJPOS)...")
        positions = robot.get_joint_positions()
        if positions is not None:
            logger.info(f"Received Joint Positions (deg): {positions}")
        else:
            logger.warning("Failed to get joint positions.")
        time.sleep(0.5)

        logger.info("Attempting to get EE pose (RDPOS - expect raw data log and warning)...")
        robot.get_end_effector_pose()
        logger.info("EE pose check completed (check logs for raw data/warnings).")
        time.sleep(0.5)

        logger.info("Attempting to get state observation vector...")
        obs = robot.get_robot_state_observation()
        if obs is not None:
            logger.info(f"Received Observation Vector (shape {obs.shape}):")
            logger.info(f"  Pos (rad): {obs[0:robot.num_controlled_joints]}")
            logger.info(f"  Vel (rad/s): {obs[robot.num_controlled_joints:robot.num_controlled_joints*2]} (Placeholder!)")
            logger.info(f"  Rel Target: {obs[robot.num_controlled_joints*2:robot.num_controlled_joints*2+3]} (Placeholder!)")
            logger.info(f"  Norm Limits: {obs[robot.num_controlled_joints*2+3:robot.num_controlled_joints*3+3]}")
            logger.info(f"  Rel Obstacle: {obs[robot.num_controlled_joints*3+3:]} (Placeholder!)")

        else:
             logger.warning("Failed to get observation vector.")
        time.sleep(0.5)

        target_pos_deg = [0.0] * robot.num_reported_joints
        target_pos_deg[1] = 10.0
        logger.info(f"Attempting to move joints to (deg): {target_pos_deg}")
        success = robot.move_to_joint_positions(target_pos_deg, speed_percent=10)
        if success:
             logger.info("MOVJ command acknowledged. (Move completion not guaranteed)")
             time.sleep(3)
             logger.info("Reading position after move attempt...")
             new_positions = robot.get_joint_positions()
             if new_positions is not None:
                 logger.info(f"Position after move attempt (deg): {new_positions}")
             else:
                  logger.warning("Failed to read position after move.")
        else:
             logger.error("MOVJ command failed.")
        time.sleep(0.5)

        logger.info("Attempting to send STOP command...")
        if robot.stop_motion():
             logger.info("STOP command acknowledged.")
        else:
             logger.error("STOP command failed or NACKed.")

        robot.disconnect()
    else:
        logger.error("Connection failed. Check settings and robot state.") 