# src/robot_api.py
import socket
import struct
import time
import numpy as np
import logging
import traceback # Import traceback for better error logging
import threading # Import threading for lock

logger = logging.getLogger(__name__)

# Default values based on deprecated/control/instructions.md and example_usage.py
DEFAULT_ROBOT_IP = "192.168.1.10" # <-- VERIFY ROBOT IP from instructions
DEFAULT_ROBOT_PORT = 6000        # <-- VERIFY PORT from instructions

# Timeout for socket operations (seconds)
DEFAULT_TIMEOUT = 5.0

class FANUCRobotAPI:
    """
    Handles communication with a FANUC robot controller using simple socket messaging.
    Based on the example logic found in deprecated/control/fanuc_robot_controller.py.

    **CRITICAL WARNING:** Assumes the controller is configured for Socket Messaging (SM)
    and responds to commands like RDJPOS, MOVJ, STOP with \r termination.
    Verify command syntax, parsing, units, and error handling against YOUR specific
    controller version and configuration.
    """
    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT):
        self.ip: str = ip
        self.port: int = port
        self.socket: socket.socket | None = None
        self.connected: bool = False
        self.timeout: float = DEFAULT_TIMEOUT
        self.lock = threading.Lock() # Add lock for thread safety if sending commands from different threads

        # Constants for joint limits (APPROXIMATE - VERIFY WITH YOUR ROBOT SPEC)
        # Using limits from fanuc_env.py (assuming LRMate 200iD - **VERIFY YOUR MODEL**)
        # Units: Degrees
        self.joint_limits_deg = {
            0: [-np.rad2deg(2.96), np.rad2deg(2.96)],  # J1 (~ +/- 170 deg)
            1: [-np.rad2deg(1.74), np.rad2deg(2.35)],  # J2 (~ -100/+135 deg)
            2: [-np.rad2deg(2.37), np.rad2deg(2.67)],  # J3 (~ -136/+153 deg)
            3: [-np.rad2deg(3.31), np.rad2deg(3.31)],  # J4 (~ +/- 190 deg)
            4: [-np.rad2deg(2.18), np.rad2deg(2.18)],  # J5 (~ +/- 125 deg)
            5: [-360*2, 360*2]                        # J6 Example (~ +/- 720 deg) - VERIFY if used
        }
        # Number of joints the RL agent controls (based on fanuc_env.py action space)
        self.num_controlled_joints: int = 5 # Store this based on RL environment
        # Number of joints reported by RDJPOS (typically 6 for LRMate)
        self.num_reported_joints: int = 6

        # Convert to radians for internal reference if needed (API uses degrees)
        self.joint_limits_rad = {
            k: [np.deg2rad(low), np.deg2rad(high)]
            for k, (low, high) in self.joint_limits_deg.items()
            # Only store rad limits for joints relevant to RL control
            if k < self.num_controlled_joints
        }
        logger.info(f"Initialized FANUCRobotAPI for {self.num_controlled_joints} controlled joints.")
        logger.info(f"  Target IP: {self.ip}, Port: {self.port}")
        logger.info("  Joint Limits (Degrees, Approx - VERIFY):")
        for i in range(self.num_controlled_joints):
            logger.info(f"    J{i+1}: {self.joint_limits_deg[i]}")

    def connect(self) -> bool:
        """Establish connection to the robot controller."""
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
            # Optional: Add an initial check like getting positions?
            # if self.get_joint_positions() is None:
            #     raise ConnectionError("Initial position check failed after connect.")
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
        """Close the connection to the robot."""
        if not self.connected:
            # logger.info("Already disconnected.")
            return
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR) # Graceful shutdown
                self.socket.close()
                logger.info("Socket closed.")
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        self.socket = None
        self.connected = False
        logger.info("Disconnected from FANUC robot.")

    def _send_command(self, command: str) -> str | None:
        """
        (Internal) Send a command string to the robot and get the response.
        Uses simple request-response with \r termination.
        Acquires a lock for thread safety.
        """
        if not self.check_connection() or self.socket is None:
            logger.warning("Not connected to robot. Cannot send command.")
            return None

        with self.lock: # Ensure only one command is sent/received at a time
            try:
                # Add termination character if not present
                if not command.endswith('\r'):
                    command += '\r'

                logger.debug(f"Sending command: '{command.strip()}'")
                self.socket.sendall(command.encode('ascii')) # Use ascii based on typical FANUC interfaces

                # Receive response (blocking until response or timeout)
                response_bytes = b''
                self.socket.settimeout(self.timeout) # Ensure timeout is set for receive
                while True:
                     # Read small chunks to find termination character
                     chunk = self.socket.recv(64)
                     if not chunk:
                          logger.error("Connection closed by robot while receiving response.")
                          self._handle_connection_error()
                          return None
                     response_bytes += chunk
                     # Check for termination character (typically \r)
                     if b'\r' in response_bytes:
                          break
                     # Optional: Add check for maximum response length if needed

                # Decode and strip response
                response_str = response_bytes.decode('ascii').strip()
                logger.debug(f"Received response: '{response_str}'")
                return response_str

            except socket.timeout:
                 logger.warning(f"Timeout waiting for robot response after sending '{command.strip()}'.")
                 # Consider if timeout means failure or just no response needed
                 return None # Assume timeout is failure for now
            except Exception as e:
                logger.error(f"Error sending/receiving for command '{command.strip()}': {e}")
                logger.error(traceback.format_exc())
                self._handle_connection_error() # Assume connection is compromised
                return None

    def _handle_connection_error(self):
        """Centralized handling for potential connection loss."""
        logger.error("Connection error detected. Marking as disconnected.")
        # Attempt to close socket cleanly, ignore errors
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.socket = None
        self.connected = False
        # Optionally: Trigger automatic reconnection attempt logic here?

    # ==========================================================================
    # --- High-Level API Methods (Adapted from deprecated controller) ---
    # ==========================================================================

    def get_joint_positions(self) -> np.ndarray | None:
        """
        Get current joint positions in **degrees** using RDJPOS command.

        Returns:
            np.ndarray | None: Array of joint positions (degrees) for the
                               number of reported joints (e.g., 6), or None if failed.
        """
        command = "RDJPOS"
        response = self._send_command(command)
        if response and isinstance(response, str):
            try:
                # Example parsing from deprecated controller:
                # Format: "JOINT: J1=X.XXX J2=X.XXX J3=X.XXX J4=X.XXX J5=X.XXX J6=X.XXX"
                if response.startswith("JOINT:"):
                    parts = response.split(" ")[1:] # Split by space, skip "JOINT:"
                    if len(parts) >= self.num_reported_joints:
                        # Extract values after '=' sign
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
        return None # Return None on failure

    def move_to_joint_positions(self, positions_deg: list | np.ndarray, speed_percent: int = 20) -> bool:
        """
        Move robot to specified joint positions using MOVJ command.

        Args:
            positions_deg: List or array of target joint positions in DEGREES.
                           Length should match the number of reported joints (e.g., 6).
            speed_percent: Speed percentage (1-100) for the move. Default 20.

        Returns:
            bool: True if the command was SENT and acknowledged successfully,
                  False otherwise. Note: This does not guarantee move completion.
        """
        if positions_deg is None or len(positions_deg) != self.num_reported_joints:
            logger.error(f"Invalid joint positions array length provided (expected {self.num_reported_joints} dims): {positions_deg}")
            return False

        # Optional: Client-side limit check (controller is primary enforcer)
        self._log_limit_warnings(np.array(positions_deg))

        # Format command based on deprecated controller example:
        # "MOVJ pos1 pos2 pos3 pos4 pos5 pos6"
        # Speed is often handled separately or via robot registers, not in MOVJ.
        # Add speed/termination args ONLY if your specific interface requires them here.
        pos_str = ' '.join([f"{pos:.3f}" for pos in positions_deg])
        command = f"MOVJ {pos_str}"
        # Example if speed/term needed: command = f"MOVJ {pos_str} S{speed_percent} CNT100"

        logger.debug(f"Sending command: '{command}'")
        response = self._send_command(command)

        # Check for successful acknowledgment (adjust "OK" based on actual response)
        if response and "OK" in response:
            logger.info(f"MOVJ command acknowledged successfully.")
            # **IMPORTANT:** Real applications NEED to monitor robot status
            # to confirm move completion before sending the next command.
            # This basic implementation only checks acknowledgment.
            return True
        else:
            logger.error(f"MOVJ command failed or NACKed: {response}")
            return False

    def stop_motion(self) -> bool:
        """Emergency stop of robot motion using STOP command."""
        logger.warning("Sending STOP command!")
        response = self._send_command("STOP")
        if response and "OK" in response:
             logger.info("STOP command acknowledged.")
             return True
        else:
             logger.error(f"STOP command failed or NACKed: {response}")
             return False

    # ==========================================================================
    # --- Methods needing implementation based on specific interface/needs ---
    # ==========================================================================

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Attempts to read pose data using a common command (RDPOS) and logs the raw response.
        This helps diagnose if pose data is available and determine its format.
        **Does not parse the response.** Implement parsing based on observed data.

        Returns:
            None: This method currently always returns None after logging.
        """
        # Try a common command for reading Cartesian position
        command = "RDPOS"
        logger.info(f"Attempting to get End-Effector pose using command: '{command}'")
        response = self._send_command(command)

        if response is not None:
            logger.info(f"Received potential pose data (raw): '{response}'")
            logger.warning("Pose data received, but parsing is NOT implemented.")
            logger.warning("Please analyse the raw response above and update the parsing logic in get_end_effector_pose.")
        else:
            logger.warning(f"No response received for '{command}'. Robot may not support this command or is not sending pose data.")

        # Always return None as parsing is not implemented
        return None

    def get_robot_state_observation(self) -> np.ndarray | None:
        """
        Get the robot state formatted for the RL agent's observation space.

        **CRITICAL:** This REQUIRES adaptation to match `src/fanuc_env.py`'s
        observation space precisely (21 dims: pos_rad[5], vel_rad_s[5], rel_target[3],
        norm_limits[5], rel_obstacle[3]).

        - Joint velocities are typically NOT available via simple socket APIs.
          Requires estimation or specific interface commands.
        - Relative target/obstacle info MUST be calculated externally.

        Returns:
            np.ndarray | None: Observation vector (21 dims, float32) or None if failed.
        """

        # 1. Get Joint Positions (degrees -> radians, taking first N)
        joint_pos_deg_all = self.get_joint_positions()
        if joint_pos_deg_all is None:
            logger.error("Failed to get joint positions for observation.")
            return None
        # Take only the joints controlled by the RL agent
        joint_pos_deg = joint_pos_deg_all[:self.num_controlled_joints]
        joint_pos_rad = np.deg2rad(joint_pos_deg)

        # 2. Get Joint Velocities (rad/s) - **NEEDS IMPLEMENTATION**
        # Placeholder: Assume zero velocity.
        joint_vel_rad_s = np.zeros(self.num_controlled_joints)
        # logger.warning("Joint velocity reporting is using zero placeholders.") # Reduce log spam

        # 3. Relative Target Position - **EXTERNAL CALCULATION NEEDED**
        # Placeholder: Zeros. Deploy script must overwrite this.
        relative_target = np.zeros(3)
        # logger.warning("Relative target in observation is using zero placeholders.")

        # 4. Normalised Joint Positions (-1 to 1)
        norm_joint_pos = np.zeros(self.num_controlled_joints)
        for i in range(self.num_controlled_joints):
             if i in self.joint_limits_rad:
                 low, high = self.joint_limits_rad[i]
                 joint_range = high - low
                 if joint_range > 1e-6:
                     norm_joint_pos[i] = 2 * (joint_pos_rad[i] - low) / joint_range - 1.0
                 # else: norm_joint_pos[i] stays 0.0
             else:
                  # This shouldn't happen if joint_limits_rad is derived correctly
                  logger.error(f"Joint index {i} not found in radian limits for normalization!")
        norm_joint_pos = np.clip(norm_joint_pos, -1.0, 1.0)

        # 5. Relative Obstacle Position - **EXTERNAL CALCULATION NEEDED**
        # Placeholder: Zeros. Deploy script must overwrite this.
        relative_obstacle = np.zeros(3)
        # logger.warning("Relative obstacle in observation is using zero placeholders.")

        # --- Concatenate Observation (Order MUST match fanuc_env.py) --- 
        try:
            obs = np.concatenate([
                joint_pos_rad,           # 5 dims
                joint_vel_rad_s,         # 5 dims
                relative_target,         # 3 dims
                norm_joint_pos,          # 5 dims
                relative_obstacle        # 3 dims
            ]).astype(np.float32)
        except ValueError as e:
             logger.error(f"Error concatenating observation components: {e}")
             logger.error(f"  Shapes: pos={joint_pos_rad.shape}, vel={joint_vel_rad_s.shape}, "
                          f"tgt={relative_target.shape}, norm={norm_joint_pos.shape}, obs={relative_obstacle.shape}")
             return None

        # Final shape check
        expected_dim = 21 # Based on fanuc_env.py
        if obs.shape != (expected_dim,):
            logger.error(f"Constructed observation has wrong shape: {obs.shape}, expected ({expected_dim},).")
            return None

        return obs

    def check_connection(self) -> bool:
        """Check if the robot connection is likely still active."""
        if not self.connected or self.socket is None:
            return False
        # Try a minimal check: can we get socket options without error?
        try:
             # Setting timeout to 0 makes getsockopt non-blocking
             # self.socket.settimeout(0.0)
             err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
             # self.socket.settimeout(self.timeout) # Restore original timeout
             if err == 0:
                 return True
             else:
                  logger.warning(f"Socket check failed with error code: {err}")
                  self._handle_connection_error()
                  return False
        except socket.timeout:
             # This shouldn't happen with timeout 0, but handle defensively
             logger.warning("Socket check timed out unexpectedly.")
             self._handle_connection_error()
             return False
        except Exception as e:
             logger.warning(f"Socket check failed with exception: {e}")
             self._handle_connection_error()
             return False

    def _log_limit_warnings(self, joint_target_deg: np.ndarray):
        """(Internal) Logs warnings if commanded positions exceed client-side limits."""
        num_to_check = min(len(joint_target_deg), len(self.joint_limits_deg))
        for i in range(num_to_check):
            pos = joint_target_deg[i]
            if i in self.joint_limits_deg:
                low, high = self.joint_limits_deg[i]
                if not (low <= pos <= high):
                    logger.warning(f"Commanded position {pos:.2f} for J{i+1} exceeds client-side limits [{low}, {high}]. Relying on controller limits.")

# Example usage (for testing connection - adapt IP, commands, parsing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, # Use DEBUG for detailed testing
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    # --- REPLACE WITH YOUR ROBOT'S ACTUAL IP --- 
    robot_ip = DEFAULT_ROBOT_IP # Use default or replace
    robot_port = DEFAULT_ROBOT_PORT # Use default or replace
    # ----------------------------------------- 
    logger.info(f"--- Robot API Test Script ---")
    logger.info(f"Attempting connection to {robot_ip}:{robot_port}")

    robot = FANUCRobotAPI(ip=robot_ip, port=robot_port)

    if robot.connect():
        logger.info("Connection successful.")

        # Test getting positions
        logger.info("Attempting to get joint positions (using RDJPOS command/parsing)...")
        positions = robot.get_joint_positions()
        if positions is not None:
            logger.info(f"Received Joint Positions (degrees): {positions}")
        else:
            logger.warning("Failed to get joint positions.")
        time.sleep(0.5)

        # Test getting pose (will show warning as it's not implemented)
        logger.info("Attempting to get EE pose (using placeholder - expect warning)...")
        pose = robot.get_end_effector_pose()
        if pose is not None:
             pos, orient = pose
             logger.info(f"Received EE Position (mm?): {pos}")
             logger.info(f"Received EE Orientation (WPR deg?): {orient}")
        else:
            logger.warning("Failed to get EE pose (expected for placeholder).")
        time.sleep(0.5)

        # Test getting observation vector
        logger.info("Attempting to get state observation vector...")
        obs = robot.get_robot_state_observation()
        if obs is not None:
            logger.info(f"Received Observation Vector (shape {obs.shape}):")
            # Print components for clarity
            logger.info(f"  Pos (rad): {obs[0:robot.num_controlled_joints]}")
            logger.info(f"  Vel (rad/s): {obs[robot.num_controlled_joints:robot.num_controlled_joints*2]} (Placeholder!)")
            logger.info(f"  Rel Target: {obs[robot.num_controlled_joints*2:robot.num_controlled_joints*2+3]} (Placeholder!)")
            logger.info(f"  Norm Limits: {obs[robot.num_controlled_joints*2+3:robot.num_controlled_joints*3+3]}")
            logger.info(f"  Rel Obstacle: {obs[robot.num_controlled_joints*3+3:]} (Placeholder!)")

        else:
             logger.warning("Failed to get observation vector.")
        time.sleep(0.5)

        # Test moving (Use example from deprecated controller)
        target_pos_deg = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0] # Example 6 joint values
        logger.info(f"Attempting to move joints to: {target_pos_deg}")
        success = robot.move_to_joint_positions(target_pos_deg, speed_percent=10)
        if success:
             logger.info("MOVJ command acknowledged. (Move completion not guaranteed by this check)")
             # In a real app, wait/check status here
             time.sleep(3) # Simple pause
             logger.info("Reading position after move attempt...")
             new_positions = robot.get_joint_positions()
             if new_positions is not None:
                 logger.info(f"Position after move attempt: {new_positions}")
             else:
                  logger.warning("Failed to read position after move.")
        else:
             logger.error("MOVJ command failed.")
        time.sleep(0.5)

        # Test stop
        logger.info("Attempting to send STOP command...")
        if robot.stop_motion():
             logger.info("STOP command acknowledged.")
        else:
             logger.error("STOP command failed or NACKed.")

        robot.disconnect()
    else:
        logger.error("Connection failed. Please check settings and robot state.") 