# robot_api.py
import socket
import struct
import time
import numpy as np

class FANUCRobotAPI:
    def __init__(self, ip='192.168.1.1', port=18735):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connected = False
        
        # Constants for joint limits (from manual)
        self.joint_limits = {
            0: [-170, 170],  # J1 axis (in degrees)
            1: [-60, 75],    # J2 axis
            2: [-70, 50],    # J3 axis
            3: [-170, 170],  # J4 axis
            4: [-110, 110],  # J5 axis
            5: [-360, 360]   # J6 axis
        }
    
    def connect(self):
        """Connect to the robot controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5 second timeout
            self.socket.connect((self.ip, self.port))
            self.connected = True
            print(f"Connected to FANUC robot at {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the robot controller"""
        if self.socket:
            self.socket.close()
        self.connected = False
        print("Disconnected from FANUC robot")
    
    def _send_command(self, command):
        """Send a command to the robot"""
        if not self.connected:
            print("Not connected to robot")
            return False
        
        try:
            self.socket.sendall(command.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def _receive_response(self, buffer_size=1024):
        """Receive response from the robot"""
        if not self.connected:
            print("Not connected to robot")
            return None
        
        try:
            response = self.socket.recv(buffer_size)
            return response
        except Exception as e:
            print(f"Error receiving response: {e}")
            return None
    
    def get_joint_positions(self):
        """Get current joint positions"""
        if not self.connected:
            print("Not connected to robot")
            return None
        
        try:
            # Send command to get joint positions
            self._send_command("RPOS")
            
            # Receive response
            response = self._receive_response()
            
            # Parse response (format depends on robot controller)
            # This is a simplified example; real implementation would depend on the protocol
            joint_positions = [float(val) for val in response.decode('utf-8').split(',')]
            
            return np.array(joint_positions)
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return None
    
    def set_joint_positions(self, joint_positions, speed=10):
        """
        Move robot to specified joint positions
        
        Args:
            joint_positions: Array of 6 joint positions in degrees
            speed: Speed percentage (1-100)
        
        Returns:
            success: Whether the command was successful
        """
        if not self.connected:
            print("Not connected to robot")
            return False
        
        try:
            # Clip joint positions to limits for safety
            for i, pos in enumerate(joint_positions):
                if i in self.joint_limits:
                    low, high = self.joint_limits[i]
                    joint_positions[i] = np.clip(pos, low, high)
            
            # Create command string (format depends on robot controller)
            # This is a simplified example; real implementation would depend on the protocol
            cmd = f"MOVJ {','.join([str(pos) for pos in joint_positions])} {speed}"
            
            # Send command
            success = self._send_command(cmd)
            
            # Wait for movement to complete
            # This is a simplified approach; real implementation would handle this differently
            if success:
                time.sleep(2)  # Simple delay
                
                # Check if we've reached the target position
                current_pos = self.get_joint_positions()
                if current_pos is not None:
                    error = np.abs(np.array(current_pos) - np.array(joint_positions)).max()
                    if error > 1.0:  # 1 degree tolerance
                        print(f"Warning: Position error of {error} degrees")
            
            return success
        except Exception as e:
            print(f"Error setting joint positions: {e}")
            return False
    
    def get_end_effector_pose(self):
        """Get current end effector pose (position and orientation)"""
        if not self.connected:
            print("Not connected to robot")
            return None
        
        try:
            # Send command to get end effector pose
            self._send_command("RPOS_TOOL")
            
            # Receive response
            response = self._receive_response()
            
            # Parse response (format depends on robot controller)
            # This is a simplified example; real implementation would depend on the protocol
            values = [float(val) for val in response.decode('utf-8').split(',')]
            
            # Assuming values are [x, y, z, rx, ry, rz]
            position = values[:3]
            orientation = values[3:]
            
            return np.array(position), np.array(orientation)
        except Exception as e:
            print(f"Error getting end effector pose: {e}")
            return None
    
    def get_robot_state(self):
        """
        Get complete robot state for RL algorithm
        
        Returns:
            state: State vector compatible with RL algorithm
        """
        if not self.connected:
            print("Not connected to robot")
            return None
        
        try:
            # Get joint positions
            joint_positions = self.get_joint_positions()
            
            # Get joint velocities (if available)
            # This is a simplified approach; real implementation would depend on the protocol
            joint_velocities = np.zeros(6)  # Placeholder
            
            # Get end effector pose
            ee_pose = self.get_end_effector_pose()
            if ee_pose is None:
                return None
                
            ee_position, ee_orientation = ee_pose
            
            # Combine into state vector
            # 6 joint positions + 6 joint velocities + 3 EE position + 4 EE orientation (assuming quaternion)
            state = np.concatenate([
                joint_positions,
                joint_velocities,
                ee_position,
                ee_orientation
            ])
            
            return state
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return None