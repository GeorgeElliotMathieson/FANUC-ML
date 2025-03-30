import socket
import time
import struct
import threading

class FanucRobotController:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        self.lock = threading.Lock()
        
    def connect(self):
        """Establish connection to the robot controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5 second timeout
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            print(f"Connected to robot at {self.ip_address}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Close the connection to the robot"""
        if self.socket:
            self.socket.close()
        self.connected = False
        print("Disconnected from robot")
        
    def send_command(self, command):
        """Send a command to the robot and return the response"""
        if not self.connected:
            print("Not connected to robot")
            return None
            
        with self.lock:
            try:
                # Add termination character if not present
                if not command.endswith('\r'):
                    command += '\r'
                
                self.socket.sendall(command.encode())
                response = self.socket.recv(4096).decode().strip()
                return response
            except Exception as e:
                print(f"Error sending command: {e}")
                return None
                
    def get_joint_positions(self):
        """Get current joint positions"""
        response = self.send_command("RDJPOS")
        # Parse response according to FANUC protocol
        # This will depend on your specific controller setup
        # Example parsing (adjust based on actual response format):
        # Format: "JOINT: J1=X.XXX J2=X.XXX J3=X.XXX J4=X.XXX J5=X.XXX J6=X.XXX"
        if response and response.startswith("JOINT:"):
            values = response.split(" ")[1:]
            joint_positions = [float(joint.split("=")[1]) for joint in values]
            return joint_positions
        return None
        
    def move_to_joint_positions(self, positions):
        """Move robot to specified joint positions"""
        if len(positions) != 6:  # For 6-axis robot
            print("Invalid joint positions array length")
            return False
            
        # Format command (adjust based on your controller's command syntax)
        command = f"MOVJ {' '.join([str(pos) for pos in positions])}"
        response = self.send_command(command)
        
        # Check for successful acknowledgment
        if response and "OK" in response:
            return True
        else:
            print(f"Move command failed: {response}")
            return False
            
    def jog_joint(self, joint_index, direction, speed_percent=5):
        """Jog a specific joint (1-6) in a direction (+1 or -1)"""
        if joint_index < 1 or joint_index > 6:
            print("Invalid joint index")
            return False
            
        if direction not in [1, -1]:
            print("Invalid direction")
            return False
            
        # Format command (adjust based on your controller's command syntax)
        command = f"JOG JOINT {joint_index} {direction} {speed_percent}"
        response = self.send_command(command)
        
        # Check for successful acknowledgment
        if response and "OK" in response:
            return True
        else:
            print(f"Jog command failed: {response}")
            return False
            
    def stop_motion(self):
        """Emergency stop of robot motion"""
        response = self.send_command("STOP")
        return response and "OK" in response