from fanuc_robot_controller import FanucRobotController
import time

def main():
    # Initialize robot connection
    robot = FanucRobotController('192.168.1.10', 6000)
    
    if not robot.connect():
        print("Failed to connect to robot")
        return
    
    try:
        # Get current joint positions
        current_positions = robot.get_joint_positions()
        print(f"Current joint positions: {current_positions}")
        
        # Move to a specific position
        target_positions = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # Example joint values in degrees
        print(f"Moving to: {target_positions}")
        success = robot.move_to_joint_positions(target_positions)
        
        if success:
            print("Move completed successfully")
            # Wait for motion to complete
            time.sleep(2)
            
            # Read back positions to verify
            new_positions = robot.get_joint_positions()
            print(f"New joint positions: {new_positions}")
        
    except Exception as e:
        print(f"Error during robot operation: {e}")
        # Emergency stop if there's an error
        robot.stop_motion()
    
    finally:
        # Always disconnect properly
        robot.disconnect()

if __name__ == "__main__":
    main()