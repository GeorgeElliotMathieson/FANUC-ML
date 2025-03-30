# deploy.py
import numpy as np
import time
import argparse
from transfer_learning import RobotTransfer
from robot_api import FANUCRobotAPI

def deploy_model():
    parser = argparse.ArgumentParser(description='Deploy trained model to real FANUC robot')
    parser.add_argument('--ip', type=str, default='192.168.1.1', help='Robot IP address')
    parser.add_argument('--port', type=int, default=18735, help='Robot port')
    parser.add_argument('--model', type=str, default='./models/fanuc_final_model.zip', help='Path to trained model')
    args = parser.parse_args()

    # Create API connection to robot
    robot = FANUCRobotAPI(ip=args.ip, port=args.port)
    
    # Connect to the robot
    if not robot.connect():
        print("Failed to connect to robot. Exiting.")
        return
    
    try:
        # Initialize transfer learning module
        transfer = RobotTransfer(model_path=args.model)
        
        # Calibration phase (collect paired sim/real data)
        print("Starting calibration...")
        sim_states = []
        real_states = []
        sim_actions = []
        real_actions = []
        
        # Move to a few predefined positions to collect calibration data
        calibration_positions = [
            [0, 0, 0, 0, 0, 0],  # Home position
            [30, 20, -10, 0, 0, 0],
            [-30, 20, -10, 0, 0, 0],
            [0, 40, -20, 0, 0, 0]
        ]
        
        for pos in calibration_positions:
            print(f"Moving to calibration position: {pos}")
            
            # Move robot to position
            robot.set_joint_positions(pos)
            time.sleep(2)  # Wait for movement to complete
            
            # Get real robot state
            real_state = robot.get_robot_state()
            if real_state is None:
                continue
                
            # Generate sim state and action for this position
            # In a real implementation, you would have a simulator running in parallel
            # This is a simplified placeholder
            sim_state = real_state + np.random.normal(0, 0.01, size=real_state.shape)  # Add some noise
            
            # Store data for calibration
            sim_states.append(sim_state)
            real_states.append(real_state)
        
        # Convert to numpy arrays
        sim_states = np.array(sim_states)
        real_states = np.array(real_states)
        sim_actions = np.array(sim_actions) if sim_actions else np.empty((0, 6))
        real_actions = np.array(real_actions) if real_actions else np.empty((0, 6))
        
        # Calibrate transfer model
        transfer.calibrate(sim_states, real_states, sim_actions, real_actions)
        print("Calibration complete.")
        
        # Main control loop
        print("Starting control loop. Press Ctrl+C to exit.")
        
        # Define target points
        targets = [
            np.array([0.5, 0, 0.5]),  # Example target position
            np.array([0.5, -0.2, 0.4]),
            np.array([0.5, 0.2, 0.4]),
            np.array([0.4, 0, 0.6])
        ]
        
        target_idx = 0
        while True:
            # Get current target
            current_target = targets[target_idx]
            print(f"Moving to target {target_idx + 1}/{len(targets)}: {current_target}")
            
            # Get robot state
            state = robot.get_robot_state()
            if state is None:
                print("Failed to get robot state. Retrying...")
                time.sleep(1)
                continue
            
            # Predict action using the transfer model
            action = transfer.predict(state)
            
            # Send action to robot
            success = robot.set_joint_positions(np.rad2deg(action))
            if not success:
                print("Failed to send action to robot. Retrying...")
                time.sleep(1)
                continue
            
            # Check if we've reached the target
            ee_pose = robot.get_end_effector_pose()
            if ee_pose is not None:
                ee_position, _ = ee_pose
                distance = np.linalg.norm(ee_position - current_target)
                print(f"Distance to target: {distance:.4f}")
                
                if distance < 0.05:  # 5cm tolerance
                    print(f"Reached target {target_idx + 1}!")
                    
                    # Move to next target
                    target_idx = (target_idx + 1) % len(targets)
                    time.sleep(1)  # Wait a bit before moving to next target
            
            time.sleep(0.1)  # Control loop rate
            
    except KeyboardInterrupt:
        print("Control interrupted by user.")
    finally:
        # Disconnect from robot
        robot.disconnect()

if __name__ == "__main__":
    deploy_model()