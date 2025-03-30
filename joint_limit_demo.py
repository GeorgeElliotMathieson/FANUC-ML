import pybullet as p
import pybullet_data
import time
import os
import numpy as np

# --- Constants (adapted from fanuc_env.py) ---
# Using the limits defined in fanuc_env.py as the URDF uses 'continuous'
JOINT_LIMITS_LOWER = np.array([-2.96, -1.74, -2.37, -3.31, -2.18]) # J1-J5 radians
JOINT_LIMITS_UPPER = np.array([ 2.96,  2.35,  2.67,  3.31,  2.18]) # J1-J5 radians
NUM_CONTROLLABLE_JOINTS = 5
SIMULATION_TIME_STEP = 1.0/240.0
MOVEMENT_DURATION_STEPS = 240 # Number of sim steps to move (e.g., 1 second at 240Hz)
PAUSE_DURATION = 0.5 # Seconds to pause at limits/home

def main():
    """Runs the joint limit demonstration."""
    physics_client = -1
    try:
        # --- PyBullet Setup ---
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Set camera position for better view
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-15, cameraTargetPosition=[0,0,0.4])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=SIMULATION_TIME_STEP)

        # Load plane
        p.loadURDF("plane.urdf")

        # Load robot (ensure paths are correct relative to this script)
        # Assumes script is run from the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) # Use absolute path of script dir
        urdf_file_path = os.path.join(script_dir, "Fanuc", "urdf", "Fanuc.urdf")
        mesh_path = os.path.join(script_dir, "Fanuc")

        if not os.path.exists(urdf_file_path):
             print(f"Error: Cannot find URDF file at {urdf_file_path}")
             print("Ensure the 'Fanuc' directory with the URDF is present.")
             return

        # Add Fanuc directory to search path *before* loading URDF
        p.setAdditionalSearchPath(mesh_path)

        print(f"Loading robot from: {urdf_file_path}")
        robot_id = p.loadURDF(
            urdf_file_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION, # Use defined collision meshes + self-collision
        )
        print(f"Robot loaded with ID: {robot_id}")

        # --- Get Robot Info ---
        num_joints = p.getNumJoints(robot_id)
        joint_indices = []
        joint_names = [] # Store names for printing
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_type = info[2]
            joint_name = info[1].decode('UTF-8')
            # Identify controllable joints (assuming REVOLUTE type in URDF corresponds to controllable axes)
            if joint_type == p.JOINT_REVOLUTE:
                if len(joint_indices) < NUM_CONTROLLABLE_JOINTS:
                    joint_indices.append(i)
                    joint_names.append(joint_name)

        if len(joint_indices) != NUM_CONTROLLABLE_JOINTS:
            print(f"Error: Found {len(joint_indices)} controllable revolute joints, expected {NUM_CONTROLLABLE_JOINTS}.")
            print(f"Found joints: {joint_names}")
            return

        print(f"Found {NUM_CONTROLLABLE_JOINTS} controllable joints:")
        for i in range(NUM_CONTROLLABLE_JOINTS):
             print(f"  - {joint_names[i]} (Index: {joint_indices[i]}) - Limits: [{JOINT_LIMITS_LOWER[i]:.2f}, {JOINT_LIMITS_UPPER[i]:.2f}] rad")

        print("\nResetting robot to home position...")
        home_position = [0.0] * NUM_CONTROLLABLE_JOINTS
        for i, joint_index in enumerate(joint_indices):
             p.resetJointState(robot_id, joint_index, targetValue=home_position[i], targetVelocity=0)
        # Let simulation settle briefly
        for _ in range(50):
             p.stepSimulation()
             time.sleep(SIMULATION_TIME_STEP)

        print("Starting joint limit demonstration...")

        # --- Animation Loop ---
        for i in range(NUM_CONTROLLABLE_JOINTS):
            current_joint_pybullet_index = joint_indices[i]
            current_joint_name = joint_names[i]
            print(f"\n--- Testing Joint {i+1}: {current_joint_name} (Index: {current_joint_pybullet_index}) ---")

            # --- Move to Lower Limit ---
            print(f"  -> Moving to Lower Limit ({JOINT_LIMITS_LOWER[i]:.2f} rad)")
            target_pos = list(home_position) # Start from home for other joints
            target_pos[i] = JOINT_LIMITS_LOWER[i]
            # Use setJointMotorControlArray for controlling multiple joints simultaneously
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
                # Optional: Add forces/maxVelocity if needed for smooth control
                # forces=[100]*NUM_CONTROLLABLE_JOINTS
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at the limit

            # --- Move to Upper Limit ---
            print(f"  -> Moving to Upper Limit ({JOINT_LIMITS_UPPER[i]:.2f} rad)")
            target_pos = list(home_position) # Start from home for other joints
            target_pos[i] = JOINT_LIMITS_UPPER[i]
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at the limit

            # --- Return to Home Position ---
            print("  -> Returning to Home Position")
            target_pos = list(home_position)
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_pos,
            )
            # Simulate the movement
            for _ in range(MOVEMENT_DURATION_STEPS):
                p.stepSimulation()
                time.sleep(SIMULATION_TIME_STEP)
            time.sleep(PAUSE_DURATION) # Pause at home


        print("\nDemonstration finished.")
        time.sleep(3) # Keep window open briefly

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- Cleanup ---
        if physics_client >= 0 and p.isConnected(physics_client):
             print("Disconnecting PyBullet.")
             p.disconnect(physics_client)

if __name__ == "__main__":
    main() 