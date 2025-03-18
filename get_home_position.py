import pybullet as p
import pybullet_data
import os
import time

# Connect to the physics server
client = p.connect(p.DIRECT)  # Use DIRECT mode to avoid GUI issues
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Open a file to write the results
with open("home_position.txt", "w") as f:
    # Load the URDF file
    urdf_path = 'res/fanuc_lrmate_200ic.urdf'
    f.write(f"Loading URDF from: {urdf_path}\n")
    f.write(f"File exists: {os.path.exists(urdf_path)}\n")

    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

    # Reset to home position
    home_position = [0, 0, 0, 0, 0, 0]  # All joints at 0 position
    for i, pos in enumerate(home_position):
        p.resetJointState(robot_id, i, pos)

    # Step simulation a few times to ensure the robot is settled
    for _ in range(10):
        p.stepSimulation()

    # Get end-effector position
    ee_link_state = p.getLinkState(robot_id, 5)  # Assuming link 5 is the end effector
    ee_position = ee_link_state[0]
    ee_orientation = ee_link_state[1]

    f.write(f"End effector home position: {ee_position}\n")
    f.write(f"End effector home orientation: {ee_orientation}\n")

    # Get joint information
    f.write("\nJoint Information:\n")
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        f.write(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}\n")

    # Get link information
    f.write("\nLink Information:\n")
    for i in range(p.getNumJoints(robot_id)):
        link_state = p.getLinkState(robot_id, i)
        f.write(f"Link {i} position: {link_state[0]}\n")

# Disconnect
p.disconnect(client)

print("Results written to home_position.txt") 