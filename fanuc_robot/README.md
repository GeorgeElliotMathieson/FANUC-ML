# FANUC Robot URDF Implementation

This directory contains a URDF implementation of a 6-axis FANUC robot arm, based on the model from [sezan92/Fanuc](https://github.com/sezan92/Fanuc.git).

## Contents

- `urdf/fanuc.urdf`: The URDF file describing the 6-axis FANUC robot
- `meshes/`: Directory containing the STL mesh files for the robot parts
- `load_fanuc_robot.py`: Python script to load and demonstrate the robot in PyBullet

## Requirements

To run the demo script, you need the following Python packages:
```
pybullet
numpy
```

You can install them with pip:
```
pip install pybullet numpy
```

## Running the Demo

To run the robot demo:

```
cd fanuc_robot
python load_fanuc_robot.py
```

The demo will:
1. Load the FANUC robot in a PyBullet simulation
2. Print information about the robot's joints
3. Move the robot through a series of poses
4. Keep the simulation running until you press Ctrl+C

## Robot Specifications

The FANUC robot model has:
- 6 revolute joints for the arm
- Accurate inertial properties from the original FANUC specifications 
- Realistic joint limits and constraints

This implementation preserves the original inertia tensors from the source repository for maximum physical accuracy. These precise values help ensure realistic dynamics simulation of the robot's motion, especially when performing rapid movements or when subjected to external forces.

## Customizing the Robot

You can customize the robot by:
- Modifying the joint limits in the URDF file
- Changing the visual appearance by updating the material colors
- Adding custom end effectors to the robot's tool flange (tool0 link)

## Using in Your Projects

To use this robot in your own PyBullet projects:

```python
import pybullet as p
import os

# Initialize PyBullet
p.connect(p.GUI)

# Load the robot URDF
urdf_path = os.path.join("path/to/fanuc_robot", "urdf", "fanuc.urdf")
robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# Control the robot
# ...
```

## Credits

The original model is from [sezan92/Fanuc](https://github.com/sezan92/Fanuc.git). This implementation includes modifications to make the robot easier to use with PyBullet, including:

- Updated joint names and structure
- Added joint limits
- Relative file paths for meshes instead of ROS package paths
- Preservation of original inertial properties for accurate physics simulation 