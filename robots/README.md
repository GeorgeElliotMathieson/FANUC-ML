# FANUC Robot Models

This directory contains the robot models, including URDF files and mesh assets used by the FANUC Robot ML Platform.

## Directory Structure

- `urdf/`: Contains URDF model files for the robots
  - `fanuc.urdf`: The main URDF file for the FANUC robot with detailed meshes
- `meshes/`: Contains STL and other 3D mesh files for the robot parts
- `fanuc/`: Contains FANUC-specific configurations and auxiliary files

## Robot Specifications

The FANUC robot models have:
- 6 revolute joints for the arm
- Accurate inertial properties from the original FANUC specifications 
- Realistic joint limits and constraints
- Properly configured collision geometries for simulation

## Using in Your Projects

To use these robot models in your own PyBullet projects:

```python
import pybullet as p
import os

# Initialize PyBullet
p.connect(p.GUI)

# Get the path to the URDF file - update path as needed
urdf_path = os.path.join("robots", "urdf", "fanuc.urdf")
robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# Control the robot
# ...
```

The FANUC robot models available in this directory implement accurate physical properties that help ensure realistic dynamics simulation, especially when performing rapid movements or when subjected to external forces.

## Demo Scripts

For examples of how to use these robot models, refer to the demo scripts in the `tools/demos` directory, which showcase basic robot loading and joint control.

## Credits

The original model is based on [sezan92/Fanuc](https://github.com/sezan92/Fanuc.git) with modifications to make the robot easier to use with PyBullet, including:

- Updated joint names and structure
- Added joint limits
- Relative file paths for meshes
- Preservation of original inertial properties for accurate physics simulation 