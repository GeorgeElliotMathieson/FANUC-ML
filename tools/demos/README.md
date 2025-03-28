# FANUC Robot Demonstration Scripts

This directory contains standalone demonstration scripts that showcase the FANUC robot's capabilities using PyBullet simulation. These scripts are independent of the main training framework and provide simple examples of how to interact with the robot.

## Available Demos

- `load_fanuc_robot.py` - Basic script to load a FANUC robot in PyBullet and demonstrate joint motion with a sequence of predefined poses.

## Running the Demos

To run a demo, simply execute the Python script:

```bash
# From the project root directory:
python tools/demos/load_fanuc_robot.py
```

## Features

- Robot loading and visualization
- Joint control and movement
- Simulation parameter optimization
- Examples of how to interact with the PyBullet API

These demos serve as practical examples for developers who want to understand the low-level robot control without diving into the full reinforcement learning framework. 