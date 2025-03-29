#!/usr/bin/env python3
# pybullet_utils.py
# Utility functions for PyBullet visualizations and robot simulation

import pybullet as p  # type: ignore
import numpy as np
import time
import os
import threading
import random
import logging
from typing import List, Tuple, Dict, Optional, Union, Any, Callable, TypeVar

import pybullet_data  # type: ignore
from pybullet_utils import bullet_client  # type: ignore

# Define type for PyBullet client
# BulletClient = TypeVar('BulletClient', bound=p.BulletClient)
BulletClient = TypeVar('BulletClient')

# Global variables to track PyBullet connections
_GUI_CONNECTION_ESTABLISHED: bool = False
_GUI_CLIENT_ID: Optional[int] = None

# Global variable to track client ID for shared client and lock for thread safety
_SHARED_CLIENT_ID: Optional[int] = None
_SHARED_CLIENT_LOCK = threading.Lock()

# Global variables for shared PyBullet client
_SHARED_PYBULLET_CLIENT: Optional[Any] = None
_SHARED_PYBULLET_CLIENT_GUI_STATE: bool = False

def get_visualization_settings_from_env():
    """
    Get visualization and verbosity settings from environment variables.
    
    Checks for FANUC_VISUALIZE and FANUC_VERBOSE environment variables and 
    returns their values as booleans. Handles different formats of boolean values.
    
    Returns:
        Tuple of (visualize, verbose) as booleans
    """
    try:
        # Get visualization setting from environment variable
        visualize = os.environ.get('FANUC_VISUALIZE')
        if visualize is not None:
            # Convert various string formats to boolean
            visualize = visualize.lower() in ('1', 'true', 'yes', 'y', 'on', 't')
        else:
            # Default value if not set
            visualize = True
        
        # Get verbosity setting from environment variable
        verbose = os.environ.get('FANUC_VERBOSE')
        if verbose is not None:
            # Convert various string formats to boolean
            verbose = verbose.lower() in ('1', 'true', 'yes', 'y', 'on', 't')
        else:
            # Default value if not set
            verbose = False
        
        return visualize, verbose
    except Exception as e:
        # Handle errors - default to visualization on, verbosity off
        print(f"Warning: Error getting visualization settings from environment: {e}")
        print("Using default values: visualization=True, verbose=False")
        return True, False

def get_pybullet_client(
    gui: bool = False, 
    realtime: bool = False, 
    fps: int = 240, 
    options: str = "", 
    training_mode: bool = False
) -> Any:
    """
    Create a PyBullet client, either GUI or Direct (headless).
    Automatically creates a new client or returns a cached one if one is available.
    
    Args:
        gui: Whether to use GUI (True) or Direct (False, headless)
        realtime: Whether to use realtime simulation
        fps: Frames per second to use for stepping
        options: Options to pass to PyBullet
        training_mode: Whether client is for training (optimized for speed)
        
    Returns:
        PyBullet client
    """
    # Optimize for training mode
    if training_mode:
        gui = False  # Force headless in training for max performance
        options += ",allowFastDeserialization=1"
        
    # Create the client connection based on mode
    client_id = None
    if gui:
        # Use options string for GUI configuration
        connect_options = options
        if realtime:
            connect_options += ",realtime=1"
        else:
            connect_options += ",realtime=0"
        
        # Create a specific GUI instance
        client_id = p.connect(p.GUI, options=connect_options)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)
        
        # Configure timestep
        p.setTimeStep(1.0 / fps, physicsClientId=client_id)
        p.setRealTimeSimulation(realtime, physicsClientId=client_id)
        p.setPhysicsEngineParameter(enableFileCaching=1, physicsClientId=client_id)
        
        if not realtime:
            # Configure non-blocking visuals
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, physicsClientId=client_id)
    else:
        # Performance optimizations for training
        if training_mode:
            # Maximum performance optimizations for training
            client_id = p.connect(p.DIRECT)
            p.setPhysicsEngineParameter(enableFileCaching=1, physicsClientId=client_id)
            p.setPhysicsEngineParameter(numSolverIterations=4, physicsClientId=client_id)  # Reduce solver iterations
            p.setTimeStep(1.0 / fps, physicsClientId=client_id)
            # Disable expensive computations during training
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)
            p.setAdditionalSearchPath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), physicsClientId=client_id)
        else:
            # Standard headless mode
            client_id = p.connect(p.DIRECT)
            p.setTimeStep(1.0 / fps, physicsClientId=client_id)
            p.setPhysicsEngineParameter(enableFileCaching=1, physicsClientId=client_id)
    
    return client_id

def get_shared_pybullet_client(
    gui: bool = False, 
    realtime: bool = False, 
    fps: int = 240, 
    options: str = "",
    training_mode: bool = False
) -> Any:
    """
    Get a shared pybullet client instance or create a new one if none exists.
    This is useful for sharing a pybullet client between multiple environments.
    
    Args:
        gui: Whether to use GUI (True) or Direct (False, headless)
        realtime: Whether to use realtime simulation
        fps: Frames per second to use for stepping
        options: Options to pass to PyBullet
        training_mode: Whether client is for training (optimized for speed)
        
    Returns:
        PyBullet client
    """
    global _SHARED_PYBULLET_CLIENT, _SHARED_PYBULLET_CLIENT_GUI_STATE
    
    # If there's no client yet, or gui state has changed, create a new one
    if (_SHARED_PYBULLET_CLIENT is None or 
        _SHARED_PYBULLET_CLIENT_GUI_STATE != gui):
        
        # Disconnect any existing connection to avoid leaking resources
        if _SHARED_PYBULLET_CLIENT is not None:
            p.disconnect(_SHARED_PYBULLET_CLIENT)
            
        # Create a new client with the requested parameters
        _SHARED_PYBULLET_CLIENT = get_pybullet_client(
            gui=gui, 
            realtime=realtime, 
            fps=fps, 
            options=options, 
            training_mode=training_mode
        )
        _SHARED_PYBULLET_CLIENT_GUI_STATE = gui
    
    return _SHARED_PYBULLET_CLIENT

def configure_visualization(client_id, clean_viz=True):
    """
    Configure the PyBullet visualization for a clean, zoomed-in view of the robot.
    
    Args:
        client_id: PyBullet physics client ID
        clean_viz: Whether to use clean visualization (True) or default (False)
    """
    try:
        # Check if we're in GUI mode
        if p.getConnectionInfo(client_id)['connectionMethod'] != p.GUI:
            print("Warning: Cannot configure visualization in DIRECT mode")
            return
            
        # Apply clean visualization if requested
        if clean_viz:
            # Disable all GUI panels and HUD elements
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client_id)
            
            # Position the camera to focus on the robot workspace
            target_pos = [0.0, 0.0, 0.3]  # Center of the workspace
            
            # Set camera parameters for a good view of the robot
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,     # Closer view
                cameraYaw=45,           # View angle around z-axis
                cameraPitch=-25,        # View angle from horizontal plane
                cameraTargetPosition=target_pos,
                physicsClientId=client_id
            )
            
            # Enable shadows for better depth perception
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
            
            # Enable rendering
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client_id)
            
            print("Camera configured for clean, zoomed-in view of the robot")
        else:
            # Basic configuration without changing the default view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client_id)
            print("Using default PyBullet visualization")
    except Exception as e:
        print(f"Warning: Could not configure visualization: {e}")

def visualize_target(position, client_id):
    """
    Create a visual marker for the target position.
    
    Args:
        position: [x, y, z] coordinates for the target
        client_id: PyBullet client ID
        
    Returns:
        int: Body ID of the created visual marker
    """
    target_visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 0.7],  # Green with transparency
        physicsClientId=client_id
    )
    
    target_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual_id,
        basePosition=position,
        physicsClientId=client_id
    )
    
    return target_body_id

def visualize_ee_position(ee_position, client_id, color=[0, 1, 0, 0.7], radius=0.01):
    """
    Create a visual marker for the end-effector position to track its trajectory
    
    Args:
        ee_position: [x, y, z] position of the end-effector
        client_id: PyBullet physics client ID
        color: RGBA color for the marker
        radius: Size of the marker
        
    Returns:
        ID of the created visual marker
    """
    # Create a visual sphere at the end-effector position
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id
    )
    
    # Create a body with the visual shape but no mass (visual only)
    marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=ee_position,
        physicsClientId=client_id
    )
    
    return marker_id

def visualize_target_line(start_pos, end_pos, client_id, color=[0.5, 0.5, 1.0, 0.6], width=2.0, lifeTime=0):
    """
    Create a visual line connecting the end effector to the target position
    
    Args:
        start_pos: [x, y, z] start position (end effector)
        end_pos: [x, y, z] end position (target)
        client_id: PyBullet physics client ID
        color: RGBA color for the line
        width: Width of the line
        lifeTime: Time in seconds before the line disappears (0 means persistent)
        
    Returns:
        ID of the created visual line
    """
    # Create a line from start to end position
    line_id = p.addUserDebugLine(
        lineFromXYZ=start_pos,
        lineToXYZ=end_pos,
        lineColorRGB=color[:3],  # RGB part of the color
        lineWidth=width,
        lifeTime=lifeTime,  # Time in seconds before the line disappears
        physicsClientId=client_id
    )
    
    return line_id

def load_workspace_data(filepath=None):
    """
    Load precomputed workspace data from a file, or return None if not available.
    This enables faster initialization of the environment by avoiding recomputation of the reachable workspace.
    
    Args:
        filepath: Path to the workspace data file (defaults to a standard location if None)
        
    Returns:
        Tuple of (positions, joint_configs) if file exists, None otherwise
    """
    import numpy as np
    import os
    
    if filepath is None:
        # Look in standard locations
        possible_paths = [
            "workspace_data.npz",
            "./data/workspace_data.npz",
            "../data/workspace_data.npz",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/workspace_data.npz")
        ]
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
    
    # Check if file exists
    if filepath is not None and os.path.exists(filepath):
        try:
            data = np.load(filepath)
            positions = data['positions']
            joint_configs = data['joint_configs']
            print(f"Loaded workspace data from {filepath}: {len(positions)} positions")
            return positions, joint_configs
        except Exception as e:
            print(f"Error loading workspace data: {e}")
    
    return None

def determine_reachable_workspace(robot, n_samples=1000, visualize=False, client_id=None):
    """
    Sample random joint configurations to determine the robot's reachable workspace.
    
    Args:
        robot: PyBullet robot object with joint limits defined
        n_samples (int): Number of samples to generate
        visualize (bool): Whether to visualize the workspace
        client_id: PyBullet client ID
        
    Returns:
        Tuple[float, Dict]: Maximum reach (float) and workspace bounds (dict)
    """
    # Extract joint information from the robot
    joint_info = []
    for joint_idx in range(p.getNumJoints(robot, physicsClientId=client_id)):
        info = p.getJointInfo(robot, joint_idx, physicsClientId=client_id)
        joint_info.append(info)
    
    # Identify joints with limits
    actuated_joints = []
    joint_limits = []
    
    for i, info in enumerate(joint_info):
        joint_type = info[2]  # Get joint type
        if joint_type == p.JOINT_REVOLUTE:
            lower_limit = info[8]  # Lower joint limit
            upper_limit = info[9]  # Upper joint limit
            
            # Valid limits should be finite
            if np.isfinite(lower_limit) and np.isfinite(upper_limit):
                actuated_joints.append(i)
                joint_limits.append((lower_limit, upper_limit))
    
    # Sample random joint configurations to determine reach
    end_effector_positions = []
    max_reach = 0
    
    for _ in range(n_samples):
        # Generate random joint configuration within limits
        random_joint_positions = []
        for lower, upper in joint_limits:
            random_pos = lower + random.random() * (upper - lower)
            random_joint_positions.append(random_pos)
        
        # Set the robot to this configuration
        for i, joint_idx in enumerate(actuated_joints):
            p.resetJointState(
                robot, 
                joint_idx, 
                random_joint_positions[i],
                physicsClientId=client_id
            )
        
        # Get the end effector position - assuming last link is end effector
        if actuated_joints:
            link_state = p.getLinkState(
                robot, 
                actuated_joints[-1],
                physicsClientId=client_id
            )
            pos = link_state[0]  # Position of the link
            end_effector_positions.append(pos)
            
            # Calculate distance from origin to determine reach
            reach = np.linalg.norm(pos)
            max_reach = max(max_reach, reach)
    
    # Calculate workspace bounds from sampled positions
    if end_effector_positions:
        positions = np.array(end_effector_positions)
        x_min, y_min, z_min = np.min(positions, axis=0)
        x_max, y_max, z_max = np.max(positions, axis=0)
        
        workspace_bounds = {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'z_min': float(z_min),
            'z_max': float(z_max)
        }
    else:
        workspace_bounds = {
            'x_min': 0.0, 'x_max': 0.0,
            'y_min': 0.0, 'y_max': 0.0,
            'z_min': 0.0, 'z_max': 0.0
        }
    
    # Visualize the workspace if requested
    if visualize and client_id is not None:
        for pos in end_effector_positions:
            p.addUserDebugPoints(
                [pos], 
                [[1, 0, 0]], 
                pointSize=2,
                physicsClientId=client_id
            )
    
    return max_reach, workspace_bounds

def adjust_camera_for_robots(client_id, num_robots=1, workspace_size=0.8, grid_layout=False, grid_size=None):
    """
    Adjust the camera view for multi-robot setups
    
    Args:
        client_id: PyBullet physics client ID
        num_robots: Number of robots in the scene
        workspace_size: Size of the workspace to show
        grid_layout: Whether robots are arranged in a grid (True) or line (False)
        grid_size: Number of robots per row in the grid layout
    """
    try:
        # Check if we're in GUI mode
        if p.getConnectionInfo(client_id)['connectionMethod'] != p.GUI:
            return
            
        if grid_layout and grid_size is not None:
            # Grid layout - position camera to see the entire grid
            # The grid is (grid_size x grid_size) with 2.5m spacing
            offset_distance = 2.5  # Should match the value in create_revamped_envs
            
            # Calculate the center of the grid
            center_x = (grid_size - 1) * offset_distance / 2.0
            center_y = (grid_size - 1) * offset_distance / 2.0
            
            # Calculate appropriate camera distance based on grid size
            # We need to see a square with side length = (grid_size-1) * offset_distance
            grid_extent = (grid_size - 1) * offset_distance
            camera_distance = max(3.0, 1.5 * grid_extent)
            
            camera_yaw = 45  # View from an angle
            camera_pitch = -35  # Look down at the grid
            target_pos = [center_x, center_y, 0.3]
        else:
            # Original line layout
            # Adjust based on number of robots
            if num_robots <= 1:
                # Single robot - close-up view
                camera_distance = 1.2
                camera_yaw = 45
                camera_pitch = -30
                target_pos = [0.0, 0.0, 0.3]
            elif num_robots <= 4:
                # 2-4 robots - medium distance view
                # Calculate the center point between robots
                # With 1.5m spacing, robots are at positions 0, 1.5, 3.0, 4.5
                # So for 4 robots, center is at (0 + 4.5) / 2 = 2.25
                center_x = (num_robots - 1) * 1.5 / 2.0
                camera_distance = 1.6 + (num_robots * 0.4)  # Increased distance to see all robots
                camera_yaw = 40
                camera_pitch = -35
                target_pos = [center_x, 0.0, 0.3]
            else:
                # Many robots - overview
                center_x = (num_robots - 1) * 1.5 / 2.0
                camera_distance = 2.0 + (num_robots * 0.5)  # Further increased for many robots
                camera_yaw = 30
                camera_pitch = -40
                target_pos = [center_x, 0.0, 0.2]
        
        # Scale by workspace size
        camera_distance *= (workspace_size / 0.8)
        
        # Apply camera settings
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_pos,
            physicsClientId=client_id
        )
        
        # Configure visualization options
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client_id)
        
    except Exception as e:
        print(f"Warning: Could not adjust camera: {e}")

def load_urdf(urdf_path, client_id=None, fixed_base=False, use_fixed_base=None):
    """
    Load a URDF file into the PyBullet simulation.
    
    Args:
        urdf_path (str): Path to the URDF file
        client_id: PyBullet client ID (optional)
        fixed_base (bool): Whether the base of the robot should be fixed
        use_fixed_base (bool): Alternative parameter name for fixed_base
        
    Returns:
        int: Body ID of the loaded URDF
    """
    # Handle legacy parameter name
    if use_fixed_base is not None:
        fixed_base = use_fixed_base
    
    if client_id is None:
        body_id = p.loadURDF(
            urdf_path,
            useFixedBase=fixed_base
        )
    else:
        body_id = p.loadURDF(
            urdf_path,
            useFixedBase=fixed_base,
            physicsClientId=client_id
        )
    return body_id

def load_robot(urdf_path, client_id=None, gui=False):
    """
    Load a robot from a URDF file and set initial configuration.
    
    Args:
        urdf_path (str): Path to the URDF file
        client_id: PyBullet client ID (None will use current active client)
        gui (bool): Whether visualization is enabled (previously 'render')
        
    Returns:
        int: Body ID of the loaded robot
    """
    # Load URDF file with fixed base
    robot_id = load_urdf(urdf_path, client_id=client_id, fixed_base=True)
    
    # Configure visualization if GUI is enabled
    if gui and client_id is not None:
        configure_visualization(client_id)
    
    return robot_id

def get_directml_settings_from_env():
    """
    Get DirectML settings from environment variables.
    
    Returns:
        Tuple[bool, Dict]: Whether to use DirectML and settings dict
    """
    use_directml = os.environ.get('FANUC_DIRECTML', '0').lower() in ('1', 'true', 'yes')
    use_directml = use_directml or os.environ.get('USE_DIRECTML', '0').lower() in ('1', 'true', 'yes')
    
    # Get additional settings from environment
    settings = {}
    
    return use_directml, settings

# Test the utilities if this file is run directly
if __name__ == "__main__":
    print("Testing PyBullet utilities...")
    
    # Test client creation
    try:
        client_id = get_pybullet_client(gui=True)
        print(f"Created PyBullet client: {client_id}")
        
        # Test basic functionality
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        print("Set gravity successfully")
        
        # Add ground plane
        plane_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=client_id
        )
        print(f"Added ground plane: {plane_id}")
        
        # Test visualization
        configure_visualization(client_id)
        print("Configured visualization")
        
        # Test sphere creation
        sphere_id = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.1,
            rgbaColor=[1, 0, 0, 0.7],
            physicsClientId=client_id
        )
        print(f"Created sphere: {sphere_id}")
        
        # Let user see the results briefly
        print("Tests completed successfully. Close the window to exit.")
        
        # Keep the window open
        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Error during tests: {e}")
    
    finally:
        # Clean up
        if 'client_id' in locals():
            p.disconnect(physicsClientId=client_id)
            print("Disconnected from PyBullet") 