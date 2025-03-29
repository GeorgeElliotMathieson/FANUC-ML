#!/usr/bin/env python3
# pybullet_utils.py
# Utility functions for PyBullet visualizations and robot simulation

import pybullet as p  # type: ignore
import numpy as np
import time
import os
import threading
from typing import List, Tuple, Dict, Optional, Union

# Global variables to track PyBullet connections
_GUI_CONNECTION_ESTABLISHED: bool = False
_GUI_CLIENT_ID: Optional[int] = None

# Global variable to track client ID for shared client and lock for thread safety
_SHARED_CLIENT_ID: Optional[int] = None
_SHARED_CLIENT_LOCK = threading.Lock()

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

def get_pybullet_client(render=False):
    """
    Get a new PyBullet client instance.
    
    Args:
        render (bool): Whether to use GUI mode for visualization
        
    Returns:
        int: Client ID
    """
    if render:
        client_id = p.connect(p.GUI)
    else:
        client_id = p.connect(p.DIRECT)
    
    # Configure basic simulation properties
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)
    
    return client_id

def get_shared_pybullet_client(render=False):
    """
    Get or create a shared PyBullet client.
    This ensures we have a single PyBullet instance across the entire application.
    
    Args:
        render (bool): Whether to use GUI mode for visualization
        
    Returns:
        int: Shared client ID
    """
    global _SHARED_CLIENT_ID
    
    # Use lock to ensure thread safety when accessing shared client
    with _SHARED_CLIENT_LOCK:
        # Check if we already have a shared client
        if _SHARED_CLIENT_ID is not None:
            # Check if it's still connected
            try:
                p.getConnectionInfo(_SHARED_CLIENT_ID)
                return _SHARED_CLIENT_ID
            except Exception as e:
                # Connection was lost, reset the client ID
                print(f"Connection to shared client lost: {e}")
                _SHARED_CLIENT_ID = None
        
        # Create a new client
        try:
            if render:
                _SHARED_CLIENT_ID = p.connect(p.GUI)
            else:
                _SHARED_CLIENT_ID = p.connect(p.DIRECT)
            
            # Configure basic simulation properties
            p.setGravity(0, 0, -9.81, physicsClientId=_SHARED_CLIENT_ID)
            p.setRealTimeSimulation(0, physicsClientId=_SHARED_CLIENT_ID)
            
            return _SHARED_CLIENT_ID
        except Exception as e:
            print(f"Error creating shared PyBullet client: {e}")
            # In case of failure, return a new non-shared client as fallback
            try:
                return get_pybullet_client(render)
            except:
                # Last resort - if everything fails
                raise RuntimeError(f"Failed to create any PyBullet client: {e}")

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

def determine_reachable_workspace(robot, n_samples=1000, visualize=False):
    """
    Determine the reachable workspace of the robot by sampling random joint configurations
    and recording the resulting end-effector positions.
    
    Args:
        robot: The robot environment instance
        n_samples: Number of samples to generate
        visualize: Whether to visualize the workspace during sampling
        
    Returns:
        Tuple of (max_reach, workspace_bounds) where:
        - max_reach is the maximum distance the robot can reach
        - workspace_bounds is a dictionary with 'x', 'y', 'z' keys containing [min, max] bounds
    """
    import numpy as np
    import time
    
    print(f"Determining reachable workspace with {n_samples} samples...")
    start_time = time.time()
    
    positions = []
    joint_configs = []
    markers = []
    
    # Create visualization markers if requested
    if visualize and robot.render_mode:
        # Set up visualization
        configure_visualization(robot.client, clean_viz=True)
    
    # Sample random joint configurations
    for i in range(n_samples):
        # Generate random joint configuration within limits
        config = []
        for j in range(robot.dof):
            if j in robot.joint_limits:
                limit_low, limit_high = robot.joint_limits[j]
                config.append(np.random.uniform(limit_low, limit_high))
            else:
                config.append(0.0)
        
        # Apply configuration
        robot.step(config)
        
        # Get end-effector position
        state = robot._get_state()
        ee_position = state[-7:-4]  # Last 7 elements are position and orientation
        
        # Store result
        positions.append(ee_position)
        joint_configs.append(config)
        
        # Visualize if requested
        if visualize and robot.render_mode and i % 10 == 0:
            # Add marker for this position
            marker_id = visualize_ee_position(ee_position, robot.client, color=[0, 0.3, 1, 0.4], radius=0.005)
            markers.append(marker_id)
            
            # Update progress
            if i % 100 == 0:
                print(f"  Sampled {i}/{n_samples} configurations...")
                time.sleep(0.01)  # Small delay for visualization
    
    # Convert to numpy arrays
    positions = np.array(positions)
    joint_configs = np.array(joint_configs)
    
    # Reset robot
    robot.reset()
    
    # Calculate workspace bounds
    x_min, y_min, z_min = np.min(positions, axis=0)
    x_max, y_max, z_max = np.max(positions, axis=0)
    
    # Create bounds dictionary
    workspace_bounds = {
        'x': [x_min, x_max],
        'y': [y_min, y_max],
        'z': [z_min, z_max]
    }
    
    # Calculate max reach (distance from origin to furthest point)
    max_distances = np.linalg.norm(positions, axis=1)
    max_reach = np.max(max_distances)
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f"Workspace determination complete: {len(positions)} positions in {elapsed_time:.2f} seconds")
    
    return max_reach, workspace_bounds

def adjust_camera_for_robots(client_id, num_robots=1, workspace_size=0.8):
    """
    Adjust the camera view for multi-robot setups
    
    Args:
        client_id: PyBullet physics client ID
        num_robots: Number of robots in the scene
        workspace_size: Size of the workspace to show
    """
    try:
        # Check if we're in GUI mode
        if p.getConnectionInfo(client_id)['connectionMethod'] != p.GUI:
            return
            
        # Adjust based on number of robots
        if num_robots <= 1:
            # Single robot - close-up view
            camera_distance = 1.2
            camera_yaw = 45
            camera_pitch = -30
            target_pos = [0.0, 0.0, 0.3]
        elif num_robots <= 4:
            # 2-4 robots - medium distance view
            camera_distance = 1.6
            camera_yaw = 40
            camera_pitch = -35
            target_pos = [0.0, 0.0, 0.3]
        else:
            # Many robots - overview
            camera_distance = 2.0 + (num_robots * 0.1)
            camera_yaw = 30
            camera_pitch = -40
            target_pos = [0.0, 0.0, 0.2]
        
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

# Test the utilities if this file is run directly
if __name__ == "__main__":
    # Create a client
    client_id = get_pybullet_client(render=True)
    
    # Configure visualization
    configure_visualization(client_id, clean_viz=True)
    
    # Add a target marker
    target_id = visualize_target([0.3, 0.2, 0.4], client_id)
    
    # Add end-effector markers
    markers = []
    for i in range(10):
        pos = [
            0.2 + (i * 0.02),
            0.1 + (i * 0.02),
            0.3 + (i * 0.01)
        ]
        marker_id = visualize_ee_position(pos, client_id)
        markers.append(marker_id)
        
        # Draw line to target
        line_id = visualize_target_line(pos, [0.3, 0.2, 0.4], client_id)
        
        time.sleep(0.1)
    
    # Wait for a bit
    time.sleep(3)
    
    # Test camera adjustment for multiple robots
    adjust_camera_for_robots(client_id, num_robots=4)
    
    # Wait a bit longer
    time.sleep(2)
    
    # Disconnect
    p.disconnect(client_id) 