# pybullet_utils.py
# Utility functions for PyBullet

import pybullet as p

# Global variable to track if a GUI connection has been established
_GUI_CONNECTION_ESTABLISHED = False
_GUI_CLIENT_ID = None

def get_pybullet_client(render=False):
    """
    Get a PyBullet client, ensuring only one GUI connection is created.
    
    Args:
        render: Whether to use GUI mode (True) or DIRECT mode (False)
        
    Returns:
        PyBullet client ID
    """
    global _GUI_CONNECTION_ESTABLISHED, _GUI_CLIENT_ID
    
    # If render is requested but we already have a GUI connection,
    # return the existing GUI client ID
    if render and _GUI_CONNECTION_ESTABLISHED and _GUI_CLIENT_ID is not None:
        return _GUI_CLIENT_ID
    
    # If render is requested and we don't have a GUI connection yet,
    # create a new GUI connection
    if render and not _GUI_CONNECTION_ESTABLISHED:
        try:
            client_id = p.connect(p.GUI)
            _GUI_CONNECTION_ESTABLISHED = True
            _GUI_CLIENT_ID = client_id
            print(f"Created GUI connection with client ID: {client_id}")
            return client_id
        except Exception as e:
            print(f"Warning: Could not create GUI connection: {e}")
            print("Falling back to DIRECT mode")
            return p.connect(p.DIRECT)
    
    # If render is not requested, create a DIRECT connection
    return p.connect(p.DIRECT)

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

def visualize_target(target_position, client_id, color=[1.0, 0.0, 0.0, 0.8], radius=0.04):
    """
    Create a visual marker for the target position in PyBullet
    
    Args:
        target_position: [x, y, z] position of the target
        client_id: PyBullet physics client ID
        color: RGBA color for the marker (default: red)
        radius: Size of the marker
        
    Returns:
        ID of the created visual marker
    """
    # Create a visual sphere at the target position
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,  # Increased size for better visibility
        rgbaColor=color,
        physicsClientId=client_id
    )
    
    # Create a body with the visual shape but no mass (visual only)
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        basePosition=target_position,
        physicsClientId=client_id
    )
    
    # No text label for cleaner visualization
    
    return target_id

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