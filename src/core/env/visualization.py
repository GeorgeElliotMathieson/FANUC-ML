"""
Visualization utilities for robot environment.
"""

import numpy as np
import pybullet as p  # type: ignore

def visualize_target(position, client_id):
    """
    Create a visual marker for the target position.
    
    Args:
        position: 3D position vector [x, y, z]
        client_id: PyBullet client ID
        
    Returns:
        Target body ID
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

def visualize_trajectory(points, client_id, lifetime=0, color=None):
    """
    Visualize a trajectory as a line in PyBullet.
    
    Args:
        points: List of 3D points
        client_id: PyBullet client ID
        lifetime: Lifetime of the visualization in seconds (0 for persistent)
        color: RGBA color (default: blue with transparency)
        
    Returns:
        Line ID
    """
    if color is None:
        color = [0, 0, 1, 0.7]  # Blue with transparency
        
    if not points or len(points) < 2:
        return None
        
    # Create line segments between consecutive points
    lines = []
    for i in range(len(points) - 1):
        lines.append([points[i], points[i+1]])
        
    # Create line visualization
    line_ids = []
    for start, end in lines:
        line_id = p.addUserDebugLine(
            lineFromXYZ=start,
            lineToXYZ=end,
            lineColorRGB=color[:3],  # RGB part of RGBA
            lineWidth=2.0,
            lifeTime=lifetime,
            physicsClientId=client_id
        )
        line_ids.append(line_id)
        
    return line_ids

def visualize_workspace(bounds, client_id, alpha=0.2):
    """
    Visualize the workspace boundaries.
    
    Args:
        bounds: Dictionary with 'x', 'y', 'z' keys, each containing [min, max]
        client_id: PyBullet client ID
        alpha: Transparency (0-1)
        
    Returns:
        List of visualization IDs
    """
    # Extract bounds
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    # Define corners of the workspace cube
    corners = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ]
    
    # Define edges
    edges = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Connecting edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Create lines for each edge
    vis_ids = []
    for i, j in edges:
        line_id = p.addUserDebugLine(
            lineFromXYZ=corners[i],
            lineToXYZ=corners[j],
            lineColorRGB=[0, 0.5, 0.5],  # Teal
            lineWidth=2.0,
            lifeTime=0,  # Persistent
            physicsClientId=client_id
        )
        vis_ids.append(line_id)
        
    return vis_ids

def remove_visualization(vis_ids, client_id):
    """
    Remove visualization objects.
    
    Args:
        vis_ids: Single ID or list of visualization IDs
        client_id: PyBullet client ID
    """
    if isinstance(vis_ids, list):
        for vis_id in vis_ids:
            try:
                p.removeUserDebugItem(vis_id, physicsClientId=client_id)
            except Exception as e:
                # Some visualization objects might be bodies, not debug items
                try:
                    p.removeBody(vis_id, physicsClientId=client_id)
                except Exception as e2:
                    if hasattr(p, 'getConnectionInfo') and p.getConnectionInfo(client_id)['connectionMethod'] == p.GUI:
                        print(f"Warning: Could not remove visualization ID {vis_id}: {e2}")
                    pass
    else:
        # Single ID
        try:
            p.removeUserDebugItem(vis_ids, physicsClientId=client_id)
        except Exception as e:
            # Might be a body, not a debug item
            try:
                p.removeBody(vis_ids, physicsClientId=client_id)
            except Exception as e2:
                if hasattr(p, 'getConnectionInfo') and p.getConnectionInfo(client_id)['connectionMethod'] == p.GUI:
                    print(f"Warning: Could not remove visualization ID {vis_ids}: {e2}")
                pass 