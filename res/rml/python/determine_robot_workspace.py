#!/usr/bin/env python3
# determine_robot_workspace.py
# Script to determine the workspace of the FANUC robot by sampling random joint configurations

import os
import numpy as np
import pybullet as p
import pybullet_data
import argparse
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parse command line arguments
parser = argparse.ArgumentParser(description='Determine the workspace of the FANUC robot')
parser.add_argument('--samples', type=int, default=10000, help='Number of random configurations to sample')
parser.add_argument('--visualize', action='store_true', help='Visualize the workspace')
parser.add_argument('--output', type=str, default='robot_workspace.json', help='Output file for workspace data')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

class FANUCRobotEnv:
    def __init__(self, render=True, verbose=False):
        # Store verbose flag
        self.verbose = verbose
        
        # Store render mode
        self.render_mode = render
        
        # Connect to the physics server
        if render:
            self.client = p.connect(p.GUI)
            # Configure visualization
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.0, 0.0, 0.3],
                physicsClientId=self.client
            )
        else:
            self.client = p.connect(p.DIRECT)
            
        if self.verbose:
            print(f"Connected to PyBullet physics server with client ID: {self.client}")
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Robot parameters from the documentation
        self.dof = 6  # 6 degrees of freedom
        self.max_force = 100  # Maximum force for joint motors
        self.position_gain = 0.3
        self.velocity_gain = 1.0
        
        # Load the robot URDF
        self.robot_id = self._load_robot()
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        
        # Joint limits from manual
        self.joint_limits = {
            0: [-720, 720],  # J1 axis - physical limit (multiple rotations allowed)
            1: [-360, 360],  # J2 axis - physical limit
            2: [-360, 360],  # J3 axis - physical limit
            3: [-720, 720],  # J4 axis - physical limit (multiple rotations allowed)
            4: [-360, 360],  # J5 axis - physical limit
            5: [-1080, 1080]  # J6 axis - physical limit (multiple rotations allowed)
        }
        
        # Convert to radians
        for joint, limits in self.joint_limits.items():
            self.joint_limits[joint] = [np.deg2rad(limits[0]), np.deg2rad(limits[1])]
            
        # Initial configuration
        self.reset()
    
    def _load_robot(self):
        # Load the URDF for the FANUC LR Mate 200iC
        
        # Check for the URDF file in various possible locations
        possible_paths = [
            "fanuc_lrmate_200ic.urdf",                  # Current directory
            "res/fanuc_lrmate_200ic.urdf",              # res directory
            "../res/fanuc_lrmate_200ic.urdf",           # One level up
            "../../res/fanuc_lrmate_200ic.urdf",        # Two levels up
            os.path.join(os.path.dirname(__file__), "../../res/fanuc_lrmate_200ic.urdf")  # Relative to this file
        ]
        
        # Try each path
        for urdf_path in possible_paths:
            if os.path.exists(urdf_path):
                if self.verbose:
                    print(f"Loading FANUC LR Mate 200iC URDF from: {urdf_path}")
                return p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
        
        # If we couldn't find the URDF, print a warning and fall back to a simple robot
        print("WARNING: Could not find FANUC LR Mate 200iC URDF file. Falling back to default robot.")
        print("Current working directory:", os.getcwd())
        print("Searched paths:", possible_paths)
        
        # Fallback to a simple robot for testing
        return p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)
    
    def reset(self):
        # Reset to home position
        home_position = [0, 0, 0, 0, 0, 0]  # All joints at 0 position
        for i, pos in enumerate(home_position):
            p.resetJointState(self.robot_id, i, pos)
        
        # Get current state
        state = self._get_state()
        return state
    
    def _get_state(self):
        # Get joint states
        joint_states = []
        for i in range(self.dof):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])  # Joint position
            joint_states.append(state[1])  # Joint velocity
        
        # Get end-effector position and orientation
        ee_link_state = p.getLinkState(self.robot_id, self.dof-1)
        ee_position = ee_link_state[0]
        ee_orientation = ee_link_state[1]
        
        # Combine for full state
        state = np.array(joint_states + list(ee_position) + list(ee_orientation))
        return state
    
    def close(self):
        p.disconnect(self.client)

def determine_workspace(robot_env, num_samples=10000, verbose=False):
    """
    Determine the workspace of the robot by sampling random joint configurations
    and recording the resulting end-effector positions.
    
    Args:
        robot_env: The robot environment
        num_samples: Number of random configurations to sample
        verbose: Whether to print verbose output
        
    Returns:
        dict: Workspace information including:
            - ee_positions: List of end-effector positions
            - max_reach: Maximum reach from home position
            - min_reach: Minimum reach from home position (inner boundary)
            - workspace_bounds: Dictionary with min/max values for each dimension
            - convex_hull: Convex hull of the workspace (if scipy is available)
            - joint_configurations: List of joint configurations that produced the positions
    """
    # Get home position (when all joints are at 0)
    robot_env.reset()
    state = robot_env._get_state()
    home_position = state[12:15]  # End-effector position
    
    if verbose:
        print(f"Home position: {home_position}")
    
    # Sample random joint configurations and record end-effector positions
    ee_positions = []
    joint_configurations = []
    distances = []
    
    if verbose:
        print(f"Sampling {num_samples} random configurations to determine workspace...")
        start_time = time.time()
    
    for i in range(num_samples):
        # Generate random joint positions within limits
        joint_positions = []
        for j in range(robot_env.dof):
            if j in robot_env.joint_limits:
                limit_low, limit_high = robot_env.joint_limits[j]
                # Add some margin to avoid edge cases
                margin = (limit_high - limit_low) * 0.05  # 5% margin
                pos = np.random.uniform(limit_low + margin, limit_high - margin)
                joint_positions.append(pos)
            else:
                joint_positions.append(0)
        
        # Set joint positions
        for j, pos in enumerate(joint_positions):
            p.resetJointState(robot_env.robot_id, j, pos, physicsClientId=robot_env.client)
        
        # Step simulation to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=robot_env.client)
        
        # Get end-effector position
        ee_link_state = p.getLinkState(robot_env.robot_id, robot_env.dof-1, physicsClientId=robot_env.client)
        ee_position = ee_link_state[0]
        
        # Calculate distance from home position
        distance = np.linalg.norm(np.array(ee_position) - home_position)
        
        # Store results
        ee_positions.append(ee_position)
        joint_configurations.append(joint_positions)
        distances.append(distance)
        
        # Print progress
        if verbose and (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{num_samples} configurations ({(i+1)/num_samples*100:.1f}%) - {elapsed:.1f}s elapsed")
    
    # Reset to home position
    robot_env.reset()
    
    # Convert to numpy arrays for easier processing
    ee_positions = np.array(ee_positions)
    joint_configurations = np.array(joint_configurations)
    distances = np.array(distances)
    
    # Calculate maximum reach (outer boundary)
    max_reach = np.max(distances)
    
    # Calculate minimum reach (inner boundary)
    # We'll use the 5th percentile to avoid outliers
    min_reach = np.percentile(distances, 5)
    
    # Calculate workspace bounds
    x_min, y_min, z_min = np.min(ee_positions, axis=0)
    x_max, y_max, z_max = np.max(ee_positions, axis=0)
    
    workspace_bounds = {
        'x': [float(x_min), float(x_max)],
        'y': [float(y_min), float(y_max)],
        'z': [float(z_min), float(z_max)]
    }
    
    # Try to compute convex hull if scipy is available
    convex_hull = None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(ee_positions)
        convex_hull = {
            'vertices': ee_positions[hull.vertices].tolist(),
            'simplices': hull.simplices.tolist()
        }
    except ImportError:
        if verbose:
            print("scipy not available, skipping convex hull computation")
    
    # Create workspace information dictionary
    workspace_info = {
        'home_position': home_position.tolist(),
        'max_reach': float(max_reach),
        'min_reach': float(min_reach),  # Add minimum reach
        'workspace_bounds': workspace_bounds,
        'num_samples': num_samples,
        'convex_hull': convex_hull,
        'ee_positions': ee_positions.tolist(),
        'joint_configurations': joint_configurations.tolist(),
        'distances': distances.tolist()  # Add distances for better analysis
    }
    
    if verbose:
        print(f"\nWorkspace determination complete in {time.time() - start_time:.1f}s")
        print(f"Home position: {home_position}")
        print(f"Maximum reach from home position: {max_reach:.3f}m")
        print(f"Minimum reach from home position: {min_reach:.3f}m")
        print(f"Workspace bounds:")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
    
    return workspace_info

def visualize_workspace(workspace_info, show_plot=True, save_path=None):
    """
    Visualize the robot's workspace.
    
    Args:
        workspace_info: Workspace information dictionary
        show_plot: Whether to show the plot
        save_path: Path to save the plot, or None to not save
    """
    # Extract data
    ee_positions = np.array(workspace_info['ee_positions'])
    home_position = np.array(workspace_info['home_position'])
    max_reach = workspace_info['max_reach']
    min_reach = workspace_info.get('min_reach', 0.0)  # Get min_reach or default to 0
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate distances from home position
    distances = np.linalg.norm(ee_positions - home_position, axis=1)
    normalized_distances = distances / max_reach
    
    # Use a scatter plot with color mapping
    scatter = ax.scatter(
        ee_positions[:, 0], 
        ee_positions[:, 1], 
        ee_positions[:, 2],
        c=normalized_distances,
        cmap='viridis',
        alpha=0.5,
        s=2  # Small point size for better visualization
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Distance from Home Position')
    
    # Plot home position
    ax.scatter([home_position[0]], [home_position[1]], [home_position[2]], 
               color='red', s=100, marker='*', label='Home Position')
    
    # Plot workspace bounds
    bounds = workspace_info['workspace_bounds']
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    # Create a wireframe box for the bounds
    x_corners = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
    y_corners = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
    z_corners = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
    
    # Plot the corners
    ax.scatter(x_corners, y_corners, z_corners, color='black', s=50, alpha=0.5)
    
    # Connect the corners with lines to form a box
    for i in range(4):
        ax.plot([x_corners[i], x_corners[i+4]], 
                [y_corners[i], y_corners[i+4]], 
                [z_corners[i], z_corners[i+4]], 'k-', alpha=0.3)
    
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([x_corners[i], x_corners[j]], 
                [y_corners[i], y_corners[j]], 
                [z_corners[i], z_corners[j]], 'k-', alpha=0.3)
        ax.plot([x_corners[i+4], x_corners[j+4]], 
                [y_corners[i+4], y_corners[j+4]], 
                [z_corners[i+4], z_corners[j+4]], 'k-', alpha=0.3)
    
    # Draw spheres for the inner and outer boundaries
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    
    # Outer boundary sphere (max reach)
    x_outer = max_reach * np.outer(np.cos(u), np.sin(v)) + home_position[0]
    y_outer = max_reach * np.outer(np.sin(u), np.sin(v)) + home_position[1]
    z_outer = max_reach * np.outer(np.ones(np.size(u)), np.cos(v)) + home_position[2]
    ax.plot_surface(x_outer, y_outer, z_outer, color='b', alpha=0.1)
    
    # Inner boundary sphere (min reach)
    if min_reach > 0:
        x_inner = min_reach * np.outer(np.cos(u), np.sin(v)) + home_position[0]
        y_inner = min_reach * np.outer(np.sin(u), np.sin(v)) + home_position[1]
        z_inner = min_reach * np.outer(np.ones(np.size(u)), np.cos(v)) + home_position[2]
        ax.plot_surface(x_inner, y_inner, z_inner, color='r', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    title = f'FANUC Robot Workspace\n{len(ee_positions)} sampled positions'
    title += f'\nMax Reach: {max_reach:.3f}m, Min Reach: {min_reach:.3f}m'
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Workspace visualization saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    # Create robot environment
    robot_env = FANUCRobotEnv(render=args.visualize, verbose=args.verbose)
    
    try:
        # Determine workspace
        workspace_info = determine_workspace(
            robot_env, 
            num_samples=args.samples, 
            verbose=args.verbose
        )
        
        # Save workspace information to file
        output_path = args.output
        with open(output_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(workspace_info, f, indent=2)
        
        print(f"Workspace information saved to {output_path}")
        
        # Visualize workspace if requested
        if args.visualize:
            # Save visualization to a PNG file
            vis_path = os.path.splitext(output_path)[0] + '_visualization.png'
            visualize_workspace(workspace_info, show_plot=True, save_path=vis_path)
    
    finally:
        # Close the environment
        robot_env.close()

if __name__ == "__main__":
    main() 