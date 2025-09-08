"""
interactive_perception.py

Handles interactive perception tasks for the Spot robot, including:
1. Generating wiggle movements for exploration
2. Analyzing trajectories to estimate joint types and parameters
3. Constructing state vectors for policy input

Author: Shivam Goel
Date: July 2025
"""
import math

import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import least_squares


class InteractivePerception:
    """
    Interactive perception module that handles:
    1. Generating wiggle movements
    2. Analyzing trajectory to estimate joint type and parameters
    3. Constructing state vectors for policy input
    """
    
    def __init__(self, movement_distance=0.05):
        """
        Args:
            movement_distance: Distance to move in each direction (meters)
        """
        self.movement_distance = movement_distance
        self.joint_params = None
        self.joint_type = None
    
    def generate_wiggle_positions(self, start_position):
        """
        Generate target positions for 4-directional wiggling exploration.
        
        Args:
            start_position: Initial gripper position [x, y, z]
            
        Returns:
            list: Target positions for wiggling sequence
        """
        positions = [start_position.copy()]  # Start position
        
        # 4 directions: forward, backward, left, right
        directions = [
            np.array([self.movement_distance, 0, 0]),   # Forward
            np.array([-self.movement_distance, 0, 0]),  # Backward
            np.array([0, -self.movement_distance, 0]),   # Right  
            np.array([0, self.movement_distance, 0]),   # Left
        ]
        
        for direction in directions:
            # Move to direction
            target = start_position + direction
            positions.append(target.copy())
            
            # Return to center
            positions.append(start_position.copy())
        
        return positions
    
    def prismatic_error_analysis(self, trajectory):
        """Calculate prismatic joint error and axis."""
        centroid = np.mean(trajectory, axis=0)
        X = trajectory - centroid
        _, _, Vt = np.linalg.svd(X)
        line_direction = Vt[0]
        projections = np.dot(X, line_direction[:, np.newaxis]) * line_direction
        residuals = np.linalg.norm(X - projections, axis=1) 
        ss_residuals = np.sum(residuals**2) / len(trajectory)
        return ss_residuals, line_direction
    
    def revolute_error_analysis(self, trajectory):
        """Calculate revolute joint error and parameters."""
        mean = np.mean(trajectory, axis=0) 
        X = trajectory - mean
        pca = PCA(n_components=3)
        pca.fit(X)

        normal_vector = pca.components_[-1]
        
        # Simple circle fitting
        def residuals(params, points):
            center = np.array([params[0], params[1], params[2]])
            radius = params[3]
            distance = np.linalg.norm(points - center, axis=1) - radius
            return distance
        
        # Initial estimate
        initial_center = mean
        initial_radius = np.std(np.linalg.norm(X, axis=1))
        initial_guess = np.concatenate([initial_center, [initial_radius]])
        
        result = least_squares(residuals, initial_guess, args=(trajectory,))
        center = result.x[:-1]
        radius = result.x[-1]
        
        # Calculate MSE
        distances = np.linalg.norm(trajectory - center, axis=1)
        mse = np.mean((distances - radius) ** 2)
        
        return mse, center, radius, normal_vector
    
    def analyze_trajectory_and_estimate_joint(self, trajectory):
        """
        Analyze trajectory to estimate joint type and parameters.
        
        Args:
            trajectory: Array of positions [N x 3]
            
        Returns:
            tuple: (joint_type, joint_params)
        """
        # Calculate errors for both joint types
        prismatic_error, prismatic_axis = self.prismatic_error_analysis(trajectory)
        revolute_error, revolute_center, revolute_radius, revolute_axis = self.revolute_error_analysis(trajectory)
        
        print(f"Prismatic error: {prismatic_error:.6f}")
        print(f"Revolute error: {revolute_error:.6f}")
        
        # Select joint type based on lower error
        if prismatic_error < revolute_error:
            self.joint_type = "prismatic"
            self.joint_params = {"axis": prismatic_axis, "error": prismatic_error}
            print("Estimated joint type: PRISMATIC")
        else:
            self.joint_type = "revolute" 
            self.joint_params = {
                "center": revolute_center,
                "radius": revolute_radius, 
                "axis": revolute_axis,
                "error": revolute_error
            }
            print("Estimated joint type: REVOLUTE")
        
        return self.joint_type, self.joint_params
    
    def construct_state_vector(self, current_position, initial_position):
        """
        Construct state vector for policy input.
        
        Args:
            current_position: Current gripper position [x, y, z] 
            initial_position: Initial grasp position [x, y, z]
            
        Returns:
            np.array: State vector [joint_axis_x, joint_axis_y, joint_axis_z, 
                                   displacement_x, displacement_y, displacement_z]
        """
        if self.joint_params is None:
            raise ValueError("Must analyze trajectory first!")
        
        # Calculate displacement from initial position
        displacement = current_position - initial_position
        
        # Get normalized joint axis
        joint_axis = self.joint_params["axis"]
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        
        # Construct state: [joint_axis (3D), displacement (3D)]
        state = np.concatenate([joint_axis, displacement])
        
        return state.astype(np.float32)
    
    def construct_path_following_state(self, current_pos, path_points, current_yaw, closest_idx):
        """
        Construct 8D state vector for path-following policy.
        
        Args:
            current_pos: Current gripper/object position [x, y, z]
            path_points: Array of path waypoints [N x 3]
            current_yaw: Current robot/object orientation (radians)
            closest_idx: Index of closest point on path
            
        Returns:
            np.array: 8D state vector [lateral_error, longitudinal_error, orientation_error,
                                    progress, deviation, speed_along_path, box_forward_x, box_forward_y]
        """
        
        # Find closest point on path
        closest_point = path_points[closest_idx]
        
        # Calculate path tangent direction
        if closest_idx < len(path_points) - 1:
            next_point = path_points[closest_idx + 1]
            path_tangent = next_point - closest_point
            tangent_norm = np.linalg.norm(path_tangent)
            if tangent_norm > 1e-8:
                path_tangent = path_tangent / tangent_norm
            else:
                path_tangent = np.array([1.0, 0.0, 0.0])  # Default forward
        else:
            # At end of path, use direction from previous point
            if closest_idx > 0:
                prev_point = path_points[closest_idx - 1]
                path_tangent = closest_point - prev_point
                tangent_norm = np.linalg.norm(path_tangent)
                if tangent_norm > 1e-8:
                    path_tangent = path_tangent / tangent_norm
                else:
                    path_tangent = np.array([1.0, 0.0, 0.0])
            else:
                path_tangent = np.array([1.0, 0.0, 0.0])  # Default forward
        
        # Calculate path normal (perpendicular to tangent)
        path_normal = np.array([-path_tangent[1], path_tangent[0], 0.0])
        
        # Calculate position error relative to path
        position_error = current_pos - closest_point
        
        # Project error onto path-relative coordinates
        lateral_error = np.dot(position_error, path_normal)
        longitudinal_error = np.dot(position_error, path_tangent)
        
        # Calculate orientation error
        desired_yaw = math.atan2(path_tangent[1], path_tangent[0])
        orientation_error = math.atan2(
            math.sin(current_yaw - desired_yaw), 
            math.cos(current_yaw - desired_yaw)
        )
        
        # Discretize orientation error (matching simulation)
        angle_bin_size = 10.0  # degrees
        num_bins = int(360 / angle_bin_size)
        bin_index = int(((orientation_error + math.pi) * 180/math.pi) / angle_bin_size) % num_bins
        discretized_orientation_error = (bin_index * angle_bin_size * math.pi/180) - math.pi
        
        # Calculate progress along path (0 to 1)
        if len(path_points) > 1:
            progress = closest_idx / (len(path_points) - 1)
        else:
            progress = 0.0
        
        # Calculate deviation from path
        deviation = np.linalg.norm(position_error)
        
        # Calculate speed along path (simplified - could use position history)
        speed_along_path = 0.0  # For now, set to 0 (could be enhanced with velocity tracking)
        
        # Robot/box orientation unit vectors
        box_forward_x = math.cos(current_yaw)
        box_forward_y = math.sin(current_yaw)
        
        # Construct 8D state vector (matching simulation environment)
        state = np.array([
            lateral_error,
            longitudinal_error,
            discretized_orientation_error,
            progress,
            deviation,
            speed_along_path,
            box_forward_x,
            box_forward_y
        ], dtype=np.float32)
        
        return state

    def update_closest_path_index(self, current_pos, path_points, last_idx):
        """
        Update the closest path point index efficiently.
        
        Args:
            current_pos: Current position [x, y, z]
            path_points: Array of path waypoints [N x 3]
            last_idx: Last known closest index
            
        Returns:
            int: Updated closest path index
        """
        # Search in a window around the last known closest point
        search_start = max(0, last_idx - 5)
        search_end = min(len(path_points), last_idx + 20)
        
        if search_start >= search_end:
            return last_idx
        
        search_points = path_points[search_start:search_end]
        distances = np.linalg.norm(search_points - current_pos, axis=1)
        
        closest_in_window = np.argmin(distances)
        return search_start + closest_in_window

    def generate_straight_line_path(self, start_pos, end_pos, num_points=50):
        """
        Generate a straight line path between two points.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_points: Number of waypoints along the path
            
        Returns:
            np.array: Path points [num_points x 3]
        """
        path_points = []
        for i in range(num_points):
            t = i / max(1, num_points - 1)  # Avoid division by zero
            point = start_pos + t * (end_pos - start_pos)
            path_points.append(point)
        
        return np.array(path_points) 