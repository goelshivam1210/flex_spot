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
    
    def estimate_box_center_from_grasp(self, gripper_pos, grasp_strategy, box_dimensions, current_yaw):
        """
        Estimate box center position from gripper position and grasp type.
        
        Box coordinate system:
        - Height: vertical dimension (how tall)
        - Width: horizontal dimension (left-to-right when facing box)  
        - Depth: front-to-back dimension (how deep)
        
        Handle specifications:
        - Located on front face of box
        - 48cm from top, 30cm from each side edge
        - Centered horizontally and vertically accessible
        
        Edge grasp specifications:
        - Usually left edge for single robot
        - Around 45cm mark from top/bottom (roughly middle height)
        
        Args:
            gripper_pos: Current gripper position [x, y, z]
            grasp_strategy: "edge_grasp", "handle_grasp", "dual_edge_grasp", "dual_handle_grasp"
            box_dimensions: {"width": 0.6, "depth": 0.4, "height": 0.3} (meters)
            current_yaw: Robot orientation (radians)
            
        Returns:
            np.array: Estimated box center position [x, y, z]
        """
        
        if grasp_strategy == "handle_grasp":
            # Handle is on front face, centered horizontally
            # Gripper is at handle position on front surface
            # Box center is half-depth behind the front face
            
            # Calculate offset from handle to box center
            offset_distance = box_dimensions["depth"] / 2  # Half box depth inward
            
            # Offset in robot's forward direction (behind the front face)
            offset_x = offset_distance * math.cos(current_yaw)
            offset_y = offset_distance * math.sin(current_yaw)
            
            # Handle is 48cm from top, so adjust Z to box center
            # If handle is 0.48m from top, and box height is H, 
            # then handle is at (H - 0.48) from bottom
            # Box center is at H/2 from bottom
            # So offset = H/2 - (H - 0.48) = 0.48 - H/2
            handle_height_from_bottom = box_dimensions["height"] - 0.48  # 48cm from top
            box_center_height = box_dimensions["height"] / 2
            offset_z = box_center_height - handle_height_from_bottom
            
            box_center = gripper_pos + np.array([offset_x, offset_y, offset_z])
            
        elif grasp_strategy == "edge_grasp":
            # Single robot grasping left edge at ~45cm mark
            # Gripper is on the left side face of the box
            
            # Box center is half-width to the right of left edge
            # and half-depth inward from the side
            
            # Calculate offset from left edge to center
            # Assuming robot approaches from the left side
            offset_width = box_dimensions["width"] / 2  # Half width rightward
            offset_depth = 0  # Edge grasp is on the side, not front/back
            
            # Calculate offsets in world coordinates
            # Robot's right direction is perpendicular to forward direction
            right_x = -math.sin(current_yaw)  # Perpendicular to forward
            right_y = math.cos(current_yaw)
            
            offset_x = offset_width * right_x
            offset_y = offset_width * right_y
            
            # Height adjustment: if grasping at 45cm mark, adjust to center
            # Assuming 45cm is from bottom
            grasp_height_from_bottom = 0.45  # 45cm mark
            box_center_height = box_dimensions["height"] / 2
            offset_z = box_center_height - grasp_height_from_bottom
            
            box_center = gripper_pos + np.array([offset_x, offset_y, offset_z])
            
        elif grasp_strategy == "dual_edge_grasp":
            # Two robots grasping opposite edges
            # Assume gripper is on edge, box center is at geometric center
            
            # For dual robot, assume robots are on opposite sides
            # Box center is simply at the midpoint between the two grasps
            # Since we only have one gripper position, estimate based on edge
            
            # Similar to edge_grasp but might need robot ID to determine which side
            # For now, assume similar to single edge grasp
            offset_width = box_dimensions["width"] / 2
            
            right_x = -math.sin(current_yaw)
            right_y = math.cos(current_yaw)
            
            offset_x = offset_width * right_x  
            offset_y = offset_width * right_y
            
            grasp_height_from_bottom = 0.45
            box_center_height = box_dimensions["height"] / 2
            offset_z = box_center_height - grasp_height_from_bottom
            
            box_center = gripper_pos + np.array([offset_x, offset_y, offset_z])
            
        elif grasp_strategy == "dual_handle_grasp":
            # Two robots grasping handles (if box has multiple handles)
            # Similar to single handle grasp
            offset_distance = box_dimensions["depth"] / 2
            
            offset_x = offset_distance * math.cos(current_yaw)
            offset_y = offset_distance * math.sin(current_yaw)
            
            handle_height_from_bottom = box_dimensions["height"] - 0.48
            box_center_height = box_dimensions["height"] / 2
            offset_z = box_center_height - handle_height_from_bottom
            
            box_center = gripper_pos + np.array([offset_x, offset_y, offset_z])
            
        else:
            # Fallback: assume gripper position is box center
            box_center = gripper_pos
        
        return box_center
    
    def construct_path_following_state(self, current_pos, path_points, current_yaw, closest_idx, 
                                       grasp_strategy="edge_grasp", box_dimensions = None, velocity_2d=None):
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
        
        if box_dimensions is None:
            box_dimensions = {"width": 0.4, "depth": 0.4, "height": 0.2}

        box_center  = self.estimate_box_center_from_grasp(
            current_pos, grasp_strategy, box_dimensions, current_yaw)
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
        position_error = box_center - closest_point
        
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
        
        # Calculate speed along path using provided velocity
        if velocity_2d is not None:
            speed_along_path = np.dot(velocity_2d, path_tangent[:2])  # Project velocity onto path tangent
        else:
            speed_along_path = 0.0  # Default for first step or when velocity unavailable
        
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
    
    def generate_arc_path(self, center, radius, start_angle, end_angle, num_points=50):
        """Generate curved arc path matching simulation training."""
        points = []
        for theta in np.linspace(start_angle, end_angle, num_points):
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            points.append([x, y])
        return np.array(points)
    
    def decompose_wrench_to_contact_forces(self, wrench, box_dimensions=None, max_force=400.0):
        """
        Decompose a centralized wrench into two contact forces for dual robot manipulation.
        
        Args:
            wrench: [Fx, Fy, τz] - desired wrench on box center
            box_dimensions: Box dimensions dict (uses width for contact spacing)
            max_force: Maximum force per contact point
            
        Returns:
            np.array: Contact forces [2x3] for [left_robot, right_robot]
        """
        if box_dimensions is None:
            box_dimensions = {"width": 0.4, "depth": 0.4, "height": 0.2}
        
        Fx, Fy, tau_z = wrench[0], wrench[1], wrench[2]
        
        # Define contact points (left and right sides of box)
        offset_x = -0.2  # Behind box center
        offset_y = box_dimensions["width"] / 2  # Half box width
        contact_points = np.array([
            [offset_x, +offset_y, 0.0],  # Left robot contact
            [offset_x, -offset_y, 0.0]   # Right robot contact  
        ])
        
        # Cap the torque based on maximum achievable with contact geometry
        torque_cap = 2 * max_force * abs(offset_y)
        tau_z = np.clip(tau_z, -torque_cap, torque_cap)
        
        # Build equilibrium matrix A for force balance
        # A * [f1x, f1y, f2x, f2y]^T = [Fx, Fy, τz]^T
        A = np.zeros((3, 4))
        
        # Force balance equations: f1x + f2x = Fx, f1y + f2y = Fy
        A[0, 0] = 1.0  # f1x coefficient
        A[0, 2] = 1.0  # f2x coefficient
        A[1, 1] = 1.0  # f1y coefficient  
        A[1, 3] = 1.0  # f2y coefficient
        
        # Moment balance: τz = r1x*f1y - r1y*f1x + r2x*f2y - r2y*f2x
        r1 = contact_points[0]  # [offset_x, +offset_y, 0]
        r2 = contact_points[1]  # [offset_x, -offset_y, 0]
        
        A[2, 0] = -r1[1]  # -r1y * f1x
        A[2, 1] = +r1[0]  # +r1x * f1y
        A[2, 2] = -r2[1]  # -r2y * f2x  
        A[2, 3] = +r2[0]  # +r2x * f2y
        
        # Solve using pseudo-inverse
        wrench_vector = np.array([Fx, Fy, tau_z])
        forces_flat = np.linalg.pinv(A) @ wrench_vector
        
        # Reshape to contact forces [2x2] then pad to [2x3]
        contact_forces_2d = forces_flat.reshape(2, 2)
        contact_forces = np.zeros((2, 3))
        contact_forces[:, :2] = contact_forces_2d  # x,y forces
        contact_forces[:, 2] = 0  # z forces are zero
        
        # Apply force limits per contact
        for i in range(2):
            force_mag = np.linalg.norm(contact_forces[i])
            if force_mag > max_force:
                contact_forces[i] *= max_force / force_mag
        
        return contact_forces