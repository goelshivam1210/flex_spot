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

def project_points_onto_plane(points, normal_vector, origin):
    """
    Project 3D points onto a plane defined by:
        - normal_vector: unit normal to the plane
        - origin: a point on the plane

    Args:
        points: (N, 3) array of 3D points
        normal_vector: (3,) plane normal (does not need to be normalized)
        origin: (3,) point on the plane

    Returns:
        (N, 3) array of projected 3D points
    """
    normal = normal_vector / np.linalg.norm(normal_vector)
    projected_points = []
    for p in points:
        v = p - origin
        distance = np.dot(v, normal)
        projected_point = p - distance * normal
        projected_points.append(projected_point)
    return np.array(projected_points)


class InteractivePerception:
    """
    Interactive perception module that handles:
    1. Generating wiggle movements
    2. Analyzing trajectory to estimate joint type and parameters
    3. Constructing state vectors for policy input
    """
    
    def __init__(self, movement_distance=0.15):
        """
        Args:
            movement_distance: Distance to move in each direction (meters)
        """
        self.movement_distance = movement_distance
        self.max_revolute_radius = 2.0
        self.small_motion_threshold = 0.10
        self.revolute_better_factor = 0.5

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
        d = self.movement_distance
        
        # Cardinal directions (x/y plane)
        directions = [
            np.array([ d,  0, 0]),   # +X (forward)
            np.array([-d,  0, 0]),   # -X (backward)
            np.array([ 0,  d, 0]),   # +Y (left)
            np.array([ 0, -d, 0]),   # -Y (right)
            # # Diagonals to excite more directions in the plane
            # np.array([ 0.7*d,  0.7*d, 0]),   # NE
            # np.array([-0.7*d, -0.7*d, 0]),   # SW
            # np.array([ 0.7*d, -0.7*d, 0]),   # SE
            # np.array([-0.7*d,  0.7*d, 0]),   # NW
        ]

        num_substeps = 5

        for direction in directions:
            for step in range(1, num_substeps+1):
                # add segments
                increment = direction * (step/num_substeps)
                target = start_position + increment
                positions.append(target.copy())
            # return to this direction at the end of this directional movement
            positions.append(start_position.copy())

            # # Move to direction
            # target = start_position + direction
            # positions.append(target.copy())
            #
            # # Return to center
            # positions.append(start_position.copy())
        
        return positions
    
    # def prismatic_error_analysis(self, trajectory):
    #     """Calculate prismatic joint error and axis."""
    #     centroid = np.mean(trajectory, axis=0)
    #     X = trajectory - centroid
    #     _, _, Vt = np.linalg.svd(X)
    #     line_direction = Vt[0]
    #     projections = np.dot(X, line_direction[:, np.newaxis]) * line_direction
    #     residuals = np.linalg.norm(X - projections, axis=1)
    #     ss_residuals = np.sum(residuals**2) / len(trajectory)
    #     return ss_residuals, line_direction

    def prismatic_error_analysis(self, trajectory):
        """
        Fit a line to the 3D trajectory and compute mean squared perpendicular distance.

        Args:
            trajectory: (N, 3) array of 3D EE positions.

        Returns:
            (mse, axis):
                mse: scalar mean squared distance from points to best-fit line.
                axis: (3,) unit vector along the prismatic joint axis.
        """
        trajectory = np.asarray(trajectory)
        centroid = np.mean(trajectory, axis=0)
        X = trajectory - centroid

        # SVD ≈ PCA: first right-singular vector is principal direction
        _, _, Vt = np.linalg.svd(X)
        axis = Vt[0]
        axis /= np.linalg.norm(axis)

        # Project points onto axis and compute perpendicular residuals
        proj_scalars = X @ axis
        projections = np.outer(proj_scalars, axis)
        residuals = np.linalg.norm(X - projections, axis=1)
        mse = np.mean(residuals**2)

        return mse, axis

    def fit_model_2d(self, points_2d):
        """
        Fit a line or circle to 2D points and return a model + confidence.

        Returns:
            (model, confidence) where model is:
                ("LINEAR", {"direction": (2,), "point": (2,)})
                ("CIRCULAR", {"center": (2,), "radius": float})
                ("UNDEFINED", None)
        """
        pts = np.asarray(points_2d, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_2d must be (N, 2).")
        x_arr = pts[:, 0]
        y_arr = pts[:, 1]
        n = len(pts)

        span = np.hypot(x_arr[-1] - x_arr[0], y_arr[-1] - y_arr[0])
        if n < 5 or span < 0.01:
            return ("UNDEFINED", None), 0.0

        x_mean, y_mean = np.mean(x_arr), np.mean(y_arr)
        _, _, vh = np.linalg.svd(np.vstack([x_arr - x_mean, y_arr - y_mean]).T)
        dir_x, dir_y = vh[0]
        normal_x, normal_y = -dir_y, dir_x
        line_residuals = (x_arr - x_mean) * normal_x + (y_arr - y_mean) * normal_y
        rss_line = np.sum(line_residuals ** 2)

        def circle_residuals(params, x_in, y_in):
            xc, yc, r = params
            return np.sqrt((x_in - xc) ** 2 + (y_in - yc) ** 2) - r

        initial_r = span / 2 + 0.1
        try:
            res = least_squares(circle_residuals, [x_mean, y_mean, initial_r], args=(x_arr, y_arr))
            rss_circle = np.sum(res.fun ** 2)
            circle_params = res.x
        except Exception:
            res = None
            rss_circle = np.inf
            circle_params = None

        aic_line = n * np.log(rss_line / n + 1e-9) + 2 * 2
        aic_circle = n * np.log(rss_circle / n + 1e-9) + 2 * 3
        is_circular = aic_circle < (aic_line - 2.0)

        if not is_circular:
            rmse = np.sqrt(rss_line / n)
            score_fit = np.clip(1.0 - (rmse / 0.02), 0.0, 1.0)
            score_span = np.clip(span / 0.10, 0.0, 1.0)
            confidence = score_fit * score_span
            model = ("LINEAR", {"direction": np.array([dir_x, dir_y]), "point": np.array([x_mean, y_mean])})
            return model, confidence

        if circle_params is None:
            return ("UNDEFINED", None), 0.0

        xc, yc, r_est = circle_params
        mse = rss_circle / max(n - 3, 1)
        try:
            j = res.jac
            cov = np.linalg.pinv(j.T @ j) * mse
            sigma_r = np.sqrt(cov[2, 2])
            score_math = np.clip(1.0 - (sigma_r / r_est), 0.0, 1.0)
        except Exception:
            score_math = 0.0

        angles = np.arctan2(y_arr - yc, x_arr - xc)
        angles = np.unwrap(angles)
        angle_span = np.abs(angles[-1] - angles[0])
        score_span = np.clip(angle_span / 0.26, 0.0, 1.0)
        confidence = score_math * score_span
        model = ("CIRCULAR", {"center": np.array([xc, yc]), "radius": r_est})
        return model, confidence

    def fit_circle_in_plane(self, points_3d, axis):
        """
        Fit a circle to 3D points that lie (approximately) in a plane
        orthogonal to a known axis.

        Args:
            points_3d: (N, 3) array of CENTERED 3D points (mean should be ~zero).
            axis: (3,) unit vector normal to the motion plane.

        Returns:
            center3d: (3,) circle center in CENTERED coordinates.
            radius: scalar circle radius.
        """
        # DON'T compute origin - points are already centered!
        # origin = np.mean(points_3d, axis=0)  # ← REMOVE THIS

        # Project points onto plane orthogonal to axis through origin (0,0,0)
        origin = np.zeros(3)  # Use origin since points_3d is already centered
        projected = project_points_onto_plane(points_3d, axis, origin)

        # Embed plane in 2D via PCA
        pca2 = PCA(n_components=2).fit(projected)
        points_2d = pca2.transform(projected)

        # Circle residuals in 2D: distance to center - mean radius
        def circle_residuals(c, pts):
            d = np.linalg.norm(pts - c, axis=1)
            return d - d.mean()

        c0 = points_2d.mean(axis=0)
        res = least_squares(circle_residuals, c0, args=(points_2d,))
        center_2d = res.x

        # Compute radius as mean distance to fitted center
        radii = np.linalg.norm(points_2d - center_2d, axis=1)
        radius = radii.mean()

        # Map 2D center back to 3D plane coordinates
        center_plane = pca2.inverse_transform(center_2d)

        # center3d is in CENTERED coordinates (relative to trajectory mean)
        return center_plane, radius

    def revolute_error_analysis(self, trajectory):
        """
        Estimate revolute joint parameters from a 3D trajectory.

        Pipeline (matches paper conceptually):
            1. Center data at mean.
            2. 3D PCA: axis = smallest-variance component (plane normal).
            3. Fit a circle in the plane orthogonal to axis.
            4. Clamp radius to max_revolute_radius.
            5. Compute MSE in 3D wrt this circle (constant distance to center).

        Args:
            trajectory: (N, 3) array of 3D EE positions.

        Returns:
            (mse, center_world, radius, axis):
                mse: scalar mean squared (‖p - center‖ - radius)^2
                center_world: (3,) circle center in world frame.
                radius: scalar radius (clamped).
                axis: (3,) unit vector for rotation axis direction.
        """
        trajectory = np.asarray(trajectory)

        # Center trajectory at mean (world <-> centered conversion)
        mean_world = np.mean(trajectory, axis=0)
        X = trajectory - mean_world

        # 3D PCA to get candidate axis: smallest variance direction
        pca3 = PCA(n_components=3).fit(X)
        axis = pca3.components_[-1]
        axis /= np.linalg.norm(axis)

        # Fit circle in plane orthogonal to axis (in centered coordinates)
        center_centered, radius = self.fit_circle_in_plane(X, axis)

        # Clamp radius to physical prior
        radius = float(min(radius, self.max_revolute_radius))

        # Convert center back to world coordinates
        center_world = center_centered + mean_world

        # Compute MSE in 3D: distance to center - radius
        d = np.linalg.norm(trajectory - center_world, axis=1)
        mse = np.mean((d - radius) ** 2)

        return mse, center_world, radius, axis
    # def revolute_error_analysis(self, trajectory):
    #     """Calculate revolute joint error and parameters."""
    #     mean = np.mean(trajectory, axis=0)
    #     X = trajectory - mean
    #     pca = PCA(n_components=3)
    #     pca.fit(X)
    #
    #     normal_vector = pca.components_[-1]
    #
    #     # Simple circle fitting
    #     def residuals(params, points):
    #         center = np.array([params[0], params[1], params[2]])
    #         radius = params[3]
    #         distance = np.linalg.norm(points - center, axis=1) - radius
    #         return distance
    #
    #     # Initial estimate
    #     initial_center = mean
    #     initial_radius = np.std(np.linalg.norm(X, axis=1))
    #     initial_guess = np.concatenate([initial_center, [initial_radius]])
    #
    #     result = least_squares(residuals, initial_guess, args=(trajectory,))
    #     center = result.x[:-1]
    #     radius = result.x[-1]
    #
    #     # Calculate MSE
    #     distances = np.linalg.norm(trajectory - center, axis=1)
    #     mse = np.mean((distances - radius) ** 2)
    #
    #     return mse, center, radius, normal_vector

    def analyze_trajectory_and_estimate_joint(self, trajectory, plot_fits=True, plot_path=None):
        """
        Analyze a *dense* EE trajectory and estimate joint type & parameters.

        Args:
            trajectory: (N, 3) array of 3D EE positions collected while
                        executing a wiggle motion.
            plot_fits: whether to save a plot with both line and circle fits.
            plot_path: optional filepath for the plot output.

        Returns:
            (joint_type, joint_params) where:
                joint_type: "prismatic" or "revolute"
                joint_params:
                    If prismatic:
                        {
                            "axis": (3,) unit vector,
                            "error": float
                        }
                    If revolute:
                        {
                            "axis": (3,) unit vector,
                            "center": (3,),
                            "radius": float,
                            "error": float
                        }
        """
        trajectory = np.asarray(trajectory)

        # Basic sanity check: if too few points, just bail out as prismatic
        if trajectory.shape[0] < 5:
            raise ValueError("Trajectory too short; need at least 5 points.")

        # Compute prismatic and revolute fits
        pris_error, pris_axis = self.prismatic_error_analysis(trajectory)
        rev_error, rev_center, rev_radius, rev_axis = self.revolute_error_analysis(trajectory)

        print(f"[InteractivePerception] Prismatic error: {pris_error:.6e}")
        model_2d, conf_2d = self.fit_model_2d(trajectory[:, :2])
        print(f"[InteractivePerception] 2D model: {model_2d[0]}, confidence: {conf_2d:.3f}")

        z_range = np.ptp(trajectory[:, 2]) if trajectory.shape[1] >= 3 else 0.0
        if z_range <= 0.5 and conf_2d >= 0.5 and model_2d[0] != "UNDEFINED":
            if model_2d[0] == "CIRCULAR":
                center_xy = model_2d[1]["center"]
                radius = float(model_2d[1]["radius"])
                center_world = np.array([center_xy[0], center_xy[1], float(np.mean(trajectory[:, 2]))])
                d = np.linalg.norm(trajectory[:, :2] - center_xy, axis=1)
                mse = np.mean((d - radius) ** 2)
                self.joint_type = "revolute"
                self.joint_params = {
                    "axis": np.array([0.0, 0.0, 1.0]),
                    "center": center_world,
                    "radius": radius,
                    "error": mse,
                }
                print("[InteractivePerception] Using 2D circular fit -> REVOLUTE")
                return self.joint_type, self.joint_params

            direction_xy = model_2d[1]["direction"]
            axis = np.array([direction_xy[0], direction_xy[1], 0.0])
            axis = axis / np.linalg.norm(axis)
            self.joint_type = "prismatic"
            self.joint_params = {
                "axis": axis,
                "error": pris_error,
            }
            print("[InteractivePerception] Using 2D linear fit -> PRISMATIC")
            return self.joint_type, self.joint_params

        print(f"[InteractivePerception] Revolute  error: {rev_error:.6e}")
        print(f"[InteractivePerception] Revolute radius: {rev_radius:.3f} m")

        if plot_fits:
            try:
                self.plot_trajectory_with_fits(
                    trajectory,
                    show=False,
                    save_path=plot_path,
                )
            except ImportError as exc:
                print(f"[InteractivePerception] Plot skipped: {exc}")

        # Heuristic 1: if radius is essentially at the max limit, treat as prismatic
        if rev_radius >= self.max_revolute_radius - 1e-6:
            print(f"[InteractivePerception] Radius {rev_radius:.2f} m ≥ max ({self.max_revolute_radius}) -> PRISMATIC")
            self.joint_type = "prismatic"
            self.joint_params = {
                "axis": pris_axis,
                "error": pris_error,
            }
            return self.joint_type, self.joint_params

        # Heuristic 2: if total motion is too small, default to prismatic
        traj_range = np.ptp(trajectory, axis=0)  # peak-to-peak per coordinate
        max_range = float(np.max(traj_range))  # max extent in any axis
        print(f"[InteractivePerception] Trajectory max range: {max_range:.3f} m")
        if max_range < self.small_motion_threshold:
            print(f"[InteractivePerception] Small motion (< {self.small_motion_threshold} m) -> PRISMATIC")
            self.joint_type = "prismatic"
            self.joint_params = {
                "axis": pris_axis,
                "error": pris_error,
            }
            return self.joint_type, self.joint_params

        # Heuristic 3: revolute must be significantly better than prismatic
        if rev_error < self.revolute_better_factor * pris_error:
            print("[InteractivePerception] Revolute significantly better -> REVOLUTE")
            self.joint_type = "revolute"
            self.joint_params = {
                "axis": rev_axis,
                "center": rev_center,
                "radius": rev_radius,
                "error": rev_error,
            }
        else:
            print("[InteractivePerception] Revolute not clearly better -> PRISMATIC")
            self.joint_type = "prismatic"
            self.joint_params = {
                "axis": pris_axis,
                "error": pris_error,
            }

        return self.joint_type, self.joint_params

    def plot_trajectory_with_fits(
        self,
        trajectory,
        num_circle_pts=200,
        show=False,
        save_path=None,
        ax=None,
    ):
        """
        Plot trajectory points with both prismatic (line) and revolute (circle) fits.

        Args:
            trajectory: (N, 3) array of 3D EE positions.
            num_circle_pts: number of samples used to draw the fitted circle.
            show: whether to call plt.show().
            save_path: optional filepath to save the figure; defaults to a local PNG.
            ax: optional matplotlib 3D axis to draw into.

        Returns:
            (fig, ax): matplotlib figure and axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting.") from exc

        trajectory = np.asarray(trajectory)
        if trajectory.shape[0] < 2:
            raise ValueError("Need at least 2 points to plot.")

        # Prismatic fit: line through centroid along principal axis
        pris_error, pris_axis = self.prismatic_error_analysis(trajectory)
        centroid = np.mean(trajectory, axis=0)
        proj_scalars = (trajectory - centroid) @ pris_axis
        line_min = centroid + pris_axis * proj_scalars.min()
        line_max = centroid + pris_axis * proj_scalars.max()
        line_pts = np.vstack([line_min, line_max])

        # 2D circle fit (preferred over revolute fit for plotting)
        model_2d, conf_2d = self.fit_model_2d(trajectory[:, :2])
        circle_pts = None
        circle_label = None
        if model_2d[0] == "CIRCULAR":
            center_xy = model_2d[1]["center"]
            radius = model_2d[1]["radius"]
            z_plane = float(np.mean(trajectory[:, 2]))
            t = np.linspace(0.0, 2.0 * np.pi, num_circle_pts)
            circle_xy = center_xy + radius * np.column_stack([np.cos(t), np.sin(t)])
            circle_pts = np.column_stack([circle_xy, np.full(num_circle_pts, z_plane)])
            circle_label = f"circle fit 2d (conf={conf_2d:.2f})"

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], s=12, label="trajectory")
        ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], "r-", label=f"line fit (mse={pris_error:.2e})")
        if circle_pts is not None:
            ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], "g-", label=circle_label)

        # Match axis scales for a sensible 3D view
        mins = trajectory.min(axis=0)
        maxs = trajectory.max(axis=0)
        center = (mins + maxs) / 2.0
        span = max(maxs - mins)
        if span > 0:
            half = span / 2.0
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        if save_path is None:
            save_path = "interactive_perception_fits.png"
        fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax

    # def analyze_trajectory_and_estimate_joint(self, trajectory):
    #     """
    #     Analyze trajectory to estimate joint type and parameters.
    #
    #     Args:
    #         trajectory: Array of positions [N x 3]
    #
    #     Returns:
    #         tuple: (joint_type, joint_params)
    #     """
    #     # Calculate errors for both joint types
    #     prismatic_error, prismatic_axis = self.prismatic_error_analysis(trajectory)
    #     revolute_error, revolute_center, revolute_radius, revolute_axis = self.revolute_error_analysis(trajectory)
    #
    #     print(f"Prismatic error: {prismatic_error:.6f}")
    #     print(f"Revolute error: {revolute_error:.6f}")
    #
    #     # add override based on the estimated max radius
    #     # if revolute_radius > 1.0:
    #     #     print (f"Revolute radius {revolute_radius:.2f}m too large -- interpreting it as prismatic")
    #     #     self.joint_type = "prismatic"
    #     #     self.joint_params = {"axis": prismatic_axis, "error": prismatic_error}
    #     #     return self.joint_type, self.joint_params
    #
    #     # Select joint type based on lower error
    #     # if prismatic_error < revolute_error:
    #     if revolute_error > 0.00000:
    #         self.joint_type = "prismatic"
    #         self.joint_params = {"axis": prismatic_axis, "error": prismatic_error}
    #         print("Estimated joint type: PRISMATIC")
    #     else:
    #         self.joint_type = "revolute"
    #         self.joint_params = {
    #             "center": revolute_center,
    #             "radius": revolute_radius,
    #             "axis": revolute_axis,
    #             "error": revolute_error
    #         }
    #         print("Estimated joint type: REVOLUTE")
    #
    #     return self.joint_type, self.joint_params
    #
    # def construct_state_vector(self, current_position, initial_position):
    #     if self.joint_params is None:
    #         raise ValueError("Must analyze trajectory first!")
    #
    #     joint_axis = self.joint_params["axis"]
    #     joint_axis = joint_axis / np.linalg.norm(joint_axis)
    #
    #     if self.joint_type == "prismatic":
    #         # Prismatic: [hp, Δpt]
    #         displacement = current_position - initial_position
    #         state = np.concatenate([joint_axis, displacement])
    #
    #     elif self.joint_type == "revolute":
    #         # Revolute: [hr, vt]
    #         # vt = (pt - pr) - [(pt - pr)·hr]hr
    #         joint_center = self.joint_params["center"]  # pr
    #         vector_to_point = current_position - joint_center  # pt - pr
    #
    #         # Project onto plane perpendicular to rotation axis
    #         projection_on_axis = np.dot(vector_to_point, joint_axis) * joint_axis
    #         vt = vector_to_point - projection_on_axis
    #
    #         state = np.concatenate([joint_axis, vt])
    #
    #     else:
    #         raise ValueError(f"Unknown joint type: {self.joint_type}")
    #
    #     return state.astype(np.float32)
    
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
            offset_depth = box_dimensions["depth"] / 2   # Edge grasp is on the side, not front/back

            # Transform arc points to vision frame
            # Account for robot's current orientation in vision frame
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            
            # Rotation matrix to transform from local to vision frame
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw,  cos_yaw]
            ])
            point = np.array([offset_depth, offset_width])  # Local coordinates
            transformed_offset = rotation_matrix @ point
      
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
            
            # box_center = gripper_pos + np.array([offset_x, offset_y, offset_z])
            box_center = gripper_pos + np.array([transformed_offset[0], transformed_offset[1], 0.0])
            
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
        
        # if box_dimensions is None:
        #     box_dimensions = {"width": 0.4, "depth": 0.4, "height": 0.2}

        # box_center  = self.estimate_box_center_from_grasp(
        #     current_pos, grasp_strategy, box_dimensions, current_yaw)
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
