# path_following_env.py

import gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from gym import spaces
from scipy.interpolate import CubicSpline

class SimplePathFollowingEnv(gym.Env):
    """
    Path Following Environment for Multi-Robot Sim-to-Real Transfer
    
    A hollow plywood box (~65kg, 1x1x1.5m) follows a curved path using force vectors
    applied in the box reference frame. The state representation is path-relative 
    and includes box orientation vectors to enable multi-robot coordination.
    
    State: [lateral_error, longitudinal_error, orientation_error, progress, 
            deviation, speed_along_path, box_forward_x, box_forward_y] (8D)
    Action: [force_x, force_y, torque_z] (3D)
    """
    def __init__(
        self, 
        gui=False, 
        max_force=400.0,       # Realistic for 65kg hollow plywood box
        max_torque=50.0,
        friction=0.4,          # Plywood on floor
        linear_damping=0.05,   # Low damping for easier movement
        angular_damping=0.1,   # Low damping for easier rotation
        goal_thresh=0.2, 
        max_steps=500,
        goal_reward=100,
        seed=None,
        segment_length=0.3,
        test_full_arc=False,
        arc_radius=1.5, arc_start=-np.pi/3, arc_end=np.pi/3
    ):
        super(SimplePathFollowingEnv, self).__init__()
        
        # Environment parameters
        self.gui = gui
        self.max_force = max_force
        self.max_torque = max_torque
        self.friction = friction
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.goal_thresh = goal_thresh
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.seed = seed

        # Simulation parameters
        self.dt = 1.0 / 240.0
        self.sim_steps = 6
        self.angle_bin_size = 10.0  # degrees for discretization
        self.num_bins = int(360/self.angle_bin_size)
        
        # Define observation space - 8 element enhanced state
        self.observation_space = spaces.Box(
            low=np.array([
                -np.inf, -np.inf,  # lateral_error, longitudinal_error
                -np.pi,            # orientation_error (discretized)
                0.0,               # progress [0,1]
                0.0,               # deviation [0,inf]
                -np.inf,           # speed_along_path
                -1.0, -1.0         # box_forward_x, box_forward_y (unit vector)
            ]),
            high=np.array([
                np.inf, np.inf,    # lateral_error, longitudinal_error
                np.pi,             # orientation_error (discretized)
                1.0,               # progress [0,1]
                np.inf,            # deviation [0,inf]
                np.inf,            # speed_along_path
                1.0, 1.0           # box_forward_x, box_forward_y (unit vector)
            ]), 
            dtype=np.float32
        )
        
        # Define action space - [force_x, force_y, torque_z]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Connect to PyBullet
        self._connect()
        
        # Generate arc path points
        # self.path_points = self._generate_arc_path()
        self.test_full_arc = test_full_arc
        self.full_path = self._generate_arc_path(arc_radius, arc_start, arc_end)
        
        # slice a random training segment
        self.segment_length = segment_length
        self._make_training_segment()

        # Previous position for speed calculation
        self.prev_position = None
        self.prev_time = None
        
        # Initialize state
        self.steps = 0
        self.reset()
    
    def _connect(self):
        """Connect to PyBullet physics server"""
        if self.gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.8)

    def _generate_random_path(self,
                              num_waypoints=5,
                              span=3.0,
                              noise=0.2,
                              num_points=200):
        """
        Build a random smooth path by:
        sampling `num_waypoints` in a box [-span,span]^2
        sorting them by x (or along some heuristic)
        fitting a cubic spline through them, then
        resampling `num_points` points along the spline
        """
        # sample waypoints
        pts = np.random.uniform(-span, span, size=(num_waypoints, 2))
        # sort by x so it “flows” forward
        pts = pts[np.argsort(pts[:,0])]
        # add little jitter so it isn’t exactly monotonic
        pts[:,1] += np.random.randn(num_waypoints)*noise

        # spline in x→y
        xs, ys = pts[:,0], pts[:,1]
        cs = CubicSpline(xs, ys, bc_type='natural')

        # sample uniformly in x
        x_min, x_max = xs.min(), xs.max()
        xs_eval = np.linspace(x_min, x_max, num_points)
        ys_eval = cs(xs_eval)

        return np.stack([xs_eval, ys_eval], axis=1)
    
    def _generate_arc_path(self, radius=1.5, start_angle=-np.pi/3, end_angle=np.pi/3, num_points=50):
        """Generate points along an arc path"""
        radius = 1.5
        start_angle = -np.pi/3
        end_angle = np.pi/3
        num_points = 50
        
        points = []
        for theta in np.linspace(start_angle, end_angle, num_points):
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append(np.array([x, y]))
        
        return np.array(points)
    
    def _make_training_segment(self):
        """
        Randomly slice a contiguous sub‐segment of length approximately self.segment_length
        from the full path self.full_path. If segment_length is None or exceeds total path
        length, we just use the entire path.
        """
        full = self.full_path
        n = len(full)
        if self.segment_length is None:
            self.path_points = full.copy()
            return

        # Compute cumulative arc‐length along the full path
        disp = np.linalg.norm(np.diff(full, axis=0), axis=1)  # (n-1,)
        cumlen = np.concatenate(([0.0], np.cumsum(disp)))     # (n,)

        total_length = cumlen[-1]
        seg_len = self.segment_length

        # If requested segment is too long, just use full path
        if seg_len >= total_length or n < 2:
            self.path_points = full.copy()
            return

        # Randomly choose a starting arc‐length so that segment fits
        start_dist = np.random.rand() * (total_length - seg_len)
        end_dist = start_dist + seg_len

        # Find the indices that bracket these distances
        i0 = np.searchsorted(cumlen, start_dist, side='right') - 1
        i1 = np.searchsorted(cumlen, end_dist, side='right')
        i0 = max(0, i0)
        i1 = min(n - 1, i1)

        # Slice the points
        self.path_points = full[i0 : i1 + 1].copy()
    
    def reset(self):
        """Reset the environment to its initial state"""
        # if in “full‐arc test” mode, rebuild full_path and disable slicing
        if self.test_full_arc:
            self.full_path = self._generate_arc_path(self.arc_radius, self.arc_start, self.arc_end)
            self.segment_length = None
        self._make_training_segment()
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create a hollow plywood box at the start of the path
        start_pos = [self.path_points[0][0], self.path_points[0][1], 0.1]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        
        # Using a default cube URDF with scaling to represent hollow plywood box
        self.box_id = p.loadURDF("cube.urdf", start_pos, start_ori, globalScaling=0.4)
        
        # Set realistic physics properties for hollow plywood box (65kg)
        p.changeDynamics(
            self.box_id, -1, 
            mass=65.0,  # Hollow plywood box mass
            lateralFriction=self.friction, 
            angularDamping=self.angular_damping,  # Lower for easier rotation
            linearDamping=self.linear_damping,    # Lower for easier movement
            rollingFriction=0.01,  # Much lower rolling friction
            spinningFriction=0.01, # Much lower spinning friction
            restitution=0.2        # Slight bounce for plywood
        )
        
        # Set initial orientation to align with path direction
        if len(self.path_points) > 1:
            # Calculate tangent direction
            tangent = self.path_points[1] - self.path_points[0]
            angle = np.arctan2(tangent[1], tangent[0])
            
            # Set orientation
            quat = p.getQuaternionFromEuler([0, 0, angle])
            p.resetBasePositionAndOrientation(self.box_id, start_pos, quat)
            
            # Ensure zero initial velocity
            p.resetBaseVelocity(self.box_id, [0, 0, 0], [0, 0, 0])
        
        # Reset step counter and tracking variables
        self.steps = 0
        self.prev_position = np.array(start_pos[:2])
        self.prev_time = 0.0
        
        # Draw path for visualization in GUI mode
        if self.gui:
            self._draw_path()
        
        # Let the box settle briefly
        for _ in range(5):
            p.stepSimulation()
        
        # Return initial state
        state = self._get_state()
        return state, {}
    
    def _draw_path(self):
        """Draw the path in the PyBullet GUI"""
        if not self.gui:
            return
        
        # Clear existing debug items
        p.removeAllUserDebugItems()
        
        # Draw path points
        path_color = [0, 0.5, 0.8]
        for i in range(len(self.path_points) - 1):
            p1 = [self.path_points[i][0], self.path_points[i][1], 0.01]
            p2 = [self.path_points[i+1][0], self.path_points[i+1][1], 0.01]
            p.addUserDebugLine(p1, p2, path_color, 2)
    
    def _get_state(self):
        """Get the enhanced 8-element state vector"""
        # Get box position and orientation
        pos, quat = p.getBasePositionAndOrientation(self.box_id)
        current_position = np.array(pos[:2])  # x, y position
        
        # Convert quaternion to yaw angle (around z-axis)
        euler = p.getEulerFromQuaternion(quat)
        orientation = euler[2]  # yaw angle
        
        # Find closest point on the path
        dists = np.linalg.norm(self.path_points - current_position, axis=1)
        closest_idx = np.argmin(dists)
        closest_point = self.path_points[closest_idx]
        
        # Calculate progress along the path
        progress = closest_idx / (len(self.path_points) - 1)
        
        # Calculate deviation from the path
        deviation = dists[closest_idx]
        
        # Calculate path tangent and normal at closest point
        next_idx = min(closest_idx + 1, len(self.path_points) - 1)
        tangent = self.path_points[next_idx] - self.path_points[closest_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-8:
            path_tangent = tangent / tangent_norm
        else:
            path_tangent = np.array([1.0, 0.0])  # Default forward direction
        
        # Path normal (perpendicular to tangent)
        path_normal = np.array([-path_tangent[1], path_tangent[0]])
        
        # Calculate position error in path coordinate system
        position_error = current_position - closest_point
        lateral_error = np.dot(position_error, path_normal)      # perpendicular to path
        longitudinal_error = np.dot(position_error, path_tangent)  # along path
        
        # Calculate desired orientation (tangent direction)
        desired_orientation = np.arctan2(path_tangent[1], path_tangent[0])
        
        # Calculate orientation error and discretize it
        orientation_error = np.arctan2(
            np.sin(orientation - desired_orientation),
            np.cos(orientation - desired_orientation)
        )
        
        # Discretize orientation error
        bin_index = int(((orientation_error + np.pi) * 180/np.pi) / self.angle_bin_size) % self.num_bins
        discretized_orientation_error = (bin_index * self.angle_bin_size * np.pi/180) - np.pi
        
        # Calculate speed along path
        current_time = self.steps * self.dt * self.sim_steps
        if self.prev_position is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 1e-8:
                velocity = (current_position - self.prev_position) / dt
                speed_along_path = np.dot(velocity, path_tangent)
            else:
                speed_along_path = 0.0
        else:
            speed_along_path = 0.0
        
        # Update previous position and time
        self.prev_position = current_position.copy()
        self.prev_time = current_time
        
        # Calculate box orientation vectors (box reference frame in world coordinates)
        box_forward_x = np.cos(orientation)  # Box's forward direction
        box_forward_y = np.sin(orientation)
        
        # Return enhanced 8-element state vector
        state = np.array([
            lateral_error,                    # [0] how far left/right of path
            longitudinal_error,               # [1] how far ahead/behind on path
            discretized_orientation_error,    # [2] heading vs path tangent (discretized)
            progress,                         # [3] progress along path [0,1]
            deviation,                        # [4] euclidean distance to path
            speed_along_path,                 # [5] speed component along path
            box_forward_x,                    # [6] box forward vector x-component
            box_forward_y                     # [7] box forward vector y-component
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute an action in the environment."""
        self.steps += 1
        
        # Parse action: [force_x, force_y, torque_z]
        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        # For compatibility, add a max_torque attribute if not present
        if not hasattr(self, "max_torque"):
            self.max_torque = 50.0
        torque_z = np.clip(action[2], -1, 1) * self.max_torque
        
        # Get state before applying forces
        state_before = self._get_state()
        
        # Apply force and torque to the box for multiple steps
        for _ in range(self.sim_steps):
            # Get current position and velocity
            pos, quat = p.getBasePositionAndOrientation(self.box_id)
            vel, ang_vel = p.getBaseVelocity(self.box_id)

            # Check for numerical instability
            if (np.any(np.isnan(pos)) or np.any(np.isinf(pos)) or
                np.any(np.isnan(vel)) or np.any(np.isinf(vel))):
                print("Warning: Physics instability detected. Resetting object velocity.")
                p.resetBaseVelocity(self.box_id, [0, 0, 0], [0, 0, 0])
                break

            # Check for excessive velocity and dampen if needed
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > 8.0:  # Higher limit for lighter box
                scale_factor = 8.0 / vel_magnitude
                p.resetBaseVelocity(
                    self.box_id,
                    [vel[0] * scale_factor, vel[1] * scale_factor, vel[2] * scale_factor],
                    [0, 0, 0]
                )

            # Apply force at center of mass in LINK_FRAME
            force_3d = [force_x, force_y, 0]
            p.applyExternalForce(
                self.box_id,
                -1,
                force_3d,
                [0, 0, 0],  # Relative position in link frame (center of mass)
                p.LINK_FRAME  # Apply in box reference frame
            )

            # Apply torque around z-axis in box (link) frame
            p.applyExternalTorque(
                self.box_id,
                -1,
                [0, 0, torque_z],
                p.LINK_FRAME
            )

            # Step simulation
            p.stepSimulation()

            # Enhanced visualization for GUI mode
            if self.gui:
                self._draw_debug_info(pos, quat, force_x, force_y, torque_z)
        
        # Get new state after action
        state_after = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(state_before, state_after)
        
        # Extract state components
        progress = state_after[3]
        deviation = state_after[4]
        
        # Check if done
        done = False
        
        # Done if reached the end of the path
        if progress > 0.95 and deviation < self.goal_thresh:
            done = True
            # Add bonus reward for completing the path
            reward += self.goal_reward
            if self.gui:
                print(f"Goal reached at step {self.steps}! Final reward: {reward:.2f}")
        
        # Done if maximum steps reached
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            if self.gui:
                print(f"Max steps ({self.max_steps}) reached.")
        
        # Info dict with enhanced information
        info = {
            "steps": self.steps,
            "progress": progress,
            "deviation": deviation,
            "orientation_error": state_after[2],
            "lateral_error": state_after[0],
            "longitudinal_error": state_after[1],
            "speed_along_path": state_after[5],
            "box_forward_x": state_after[6],
            "box_forward_y": state_after[7]
        }
        
        # Display state information
        if self.gui and self.steps % 10 == 0:
            self._display_state_info(state_after, reward)
        
        return state_after, reward, done, truncated, info
    
    def _draw_debug_info(self, pos, quat, force_x, force_y, torque_z):
        """Draw comprehensive debug information in GUI mode (forces and torque)"""
        current_pos = np.array(pos[:2])
        
        # Find closest point on path
        dists = np.linalg.norm(self.path_points - current_pos, axis=1)
        closest_idx = np.argmin(dists)
        closest_point = self.path_points[closest_idx]
        
        # Draw current path connection (red line showing deviation)
        p.addUserDebugLine(
            [*closest_point, 0.05],
            [pos[0], pos[1], 0.05],
            [1, 0, 0], 1, 0.1
        )
        
        # Get and draw box orientation
        current_angle = p.getEulerFromQuaternion(quat)[2]
        next_idx = min(closest_idx + 1, len(self.path_points) - 1)
        tangent = self.path_points[next_idx] - self.path_points[closest_idx]
        desired_angle = np.arctan2(tangent[1], tangent[0])
        # current
        p.addUserDebugLine(
            [pos[0], pos[1], 0.05],
            [pos[0] + 0.3*np.cos(current_angle), pos[1] + 0.3*np.sin(current_angle), 0.05],
            [1, 0, 0], 3, 0.1
        )
        # desired
        if np.linalg.norm(tangent) > 1e-8:
            p.addUserDebugLine(
                [pos[0], pos[1], 0.05],
                [pos[0] + 0.3*np.cos(desired_angle), pos[1] + 0.3*np.sin(desired_angle), 0.05],
                [0, 0, 1], 3, 0.1
            )
        
        # Draw applied force vector (green) in world frame
        if np.hypot(force_x, force_y) > 1e-3:
            world_fx = force_x*np.cos(current_angle) - force_y*np.sin(current_angle)
            world_fy = force_x*np.sin(current_angle) + force_y*np.cos(current_angle)
            scale = 0.5 / max(self.max_force, 1)
            p.addUserDebugLine(
                [pos[0], pos[1], 0.05],
                [pos[0] + world_fx*scale, pos[1] + world_fy*scale, 0.05],
                [0, 1, 0], 4, 0.1
            )
        
        # Draw torque indicator (purple circle) scaled by magnitude of torque_z
        if abs(torque_z) > 1e-3:
            radius = 0.1 + 0.2 * (abs(torque_z)/max(self.max_torque,1))
            color = [1, 0, 1] if torque_z > 0 else [0.5, 0, 0.5]
            for i in range(8):
                a1 = i*np.pi/4
                a2 = (i+1)*np.pi/4
                p1 = [pos[0] + radius*np.cos(a1), pos[1] + radius*np.sin(a1), 0.05]
                p2 = [pos[0] + radius*np.cos(a2), pos[1] + radius*np.sin(a2), 0.05]
                p.addUserDebugLine(p1, p2, color, 2, 0.1)
        
        # Draw path tangent & normal at closest point for debugging
        if closest_idx < len(self.path_points)-1 and np.linalg.norm(tangent)>1e-8:
            tn = tangent/np.linalg.norm(tangent)
            nn = np.array([-tn[1], tn[0]])
            p.addUserDebugLine([*closest_point,0.02], [closest_point[0]+0.2*tn[0], closest_point[1]+0.2*tn[1],0.02], [0,1,1],2,0.1)
            p.addUserDebugLine([*closest_point,0.02], [closest_point[0]+0.2*nn[0], closest_point[1]+0.2*nn[1],0.02], [1,0,1],2,0.1)
    
    def _display_state_info(self, state, reward):
        """Display enhanced state information in GUI"""
        state_text = (
            f"Lateral Error: {state[0]:.2f}\n"
            f"Longitudinal Error: {state[1]:.2f}\n"
            f"Orient Error: {state[2]:.2f}\n"
            f"Progress: {state[3]:.2f}\n"
            f"Deviation: {state[4]:.2f}\n"
            f"Speed Along Path: {state[5]:.2f}\n"
            f"Box Forward: ({state[6]:.2f}, {state[7]:.2f})\n"
            f"Reward: {reward:.2f}"
        )
        
        # Add text at the top of the screen
        p.addUserDebugText(
            state_text,
            [0, 0, 1.5],  # Position above the scene
            textColorRGB=[1, 1, 1],
            textSize=1.0,
            lifeTime=0.5
        )
    
    def _calculate_reward(self, state_before, state_after):
        """Calculate reward based on enhanced path following performance"""
        # Extract relevant state components
        progress_before = state_before[3]
        progress_after = state_after[3]
        deviation = state_after[4]
        lateral_error = abs(state_after[0])
        orientation_error = abs(state_after[2])
        speed_along_path = state_after[5]
        
        # Progress reward - encourage forward movement
        progress_reward = 10.0 * (progress_after - progress_before)
        
        # Path following rewards
        deviation_penalty = -3.0 * deviation
        lateral_penalty = -2.0 * lateral_error
        orientation_penalty = -1.0 * orientation_error
        
        # Speed regulation - encourage reasonable speed along path
        target_speed = 0.3  # m/s
        speed_reward = -1.0 * abs(speed_along_path - target_speed)
        
        # Combine rewards
        total_reward = (progress_reward + deviation_penalty + 
                       lateral_penalty + orientation_penalty + speed_reward)
        
        # Goal completion bonus
        if progress_after > 0.95 and deviation < self.goal_thresh:
            total_reward += 20.0
        
        return float(total_reward)
    
    def close(self):
        """Close the environment"""
        p.disconnect()