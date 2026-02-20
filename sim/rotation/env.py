import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
from scipy.spatial.transform import Rotation


class SimplePathFollowingEnv(gym.Env):
    """
    Path Following Environment for Multi-Robot Sim-to-Real Transfer

    A hollow plywood box (~65kg, 0.4x0.4x0.4m) follows a curved path using force vectors
    applied in the box reference frame. The state representation is path-relative
    and includes egocentric velocity and physics features for sim-to-real transfer.

    State: [lateral_err, orientation_err, speed_fwd, speed_lat, angular_vel, norm_reactive_force] (6D)
    Action: [force_x, force_y, torque_z] (3D)

    goal_thresh is computed dynamically at each reset() as:
        goal_thresh = goal_thresh_pct * segment_length
    This ensures the success criterion scales consistently with segment length,
    whether training on short segments or evaluating on full arcs.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, model_path='scene.xml', render_mode=None, **kwargs):
        super(SimplePathFollowingEnv, self).__init__()
        self.render_mode = render_mode

        # Load parameters from kwargs, with defaults
        self.gui = kwargs.get('gui', False)
        self.max_force = kwargs.get('max_force', 400.0)
        self.max_torque = kwargs.get('max_torque', 50.0)
        self.goal_thresh_pct = kwargs.get('goal_thresh_pct', 0.10)
        self.max_steps = kwargs.get('max_steps', 500)
        self.goal_pos = kwargs.get('goal_pos', None)
        self.friction = kwargs.get('friction', 0.2)
        self.mass_range = kwargs.get('mass_range', [15.0, 25.0])
        self.friction_range = kwargs.get('friction_range', [0.4, 0.6])
        self.goal_reward = kwargs.get('goal_reward', 100)
        self.segment_length = kwargs.get('segment_length', 0.3)
        self.test_full_arc = kwargs.get('test_full_arc', False)
        self.arc_radius = kwargs.get('arc_radius', 1.5)
        self.arc_start = kwargs.get('arc_start', -np.pi / 3)
        self.arc_end = kwargs.get('arc_end', np.pi / 3)
        self.spinning_friction = kwargs.get('spinning_friction', 0.01)
        self.rolling_friction = kwargs.get('rolling_friction', 0.01)
        self.spin_penalty_k = kwargs.get('spin_penalty_k', 0.0)
        self.deviation_tolerance = kwargs.get('deviation_tolerance', 0.15)
        self.strict_terminal = kwargs.get('strict_terminal', False)

        # goal_thresh is set dynamically at reset() based on segment length
        self.goal_thresh = None

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)

        # Get the address of the first DoF for the box joint
        dof_adr = self.model.joint('box_joint').dofadr[0]

        # Set damping for the 3 translational (linear) DoFs
        linear_damping = kwargs.get('linear_damping', 0.05)
        self.model.dof_damping[dof_adr:dof_adr + 3] = linear_damping

        # Set damping for the 3 rotational (angular) DoFs
        angular_damping = kwargs.get('angular_damping', 0.1)
        self.model.dof_damping[dof_adr + 3:dof_adr + 6] = angular_damping

        # Get geom IDs for the box and floor
        box_geom_id = self.model.geom('box_geom').id
        floor_geom_id = self.model.geom('floor').id

        friction_coeffs = np.array([
            self.friction,
            self.spinning_friction,
            self.rolling_friction
        ])

        # Apply these friction values to both the box and the floor
        self.model.geom_friction[box_geom_id] = friction_coeffs
        self.model.geom_friction[floor_geom_id] = friction_coeffs
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.box_body_id = self.model.body('box').id
        self.box_joint_id = self.model.joint('box_joint').id

        # dt per timestep
        self.model.opt.timestep = 0.0025
        self.sim_steps = 10

        # State: [lateral_err, orientation_err, speed_fwd, speed_lat, angular_vel, norm_reactive_force]
        low = np.array([-np.inf, -np.pi, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32)
        high = np.array([np.inf, np.pi, np.inf, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.full_path = self._generate_arc_path(self.arc_radius, self.arc_start, self.arc_end)

        self.prev_position = None
        self.prev_time = None
        self.steps = 0
        self.last_closest_idx = 0

    def _generate_arc_path(self, radius=1.5, start_angle=-np.pi / 3, end_angle=np.pi / 3, num_points=50):
        points = []
        for theta in np.linspace(start_angle, end_angle, num_points):
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append(np.array([x, y]))
        return np.array(points)

    def _make_training_segment(self):
        full = self.full_path
        n = len(full)
        if self.segment_length is None:
            self.path_points = full.copy()
            return

        disp = np.linalg.norm(np.diff(full, axis=0), axis=1)
        cumlen = np.concatenate(([0.0], np.cumsum(disp)))

        total_length = cumlen[-1]
        seg_len = self.segment_length

        if seg_len >= total_length or n < 2:
            self.path_points = full.copy()
            return

        start_dist = self.np_random.random() * (total_length - seg_len)
        end_dist = start_dist + seg_len

        i0 = np.searchsorted(cumlen, start_dist, side='right') - 1
        i1 = np.searchsorted(cumlen, end_dist, side='right')
        i0 = max(0, i0)
        i1 = min(n - 1, i1)
        self.path_points = full[i0: i1 + 1].copy()

    def _compute_segment_length(self):
        """Compute the arc length of the current path_points segment."""
        if len(self.path_points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.path_points, axis=0), axis=1)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.test_full_arc:
            self.full_path = self._generate_arc_path(self.arc_radius, self.arc_start, self.arc_end)
            self.segment_length = None
        self._make_training_segment()

        # Compute goal_thresh dynamically, but cap it at a strict maximum of 5 cm (0.05m)
        seg_len = self._compute_segment_length()
        self.goal_thresh = min(self.goal_thresh_pct * seg_len, 0.05)

        mujoco.mj_resetData(self.model, self.data)

        # --- Domain Randomization: sample mass and friction each episode ---
        sampled_mass = self.np_random.uniform(self.mass_range[0], self.mass_range[1])
        self.model.body_mass[self.box_body_id] = sampled_mass

        sampled_friction = self.np_random.uniform(self.friction_range[0], self.friction_range[1])
        box_geom_id = self.model.geom('box_geom').id
        floor_geom_id = self.model.geom('floor').id
        self.model.geom_friction[box_geom_id][0] = sampled_friction
        self.model.geom_friction[floor_geom_id][0] = sampled_friction

        start_pos = [self.path_points[0][0], self.path_points[0][1], 0.2]

        tangent = self.path_points[1] - self.path_points[0]
        angle = np.arctan2(tangent[1], tangent[0])
        start_quat = Rotation.from_euler('xyz', [0, 0, angle]).as_quat()
        start_quat /= np.linalg.norm(start_quat)

        qpos = np.zeros(self.model.nq)
        qpos[0:3] = start_pos
        qpos[3:7] = [start_quat[3], start_quat[0], start_quat[1], start_quat[2]]
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.prev_position = np.array(start_pos[:2])
        self.prev_time = 0.0
        self.last_closest_idx = 0

        if self.gui and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        state, _ = self._get_state()
        return state, {}

    def _get_state(self):
        pos = self.data.body('box').xpos
        quat = self.data.body('box').xquat
        current_position = pos[:2]
        orientation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')[2]

        # --- Find Closest Point ---
        search_start_idx = max(0, self.last_closest_idx - 10)
        search_end_idx = min(self.last_closest_idx + 20, len(self.path_points))
        path_segment_to_search = self.path_points[search_start_idx:search_end_idx]

        if len(path_segment_to_search) > 0:
            dists = np.linalg.norm(path_segment_to_search - current_position, axis=1)
            segment_closest_idx = np.argmin(dists)
            closest_idx = search_start_idx + segment_closest_idx
        else:
            closest_idx = self.last_closest_idx
        self.last_closest_idx = closest_idx

        closest_point = self.path_points[closest_idx]
        num_path_points = len(self.path_points)
        progress = closest_idx / (num_path_points - 1) if num_path_points > 1 else 0.0
        deviation = np.linalg.norm(current_position - closest_point)

        # --- Tangent Singularity Fix ---
        if closest_idx == len(self.path_points) - 1 and len(self.path_points) > 1:
            tangent = self.path_points[closest_idx] - self.path_points[closest_idx - 1]
        else:
            next_idx = min(closest_idx + 1, len(self.path_points) - 1)
            tangent = self.path_points[next_idx] - self.path_points[closest_idx]

        tangent_norm = np.linalg.norm(tangent)
        path_tangent = tangent / tangent_norm if tangent_norm > 1e-8 else np.array([1.0, 0.0])
        path_normal = np.array([-path_tangent[1], path_tangent[0]])

        # --- Errors ---
        position_error = current_position - closest_point
        lateral_error = np.dot(position_error, path_normal)
        longitudinal_error = np.dot(position_error, path_tangent)

        desired_orientation = np.arctan2(path_tangent[1], path_tangent[0])
        # Continuous orientation error (no binning)
        orientation_error = np.arctan2(
            np.sin(orientation - desired_orientation),
            np.cos(orientation - desired_orientation)
        )

        # --- Velocities ---
        current_time = self.steps * self.model.opt.timestep * self.sim_steps
        dt = current_time - self.prev_time if self.prev_time is not None else 0

        if dt > 1e-8:
            velocity_2d = (current_position - self.prev_position) / dt
            speed_forward = np.dot(velocity_2d, path_tangent)
            speed_lateral = np.dot(velocity_2d, path_normal)
        else:
            speed_forward = 0.0
            speed_lateral = 0.0

        angular_velocity = self.data.qvel[5]

        self.prev_position = current_position.copy()
        self.prev_time = current_time

        # --- Physics Feature (Reactive Force) ---
        mass = self.model.body_mass[self.box_body_id]
        friction = self.model.geom_friction[self.model.geom('box_geom').id][0]
        norm_reactive_force = (mass * friction * 9.81) / self.max_force
        norm_reactive_force = np.clip(norm_reactive_force, 0.0, 1.0)

        # 6D Egocentric State
        state = np.array([
            lateral_error, orientation_error, speed_forward, speed_lateral,
            angular_velocity, norm_reactive_force
        ], dtype=np.float32)

        # Return state AND tracking metrics
        metrics = {
            "progress": progress,
            "deviation": deviation,
            "longitudinal_error": longitudinal_error
        }
        return state, metrics

    def step(self, action):
        self.steps += 1

        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        torque_z = np.clip(action[2], -1, 1) * self.max_torque

        for _ in range(self.sim_steps):
            box_quat = self.data.body('box').xquat
            rot_matrix = Rotation.from_quat(
                [box_quat[1], box_quat[2], box_quat[3], box_quat[0]]
            ).as_matrix()

            force_local = np.array([force_x, force_y, 0])
            torque_local = np.array([0, 0, torque_z])
            force_world = rot_matrix @ force_local
            torque_world = rot_matrix @ torque_local

            wrench_world = np.concatenate([force_world, torque_world])
            self.data.xfrc_applied[self.box_body_id] = wrench_world
            mujoco.mj_step(self.model, self.data)

        state_after, metrics = self._get_state()

        # Unpack metrics
        progress = metrics["progress"]
        deviation = metrics["deviation"]
        longitudinal_error = metrics["longitudinal_error"]

        # New index mapping for 6D state
        lateral_error = abs(state_after[0])
        orientation_error = abs(state_after[1])
        speed_along_path = state_after[2]
        angular_velocity = state_after[4]

        # Pass action into reward calculation
        reward, reward_comps = self._calculate_reward(state_after, action, angular_velocity)

        done = False
        terminal_event = None
        terminal_adj = 0.0

        # 1. Strict Failure: out of the virtual tube
        if lateral_error > self.deviation_tolerance:
            done = True
            terminal_event = "wandered_off"
            terminal_adj = -50.0

        # 2. Strict Success: reached end while staying on path
        elif progress > 0.95 and deviation < self.goal_thresh:
            done = True
            if self.strict_terminal:
                if orientation_error < 0.2:
                    terminal_event = "success"
                else:
                    terminal_event = "orient_fail"
                    terminal_adj = -10.0  # Partial penalty for bad final orientation
            else:
                terminal_event = "success"

        # Apply goal reward only on clean success
        if terminal_event == "success":
            terminal_adj = self.goal_reward

        reward += terminal_adj

        truncated = self.steps >= self.max_steps

        info = {
            "progress": progress,
            "deviation": deviation,
            "lateral_error": lateral_error,
            "longitudinal_error": abs(longitudinal_error),
            "orientation_error": orientation_error,
            "speed_along_path": speed_along_path,
            "goal_thresh": self.goal_thresh,
            "deviation_tolerance": self.deviation_tolerance,
            "reward_comps": reward_comps,
            "terminal_event": terminal_event,
            "terminal_adjustment": terminal_adj,
        }

        return state_after, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            return self.model.render(
                height=600,
                width=800,
                camera_id=0,
                segmentation=False,
                depth=False
            )
        else:
            return None

    def _calculate_reward(self, state_after, action, angular_velocity_z):
        # state: [lateral_err, orientation_err, speed_fwd, speed_lat, angular_vel, norm_reactive_force]
        lateral_error = abs(state_after[0])
        orientation_error = abs(state_after[1])
        speed_along_path = state_after[2]
        dt = self.model.opt.timestep * self.sim_steps

        # 1. Positive Progress: earn points for distance covered along the path tangent
        distance_forward = speed_along_path * dt
        r_progress = 100.0 * distance_forward  # completing a 0.3m segment yields ~+30 steady points

        # 2. Speed Penalty: gentle nudge to keep near target speed
        target_speed = 0.3
        r_speed = -2.0 * abs(speed_along_path - target_speed)

        # 3. Virtual Constraints: keep box on the rails (softened)
        r_constraint_lat = -10.0 * (lateral_error ** 2)
        r_constraint_ori = -2.0 * (orientation_error ** 2)

        # 4. Mechanical Efficiency: penalize spinning
        r_eff_spin = -0.5 * (angular_velocity_z ** 2)

        total_reward = r_progress + r_speed + r_constraint_lat + r_constraint_ori + r_eff_spin

        return float(total_reward), {
            "R_Progress": r_progress,
            "R_Speed": r_speed,
            "R_Constraint_Lat": r_constraint_lat,
            "R_Constraint_Ori": r_constraint_ori,
            "R_Eff_Spin": r_eff_spin,
        }

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
