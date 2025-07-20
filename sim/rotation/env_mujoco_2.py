# env_mujoco.py

import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time
from gymnasium import spaces
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

class SimplePathFollowingEnv(gym.Env):
    """
    Path Following Environment for Multi-Robot Sim-to-Real Transfer (MuJoCo Version)
    
    A hollow plywood box (~65kg, 0.4x0.4x0.4m) follows a curved path using force vectors
    applied in the box reference frame. The state representation is path-relative 
    and includes box orientation vectors to enable multi-robot coordination.
    
    State: [lateral_error, longitudinal_error, orientation_error, progress, 
            deviation, speed_along_path, box_forward_x, box_forward_y] (8D)
    Action: [force_x, force_y, torque_z] (3D)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, model_path='scene.xml', render_mode=None, **kwargs):
        super(SimplePathFollowingEnv, self).__init__()
        self.render_mode = render_mode

        # Load parameters from kwargs, with defaults
        self.gui = kwargs.get('gui', False)
        self.max_force = kwargs.get('max_force', 400.0)
        self.max_torque = kwargs.get('max_torque', 50.0)
        self.goal_thresh = kwargs.get('goal_thresh', 0.2)
        self.max_steps = kwargs.get('max_steps', 500)
        self.goal_pos = kwargs.get('goal_pos', None) 
        self.friction = kwargs.get('friction', 0.2)
        self.goal_reward = kwargs.get('goal_reward', 100)
        self.segment_length = kwargs.get('segment_length', 0.3)
        self.test_full_arc = kwargs.get('test_full_arc', False)
        self.arc_radius = kwargs.get('arc_radius', 1.5)
        self.arc_start = kwargs.get('arc_start', -np.pi/3)
        self.arc_end = kwargs.get('arc_end', np.pi/3)
        self.spinning_friction = kwargs.get('spinning_friction', 0.01)
        self.rolling_friction = kwargs.get('rolling_friction',  0.01)
        self.strict_terminal = kwargs.get('strict_terminal',  False)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Get the address of the first DoF for the box joint
        dof_adr = self.model.joint('box_joint').dofadr[0]

        # Set damping for the 3 translational (linear) DoFs
        linear_damping = kwargs.get('linear_damping', 0.05)
        self.model.dof_damping[dof_adr:dof_adr+3] = linear_damping

        # Set damping for the 3 rotational (angular) DoFs
        angular_damping = kwargs.get('angular_damping', 0.1)
        self.model.dof_damping[dof_adr+3:dof_adr+6] = angular_damping
        
        # Get geom IDs for the box and floor
        box_geom_id = self.model.geom('box_geom').id
        floor_geom_id = self.model.geom('floor').id
        
        # Define the friction properties based on the PyBullet baseline
        # [sliding_friction, torsional_friction, rolling_friction]
        # We use the 'friction' from config for sliding, and PyBullet's defaults for the others.
        friction_coeffs = np.array([
            self.friction,  # Use sliding friction from config (0.5)
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
        
        self.angle_bin_size = 10.0
        self.num_bins = int(360 / self.angle_bin_size)
        
        low = np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0, -np.inf, -1.0, -1.0], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.pi, 1.0, np.inf, np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        self.full_path = self._generate_arc_path(self.arc_radius, self.arc_start, self.arc_end)
        self._make_training_segment()

        self.prev_position = None
        self.prev_time = None
        self.steps = 0

    def _generate_arc_path(self, radius=1.5, start_angle=-np.pi/3, end_angle=np.pi/3, num_points=50):
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

        start_dist = np.random.rand() * (total_length - seg_len)
        end_dist = start_dist + seg_len

        i0 = np.searchsorted(cumlen, start_dist, side='right') - 1
        i1 = np.searchsorted(cumlen, end_dist, side='right')
        i0 = max(0, i0)
        i1 = min(n - 1, i1)
        self.path_points = full[i0 : i1 + 1].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.test_full_arc:
            self.full_path = self._generate_arc_path(self.arc_radius, self.arc_start, self.arc_end)
            self.segment_length = None
        self._make_training_segment()
        
        mujoco.mj_resetData(self.model, self.data)
        
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
        
        if self.gui and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        return self._get_state(), {}

    def _get_state(self):
        pos = self.data.body('box').xpos
        quat = self.data.body('box').xquat
        current_position = pos[:2]
        
        orientation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')[2]
        
        dists = np.linalg.norm(self.path_points - current_position, axis=1)
        closest_idx = np.argmin(dists)
        closest_point = self.path_points[closest_idx]
        progress = closest_idx / (len(self.path_points) - 1)
        deviation = dists[closest_idx]
        
        next_idx = min(closest_idx + 1, len(self.path_points) - 1)
        tangent = self.path_points[next_idx] - self.path_points[closest_idx]
        tangent_norm = np.linalg.norm(tangent)
        path_tangent = tangent / tangent_norm if tangent_norm > 1e-8 else np.array([1.0, 0.0])
        path_normal = np.array([-path_tangent[1], path_tangent[0]])
        
        position_error = current_position - closest_point
        lateral_error = np.dot(position_error, path_normal)
        longitudinal_error = np.dot(position_error, path_tangent)
        
        desired_orientation = np.arctan2(path_tangent[1], path_tangent[0])
        orientation_error = np.arctan2(np.sin(orientation - desired_orientation), np.cos(orientation - desired_orientation))
        
        bin_index = int(((orientation_error + np.pi) * 180/np.pi) / self.angle_bin_size) % self.num_bins
        discretized_orientation_error = (bin_index * self.angle_bin_size * np.pi/180) - np.pi
        
        current_time = self.steps * self.model.opt.timestep * self.sim_steps
        dt = current_time - self.prev_time if self.prev_time is not None else 0
        if dt > 1e-8:
            velocity = (current_position - self.prev_position) / dt
            speed_along_path = np.dot(velocity, path_tangent)
        else:
            speed_along_path = 0.0
            
        self.prev_position = current_position.copy()
        self.prev_time = current_time
        
        box_forward_x = np.cos(orientation)
        box_forward_y = np.sin(orientation)
        
        state = np.array([
            lateral_error, longitudinal_error, discretized_orientation_error, progress, deviation,
            speed_along_path, box_forward_x, box_forward_y
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        self.steps += 1
        
        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        torque_z = np.clip(action[2], -1, 1) * self.max_torque
        
        # This can be called once before the loop
        state_before = self._get_state()
        
        for _ in range(self.sim_steps):
            box_quat = self.data.body('box').xquat
            rot_matrix = Rotation.from_quat([box_quat[1], box_quat[2], box_quat[3], box_quat[0]]).as_matrix()

            force_local = np.array([force_x, force_y, 0])
            torque_local = np.array([0, 0, torque_z])
            force_world = rot_matrix @ force_local
            torque_world = rot_matrix @ torque_local
            
            wrench_world = np.concatenate([force_world, torque_world])
            
            # Re-apply the force before every single mj_step
            self.data.xfrc_applied[self.box_body_id] = wrench_world

            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            
        state_after = self._get_state()
        reward, reward_comps = self._calculate_reward(state_before, state_after)
        
        progress = state_after[3]
        deviation = state_after[4]
        orientation_error = abs(state_after[2])

        done = False

        # # STRICT VERSION, NO LOGS
        # if progress > 0.95 and deviation < self.goal_thresh:
        #     done = True
        #     # reward += self.goal_reward
        #     if orientation_error < 0.2: # Check for low orientation error (~11 degrees)
        #         reward += self.goal_reward
        #     else:
        #         # Reached the area but with bad orientation, penalize slightly
        #         reward -= 10.0

        # # STRICT VERSION, WITH LOGS
        # terminal_adj = 0.0
        # terminal_event = None
        # if progress > 0.95 and deviation < self.goal_thresh:
        #     done = True
        #     if orientation_error < 0.2:
        #         terminal_adj = self.goal_reward
        #         reward += self.goal_reward
        #         terminal_event = "success"
        #     else:
        #         terminal_adj = -10.0
        #         reward -= 10.0
        #         terminal_event = "orient_fail"

        # # UNSTRICT VERSION WITH LOGS:
        # terminal_adj = 0.0
        # terminal_event = None
        # if progress > 0.95 and deviation < self.goal_thresh:
        #     done = True
        #     terminal_adj = self.goal_reward
        #     reward += self.goal_reward
        #     terminal_event = "success"

        # COMPREHENSIVE VERSION:
        terminal_adj  = 0.0
        terminal_event = None

        if progress > 0.95 and deviation < self.goal_thresh:
            done = True

            if self.strict_terminal:
                # orientation check enforced
                if orientation_error < 0.2:
                    terminal_adj = self.goal_reward
                    terminal_event = "success"
                else:
                    terminal_adj = 0.1 * self.goal_reward
                    terminal_event = "orient_fail"
            else:
                # disregard orientation
                terminal_adj = self.goal_reward
                terminal_event = "success"

            reward += terminal_adj
        
        truncated = self.steps >= self.max_steps
        
        info = {
            "progress":           progress,
            "deviation":          deviation,
            "lateral_error":      abs(state_after[0]),
            "longitudinal_error": abs(state_after[1]),
            "orientation_error":  orientation_error,
            "speed_along_path":   state_after[5],
            "reward_comps":       reward_comps,
            "terminal_adjustment": terminal_adj,
            "terminal_event":      terminal_event,
        }
        
        return state_after, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

        elif self.render_mode == "rgb_array":
            # off-screen render
            # note: mujoco-python has a built-in convenience call:
            return self.model.render(
                height=600,
                width=800,
                camera_id=0,      # or whichever camera you want
                segmentation=False,
                depth=False
            )

        else:
            # no-op
            return None
    def _calculate_reward(self, state_before, state_after):
        progress_before, progress_after = state_before[3], state_after[3]
        deviation, lateral_error, orientation_error = state_after[4], abs(state_after[0]), abs(state_after[2])
        speed_along_path = state_after[5]
        
        progress_reward = 10.0 * (progress_after - progress_before)
        # set these coefficients the same (all -1.5 for example)
        # deviation_penalty = -3.0 * deviation
        # lateral_penalty = -2.0 * lateral_error
        # orientation_penalty = -1.0 * orientation_error
        deviation_penalty = -6 * deviation
        lateral_penalty = -4 * lateral_error
        orientation_penalty = -4 * orientation_error

        adherence_reward = 0.5 * np.exp(-20.0 * deviation)    
        
        target_speed = 0.3
        speed_reward = -1.0 * abs(speed_along_path - target_speed)
        
        total_reward = progress_reward + deviation_penalty + lateral_penalty + orientation_penalty + speed_reward + adherence_reward
        
        # goal_bonus = 0

        # if progress_after > 0.95 and deviation < self.goal_thresh:
        #     total_reward += 20.0
        #     goal_bonus += 20
            
        return float(total_reward), {
            "ProgressReward": progress_reward,
            "DeviationPenalty": deviation_penalty,
            "LateralPenalty": lateral_penalty,
            "OrientationPenalty": orientation_penalty,
            "SpeedReward": speed_reward,
            "AdherenceReward": adherence_reward,
            # "GoalBonus": goal_bonus
        }

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None