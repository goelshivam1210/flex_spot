# sim/env.py
import gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

'''
env design overview:
- A cube is placed on a plane and must reach a target XY coordinate
- Two pushers (the bots), are mounted on the rear face, offset to the left and
  right by 'pusher_offset_distance'.
- Action = [left_thrust, right_thrust]: [0,1]^2, forces applied in the same
  forward direction but at the two lateral contact points
- Reward: projection of step-wise displacement onto goal direction, and major
  reward (+100) when within the threshold distance
'''

class PrismaticEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, gui=False, max_force_per_pusher=50.0, friction=0.5,
                 angular_damping_factor=0.2,
                 goal_pos=np.array([2.0, 0.0]), goal_thresh=0.1, max_steps=500,
                 pusher_offset_distance=0.15,
                 randomize_initial_yaw=True,
                 initial_yaw_range=[-math.pi, math.pi]): # Default to full 360 randomization
        super(PrismaticEnv, self).__init__()
        self.gui = gui
        self.max_force_per_pusher = max_force_per_pusher
        self.friction = friction
        self.angular_damping_factor = angular_damping_factor
        self.goal_pos_world = np.array(goal_pos)
        self.goal_thresh = goal_thresh
        self.max_steps = max_steps
        self.dt = 1.0 / 240.0
        self.sim_steps = 12

        # pusher geometry
        self.pusher_offset_distance = pusher_offset_distance
        self.box_half_extents = np.array([0.25, 0.25, 0.25]) #default

        # initial yaw randomization
        self.randomize_initial_yaw = randomize_initial_yaw
        self.initial_yaw_range = initial_yaw_range


        # Observation: [cos(yaw), sin(yaw), disp_x, disp_y, vel_x, vel_y, ang_vel_z] (7D)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # Action: [left_thrust_scale, right_thrust_scale], scales between 0 and 1 (2D)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.client = -1
        self._connect()
        
        self.fixed_local_push_direction = np.array([1.0, 0.0]) #default
        self.cp1_local_fixed = np.zeros(2)
        self.cp2_local_fixed = np.zeros(2)

        # Visual helpers
        self.bot1_vis = -1; self.bot2_vis = -1
        self.arrow1_shaft, self.arrow1_head, self.arrow1_shaft_len, self.arrow1_head_len = -1,-1,0.3,0.06
        self.arrow2_shaft, self.arrow2_head, self.arrow2_shaft_len, self.arrow2_head_len = -1,-1,0.3,0.06
        
        self.initial_box_pos_xy = np.zeros(2)
        self.prev_box_pos_xy = np.zeros(2)
        self.steps = 0

    #
    # Creates arrows for GUI debugging, didn't end up using it
    #
    def _build_arrow(self, color):
        shaft_len = 0.3; head_len = 0.06;
        shaft_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01, length=shaft_len, rgbaColor=color + [0.8], physicsClientId=self.client)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001,0.001,0.001], physicsClientId=self.client)
        try: head_vis = p.createVisualShape(p.GEOM_CONE, radius=0.03, length=head_len, rgbaColor=color + [0.8], physicsClientId=self.client)
        except: head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=color + [0.8], physicsClientId=self.client)
        shaft_id  = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=shaft_vis, physicsClientId=self.client)
        head_id   = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=head_vis, physicsClientId=self.client)
        p.setCollisionFilterGroupMask(shaft_id, -1, 0, 0, physicsClientId=self.client)
        p.setCollisionFilterGroupMask(head_id,  -1, 0, 0, physicsClientId=self.client)
        return shaft_id, head_id, shaft_len, head_len


    def _create_bot_visual(self, radius=0.05, height=0.03, color=[1,0,0,0.7]): # As provided, with client id
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=self.client)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001,0.001,0.001], physicsClientId=self.client)
        bot_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis, basePosition=[0,0,height/2], physicsClientId=self.client)
        p.setCollisionFilterGroupMask(bot_id, -1, 0, 0, physicsClientId=self.client)
        return bot_id
        
    #
    # Connect to pyullet physics engine at start, (kept across episodes)
    #
    def _connect(self):
        if self.client < 0:
            if self.gui:
                self.client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
                p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30,
                                             cameraTargetPosition=[0.75, 0.0, 0.1], physicsClientId=self.client)
            else:
                self.client = p.connect(p.DIRECT)
        # set search path, timestep, gravity
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

    #
    # get initial position, choose rear face, pre-compute contact points
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and hasattr(self.np_random, 'seed'):
            self.np_random.seed(seed)

        # wipe phyiscs world
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # spawn cube with random yaw
        initial_yaw = 0.0
        if self.randomize_initial_yaw:
            initial_yaw = self.np_random.uniform(self.initial_yaw_range[0], self.initial_yaw_range[1])
        
        start_pos_3d = [0, 0, 0.1]
        start_ori_quat = p.getQuaternionFromEuler([0, 0, initial_yaw])
        self.box_id = p.loadURDF("cube.urdf", start_pos_3d, start_ori_quat,
                                 globalScaling=0.5, physicsClientId=self.client)

        aabb = p.getAABB(self.box_id, -1, physicsClientId=self.client)
        self.box_half_extents = (np.array(aabb[1]) - np.array(aabb[0])) / 2.0
        # print(f"DEBUG Reset: Box half extents: {self.box_half_extents}")

        p.changeDynamics(self.box_id, -1,
                         lateralFriction=self.friction,
                         angularDamping=self.angular_damping_factor,
                         physicsClientId=self.client)
        
        if self.gui: # Create visuals only if GUI is enabled
            if self.bot1_vis == -1: # Create only once
                self.bot1_vis = self._create_bot_visual(color=[1,0,0,0.7]) # Red
                self.bot2_vis = self._create_bot_visual(color=[0,0,1,0.7]) # Blue
                self.arrow1_shaft, self.arrow1_head, self.arrow1_shaft_len, self.arrow1_head_len = self._build_arrow([1,0,0])
                self.arrow2_shaft, self.arrow2_head, self.arrow2_shaft_len, self.arrow2_head_len = self._build_arrow([0,0,1])

        # decide which face is the rear
        box_pos_w_initial, box_orn_w_quat_initial = p.getBasePositionAndOrientation(self.box_id, physicsClientId=self.client)
        self.initial_box_pos_xy = np.array(box_pos_w_initial[:2]) # Store absolute initial XY
        
        current_initial_yaw = p.getEulerFromQuaternion(box_orn_w_quat_initial, physicsClientId=self.client)[2]

        goal_dir_world = self.goal_pos_world - self.initial_box_pos_xy
        
        # Transform goal direction to box's initial local frame
        R_world_to_local_initial = np.array([
            [ math.cos(-current_initial_yaw), -math.sin(-current_initial_yaw)],
            [ math.sin(-current_initial_yaw),  math.cos(-current_initial_yaw)]])
        goal_dir_local = R_world_to_local_initial @ goal_dir_world

        # Outward normals in box's local frame
        face_normals_local_frame = { 
            "pos_x": np.array([ 1,  0]), "neg_x": np.array([-1,  0]),
            "pos_y": np.array([ 0,  1]), "neg_y": np.array([ 0, -1])}
        
        # pick the face whose normal is most opposite the local goal direction
        min_dot_product = np.inf
        best_face_local_normal = None
        best_face_name_for_extent = "neg_x" # Default

        for name, local_normal_vec in face_normals_local_frame.items():
            dot_p = np.dot(local_normal_vec, goal_dir_local) # How much face normal aligns with local goal direction
            if dot_p < min_dot_product: # We want the face whose normal is most ANTI-ALIGNED with goal_dir_local
                min_dot_product = dot_p
                best_face_local_normal = local_normal_vec # This is the normal of the "rear face"
                best_face_name_for_extent = name

        self.fixed_local_push_direction = -best_face_local_normal # We push *into* this rear face, so opposite its normal
        
        # Determine local lateral axis for pusher offset (perpendicular to push_direction in local frame)
        # If pushing along local X (+/-1,0), lateral is local Y (0, +/-1)
        # If pushing along local Y (0, +/-1), lateral is local X (+/-1, 0)
        if abs(self.fixed_local_push_direction[0]) > abs(self.fixed_local_push_direction[1]): # Push is primarily along X
            self.fixed_local_lateral_axis = np.array([0.0, 1.0]) # Use local Y for lateral offset
            face_half_extent = self.box_half_extents[0] # Use X half-extent for depth
        else: # Push is primarily along Y
            self.fixed_local_lateral_axis = np.array([1.0, 0.0]) # Use local X for lateral offset
            face_half_extent = self.box_half_extents[1] # Use Y half-extent for depth
        
        # Ensure lateral axis is normalized (it is by construction here)
        # lateral_norm = np.linalg.norm(self.fixed_local_lateral_axis)
        # if lateral_norm > 1e-6: self.fixed_local_lateral_axis /= lateral_norm

        # Contact points are relative to the CoM, on the chosen rear face, offset laterally
        # The "rear face" surface is at best_face_local_normal * face_half_extent from CoM
        local_face_center = best_face_local_normal * face_half_extent

        self.cp1_local_fixed = local_face_center + self.fixed_local_lateral_axis * self.pusher_offset_distance
        self.cp2_local_fixed = local_face_center - self.fixed_local_lateral_axis * self.pusher_offset_distance

        for _ in range(30): # Settling steps
            p.stepSimulation(physicsClientId=self.client)
        
        # Re-fetch initial position AFTER settling and after fixed strategy is defined
        box_start_pos_w_final, _ = p.getBasePositionAndOrientation(self.box_id, physicsClientId=self.client)
        self.initial_box_pos_xy = np.array(box_start_pos_w_final[:2])
        self.prev_box_pos_xy = self.initial_box_pos_xy.copy()
        self.steps = 0
        
        return self._get_obs(), {}

    #
    # Observation helper
    #
    def _get_obs(self):
        box_pos_w, box_orn_w_quat = p.getBasePositionAndOrientation(self.box_id, physicsClientId=self.client)
        box_lin_vel_w, box_ang_vel_w = p.getBaseVelocity(self.box_id, physicsClientId=self.client)
        current_pos_xy_world = np.array(box_pos_w[:2])
        displacement_xy = current_pos_xy_world - self.initial_box_pos_xy
        _, _, yaw = p.getEulerFromQuaternion(box_orn_w_quat, physicsClientId=self.client)
        cos_yaw = math.cos(yaw); sin_yaw = math.sin(yaw)
        lin_vel_xy = np.array(box_lin_vel_w[:2]); ang_vel_z = box_ang_vel_w[2]
        obs = np.array([cos_yaw, sin_yaw, displacement_xy[0], displacement_xy[1],
                        lin_vel_xy[0], lin_vel_xy[1], ang_vel_z], dtype=np.float32)
        return obs

    #
    # Apply thrust and compute reward
    #
    def step(self, action):
        left_thrust_scale = np.clip(action[0], 0.0, 1.0)
        right_thrust_scale = np.clip(action[1], 0.0, 1.0)

        # get pose
        box_pos_w, box_orn_w_quat = p.getBasePositionAndOrientation(self.box_id, physicsClientId=self.client)
        _, _, box_yaw_rad_world = p.getEulerFromQuaternion(box_orn_w_quat, physicsClientId=self.client)

        cos_yaw = math.cos(box_yaw_rad_world)
        sin_yaw = math.sin(box_yaw_rad_world)
        R_local_to_world = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Transform fixed local push direction to current world push direction
        current_push_direction_world = R_local_to_world @ self.fixed_local_push_direction
        
        force_left_val = left_thrust_scale * self.max_force_per_pusher
        force_right_val = right_thrust_scale * self.max_force_per_pusher

        applied_force1_3d = np.array([current_push_direction_world[0] * force_left_val,
                                      current_push_direction_world[1] * force_left_val, 0.0])
        applied_force2_3d = np.array([current_push_direction_world[0] * force_right_val,
                                      current_push_direction_world[1] * force_right_val, 0.0])
        
        # Transform fixed local contact point offsets to current world CoM offsets
        cp1_world_offset = R_local_to_world @ self.cp1_local_fixed
        cp2_world_offset = R_local_to_world @ self.cp2_local_fixed
        
        force_app_pos1_w = np.array([box_pos_w[0] + cp1_world_offset[0], 
                                     box_pos_w[1] + cp1_world_offset[1], 
                                     box_pos_w[2]]) # Apply at CoM height
        force_app_pos2_w = np.array([box_pos_w[0] + cp2_world_offset[0], 
                                     box_pos_w[1] + cp2_world_offset[1], 
                                     box_pos_w[2]])

        # apply forces in world frame
        p.applyExternalForce(self.box_id, -1, applied_force1_3d, force_app_pos1_w, p.WORLD_FRAME, physicsClientId=self.client)
        p.applyExternalForce(self.box_id, -1, applied_force2_3d, force_app_pos2_w, p.WORLD_FRAME, physicsClientId=self.client)

        # step simulation
        for _ in range(self.sim_steps):
            p.stepSimulation(physicsClientId=self.client)
            if self.gui: time.sleep(self.dt / self.sim_steps)

        self.steps += 1
        
        obs = self._get_obs()
        current_pos_xy_world = obs[2:4] + self.initial_box_pos_xy # Reconstruct current world position from displacement

        delta_disp_vector = current_pos_xy_world - self.prev_box_pos_xy
        self.prev_box_pos_xy = current_pos_xy_world.copy()

        # compute reward
        reward = 0
        vec_to_goal_from_prev_pos = self.goal_pos_world - (current_pos_xy_world - delta_disp_vector)
        norm_vec_to_goal = np.linalg.norm(vec_to_goal_from_prev_pos)
        if norm_vec_to_goal > 1e-6:
            unit_vec_to_goal = vec_to_goal_from_prev_pos / norm_vec_to_goal
            reward += np.dot(delta_disp_vector, unit_vec_to_goal)
        
        terminated = False
        distance_to_goal_val = np.linalg.norm(current_pos_xy_world - self.goal_pos_world)
        if distance_to_goal_val < self.goal_thresh:
            reward += 100
            terminated = True
            # print("Goal reached at step {}.".format(self.steps))

        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                 reward -= 0.1 # Smaller penalty for timeout
                # print("Max steps reached. Dist: {:.2f}".format(distance_to_goal_val))
        
        # updating GUI helpers
        if self.gui:
            # Simplified puck visualization at contact points
            puck_z = 0.015 
            p.resetBasePositionAndOrientation(self.bot1_vis, [force_app_pos1_w[0], force_app_pos1_w[1], puck_z], [0,0,0,1], physicsClientId=self.client)
            p.resetBasePositionAndOrientation(self.bot2_vis, [force_app_pos2_w[0], force_app_pos2_w[1], puck_z], [0,0,0,1], physicsClientId=self.client)
            
            # Simplified arrow viz (just direction, not perfect placement yet)
            def _update_arrow_viz(arrow_parts, app_point_world_3d, force_vec_3d_val, s_len, h_len):
                shaft_id, head_id = arrow_parts
                force_magnitude_2d = np.linalg.norm(force_vec_3d_val[:2])
                if force_magnitude_2d < 1e-2: # Threshold for drawing arrow
                    p.resetBasePositionAndOrientation(shaft_id, [0,0,-10], [0,0,0,1], physicsClientId=self.client)
                    p.resetBasePositionAndOrientation(head_id, [0,0,-10], [0,0,0,1], physicsClientId=self.client)
                    return
                
                force_dir_2d_unit = force_vec_3d_val[:2] / force_magnitude_2d
                yaw_arrow = math.atan2(force_dir_2d_unit[1], force_dir_2d_unit[0])
                quat_arrow = p.getQuaternionFromEuler([0,0,yaw_arrow])
                
                # Position arrows at the application points, pointing in force direction
                eff_shaft_center_world = app_point_world_3d + np.array([force_dir_2d_unit[0]*s_len/2, force_dir_2d_unit[1]*s_len/2, 0])
                eff_head_base_world  = app_point_world_3d + np.array([force_dir_2d_unit[0]*s_len, force_dir_2d_unit[1]*s_len, 0])

                p.resetBasePositionAndOrientation(shaft_id, [eff_shaft_center_world[0], eff_shaft_center_world[1], app_point_world_3d[2]], quat_arrow, physicsClientId=self.client)
                p.resetBasePositionAndOrientation(head_id, [eff_head_base_world[0], eff_head_base_world[1], app_point_world_3d[2]], quat_arrow, physicsClientId=self.client)

            _update_arrow_viz((self.arrow1_shaft, self.arrow1_head), force_app_pos1_w, applied_force1_3d, self.arrow1_shaft_len, self.arrow1_head_len)
            _update_arrow_viz((self.arrow2_shaft, self.arrow2_head), force_app_pos2_w, applied_force2_3d, self.arrow2_shaft_len, self.arrow2_head_len)

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.client >=0 and p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)
            self.client = -1

    def render(self, mode='human'):
        if mode == 'rgb_array' and self.gui:
            box_pos_w, _ = p.getBasePositionAndOrientation(self.box_id, physicsClientId=self.client)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=box_pos_w, distance=1.5, yaw=45, pitch=-30, roll=0, upAxisIndex=2, physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(640)/480, nearVal=0.1, farVal=100.0, physicsClientId=self.client)
            _, _, rgba, _, _ = p.getCameraImage(
                640, 480, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client)
            rgb_array = np.array(rgba, dtype=np.uint8).reshape(480, 640, 4)
            return rgb_array[:, :, :3]
        elif mode == 'human' and self.gui: pass
        else:
            # Fallback for non-standard modes or if super().render() is needed
            # For basic gym.Env, this might not exist unless inherited from a specific wrapper
            if hasattr(super(), 'render'):
                 return super().render(mode=mode)
            return None # Or raise NotImplementedError


if __name__ == '__main__':
    # Use parameters from your config.yaml for consistency in testing
    env_params = {
        "gui": True, # Set to True for visual testing
        "max_force_per_pusher": 100.0, # From your config
        "friction": 0.5,
        "angular_damping_factor": 0.2, # From your config
        "goal_pos": np.array([2.0, 0.0]),
        "goal_thresh": 0.1,
        "max_steps": 200, # Shorter for testing individual scenarios
        "pusher_offset_distance": 0.15, # From your config
        "randomize_initial_yaw": True, # Test with randomization
        "initial_yaw_range": [-math.pi/4, math.pi/4] # e.g. +/- 45 degrees
    }
    env = PrismaticEnv(**env_params)
    
    print("Goal Position:", env.goal_pos_world)
    print("Box Half Extents (after reset):", env.box_half_extents) # Will be printed after first reset

    test_actions_scenarios = {
        "Both_Push_Forward_Hard": np.array([1.0, 1.0], dtype=np.float32),
        "Steer_Right (Left_More_Thrust)": np.array([1.0, 0.5], dtype=np.float32),
        "Steer_Left (Right_More_Thrust)": np.array([0.5, 1.0], dtype=np.float32),
        "One_Pushes_Left_Only": np.array([0.8, 0.0], dtype=np.float32),
        "One_Pushes_Right_Only": np.array([0.0, 0.8], dtype=np.float32),
    }

    for scenario_name, test_action in test_actions_scenarios.items():
        print(f"\n--- Testing Scenario: {scenario_name} ---")
        obs, info = env.reset() 
        print(f"Initial Obs (cos_yaw, sin_yaw, dx, dy, vx, vy, wz): {obs}")
        print(f"Initial Box Pos XY (world): {env.initial_box_pos_xy}, Initial Box Yaw (rad): {math.atan2(obs[1], obs[0]):.2f}")
        print(f"Fixed Local Push Direction: {env.fixed_local_push_direction}")
        print(f"Fixed CP1 Local: {env.cp1_local_fixed}, CP2 Local: {env.cp2_local_fixed}")

        total_reward_scenario = 0
        for i in range(env.max_steps): # Run for one full episode
            obs, reward, terminated, truncated, info = env.step(test_action)
            done = terminated or truncated
            
            if i % 30 == 0 or done or i == env.max_steps -1 :
                box_pos_w, box_orn_w_quat = p.getBasePositionAndOrientation(env.box_id, physicsClientId=env.client)
                _,_,current_yaw_rad = p.getEulerFromQuaternion(box_orn_w_quat)
                current_pos_xy_world = np.array(box_pos_w[:2])
                
                print("Step {:3d}: Act L({:.1f}) R({:.1f}) | Pos ({:.2f},{:.2f}) Yaw_deg {:.1f} | Disp ({:.2f},{:.2f}) | Rew {:.2f}".format(
                    i, test_action[0], test_action[1], 
                    current_pos_xy_world[0], current_pos_xy_world[1], math.degrees(current_yaw_rad),
                    obs[2], obs[3], 
                    reward))
            total_reward_scenario += reward
            if env.gui: time.sleep(1./60.)
            if done:
                if terminated and not truncated: print("Goal Reached!")
                elif truncated: print("Max steps reached.")
                break
        print(f"--- Scenario {scenario_name} End --- Total Reward: {total_reward_scenario:.3f}, Steps: {env.steps}")
        if env.gui: time.sleep(1)

    env.close()