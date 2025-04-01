import gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class PrismaticEnv(gym.Env):
    """
    Gym environment for prismatic object manipulation in 2D using PyBullet.
    
    The goal is to move a box from an initial position to a goal position
    by applying force at the object's centroid. The object moves like a prismatic
    joint, i.e. along a single axis that can change over time.
    
    State:
      [joint_axis (2D), displacement (2D)]
      
      - joint_axis: a 2D unit vector indicating the current force application axis.
      - displacement: difference between current box centroid and initial position.
    
    Action:
      [force_direction (2D), force_scale (scalar ∈ [0,1])]
      
      The applied force is computed as:
          applied_force = normalize(force_direction) * force_scale * max_force
      
    Reward:
      A dense reward based on the effective displacement in the direction of
      the applied force, with an additional large reward when the box reaches the goal.
    """
    def __init__(self, gui=False, max_force=200.0, friction=0.5, 
                 goal_pos=np.array([2.0, 0.0]), goal_thresh=0.1, max_steps=2000):
        super(PrismaticEnv, self).__init__()
        self.gui = gui
        self.max_force = max_force       # η: maximum force
        self.friction = friction
        self.goal_pos = goal_pos
        self.goal_thresh = goal_thresh
        self.max_steps = max_steps
        self.dt = 1.0 / 240.0            # simulation time step
        self.sim_steps = 12              # number of simulation steps per env step

        # Define observation space: joint_axis (2) and displacement (2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Define action space: force_direction (2) and force_scale (1)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        self._connect()
        self.reset()

    def _connect(self):
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.8)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # Load the ground plane and the box (object)
        self.plane_id = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.1]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.box_id = p.loadURDF("cube.urdf", start_pos, start_ori, globalScaling=0.5)
        p.changeDynamics(self.box_id, -1, lateralFriction=self.friction, angularDamping=1)
        
        # Record the initial position (x,y) of the box
        self.initial_pos = np.array(p.getBasePositionAndOrientation(self.box_id)[0][:2])
        # Initialize the joint axis as [1, 0] (can be updated each step)
        self.joint_axis = np.array([1.0, 0.0])
        self.prev_disp = np.zeros(2)
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.box_id)
        current_pos = np.array(pos[:2])
        displacement = current_pos - self.initial_pos
        obs = np.concatenate([self.joint_axis, displacement])
        return obs.astype(np.float32)

    def step(self, action):
        # Parse action: first 2 for force direction, third for force scale.
        force_dir = np.array(action[:2])
        norm = np.linalg.norm(force_dir)
        if norm > 0:
            force_dir = force_dir / norm
        else:
            force_dir = self.joint_axis  # fallback if zero vector
        scale = np.clip(action[2], 0, 1)
        # Convert to a 3D force vector by appending 0 for the z-component.
        applied_force = np.array([force_dir[0], force_dir[1], 0.0]) * scale * self.max_force
        
        # Apply force at the object's centroid.
        pos, _ = p.getBasePositionAndOrientation(self.box_id)
        p.applyExternalForce(self.box_id, -1, applied_force.tolist(), pos, p.WORLD_FRAME)
        
        # Run simulation steps to emulate a discrete time step.
        for _ in range(self.sim_steps):
            p.stepSimulation()
            # Lock the object's rotation by resetting its orientation to identity.
            pos_current, _ = p.getBasePositionAndOrientation(self.box_id)
            p.resetBasePositionAndOrientation(self.box_id, pos_current, [0, 0, 0, 1])
            if self.gui:
                time.sleep(self.dt)
        
        self.steps += 1
        
        # Compute the new displacement and reward.
        pos_after, _ = p.getBasePositionAndOrientation(self.box_id)
        current_pos = np.array(pos_after[:2])
        displacement = current_pos - self.initial_pos
        delta_disp = displacement - self.prev_disp
        self.prev_disp = displacement.copy()
        # Reward is the effective displacement along the applied force direction.
        eff_disp = np.dot(delta_disp, force_dir)
        reward = eff_disp if eff_disp >= 0 else -eff_disp
        
        # Check if the goal is reached or max steps exceeded.
        done = False
        if np.linalg.norm(current_pos - self.goal_pos) < self.goal_thresh:
            reward += 100  # large reward on reaching the goal
            done = True
            print (f"Goal reached at step {self.steps} with displacement {displacement}.")
            # time.sleep(2)
        if self.steps >= self.max_steps:
            done = True

        # Update the joint axis based on the applied force direction.
        self.joint_axis = force_dir.copy()

        obs = self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def close(self):
        p.disconnect()

# For quick manual testing of the gym environment with keyboard controls.
if __name__ == '__main__':
    env = PrismaticEnv(gui=True)
    obs, _ = env.reset()  # unpack observation and info
    print("Initial observation:", obs)
    try:
        while True:
            keys = p.getKeyboardEvents()
            
            # If no arrow key is pressed, just print the observation and sleep.
            if not (p.B3G_LEFT_ARROW in keys or p.B3G_RIGHT_ARROW in keys or 
                    p.B3G_UP_ARROW in keys or p.B3G_DOWN_ARROW in keys):
                print("No arrow key pressed. Current observation:", obs)
                time.sleep(0.1)
                continue

            # Set default action: zero force direction, with force scale set high (which will be clipped)
            action = np.array([0.0, 0.0, 200.0])
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                action[0] = -1.0
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                action[0] = 1.0
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                action[1] = 1.0
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                action[1] = -1.0
            
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                break

            obs, reward, done, _, _ = env.step(action)
            print("Action:", action, "Obs:", obs, "Reward:", reward, "Done:", done)
            if done:
                obs, _ = env.reset()
    finally:
        env.close()