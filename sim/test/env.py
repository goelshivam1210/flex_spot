# path_following_env.py

import gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from gym import spaces

class SimplePathFollowingEnv(gym.Env):
    """
    Simple PyBullet environment for a box following an arc path in 2D.
    
    State: [position_error(2), orientation_error(1), progress(1), deviation(1)]
    Action: [force_direction(2), force_magnitude(1)]
    """
    def __init__(
        self, 
        gui=False, 
        max_force=20.0,
        friction=0.5,
        linear_damping=0.9,
        angular_damping=0.9,
        goal_thresh=0.2, 
        max_steps=500,
        goal_reward = 100,
        seed = None
    ):
        super(SimplePathFollowingEnv, self).__init__()
        
        # Environment parameters
        self.gui = gui
        self.max_force = max_force
        self.friction = friction
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.goal_thresh = goal_thresh
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.seed = seed

        # Simulation parameters
        self.dt = 1.0 / 240.0
        self.sim_steps = 6  # Small number of steps for stability
        self.angle_bin_size = 10.0 # degrees
        self.num_bins = int(360/self.angle_bin_size)
        
        # Define observation space
        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0]),
        #     high=np.array([np.inf, np.inf, np.pi, 1.0, np.inf]), 
        #     dtype=np.float32
        # )

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0]),  # Keep as radians but discretized
            high=np.array([np.inf, np.inf, np.pi, 1.0, np.inf]), 
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Connect to PyBullet
        self._connect()
        
        # Generate arc path points
        self.path_points = self._generate_arc_path()
        
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
    
    def _generate_arc_path(self):
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
    
    def reset(self):
        """Reset the environment to its initial state"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create a box at the start of the path
        start_pos = [self.path_points[0][0], self.path_points[0][1], 0.1]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        
        # Using a default cube URDF with scaling
        self.box_id = p.loadURDF("cube.urdf", start_pos, start_ori, globalScaling=0.3)
        p.changeDynamics(
            self.box_id, -1, 
            lateralFriction=self.friction, 
            angularDamping=self.angular_damping,
            linearDamping=self.linear_damping,
            rollingFriction=0.1,
            spinningFriction=0.1,
            restitution=0.1
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
        
        # Reset step counter
        self.steps = 0
        
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
        """Get the current state of the environment"""
        # Get box position and orientation
        pos, quat = p.getBasePositionAndOrientation(self.box_id)
        pos = np.array(pos[:2])  # x, y position
        
        # Convert quaternion to yaw angle (around z-axis)
        euler = p.getEulerFromQuaternion(quat)
        orientation = euler[2]  # yaw angle
        
        # Find closest point on the path
        dists = np.linalg.norm(self.path_points - pos, axis=1)
        closest_idx = np.argmin(dists)
        
        # Calculate position error
        closest_point = self.path_points[closest_idx]
        position_error = pos - closest_point
        
        # Calculate deviation from the path
        deviation = dists[closest_idx]
        
        # Calculate progress along the path
        progress = closest_idx / (len(self.path_points) - 1)
        
        # Calculate desired orientation (tangent at the closest path point)
        next_idx = min(closest_idx + 1, len(self.path_points) - 1)
        tangent = self.path_points[next_idx] - self.path_points[closest_idx]
        desired_orientation = np.arctan2(tangent[1], tangent[0])
        
        # Calculate orientation error
        orientation_error = np.arctan2(
            np.sin(orientation - desired_orientation),
            np.cos(orientation - desired_orientation)
        )

        # Add discretization for orientation_error 
        angle_bin_size = 10.0  # degrees (can be adjusted)
        num_bins = int(360 / angle_bin_size)
        
        # Convert orientation_error to discretized bin index
        bin_index = int(((orientation_error + np.pi) * 180/np.pi) / angle_bin_size) % num_bins
        
        # Optional: You can either return the bin_index directly, or convert back to radians
        discretized_orientation_error = (bin_index * angle_bin_size * np.pi/180) - np.pi
        
        # Return state vector with discretized orientation error
        state = np.array([
            position_error[0],
            position_error[1],
            discretized_orientation_error,  # or bin_index if you prefer the integer
            progress,
            deviation
        ], dtype=np.float32)
        
        # # Return state vector: [position_error, orientation_error, progress, deviation]
        # state = np.array([
        #     position_error[0],
        #     position_error[1],
        #     orientation_error,
        #     progress,
        #     deviation
        # ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute an action in the environment."""
        self.steps += 1
        
        # Parse action
        force_dir = np.array(action[:2])
        force_magnitude = np.clip(action[2], 0, 1)
        
        # Normalize force direction
        norm = np.linalg.norm(force_dir)
        if norm > 0 and norm != np.inf:
            force_dir = force_dir / norm
        else:
            # Default to no force if direction invalid
            force_dir = np.array([0.0, 0.0])
        
        # Get state before applying force
        state_before = self._get_state()
        
        # Apply force to the box for multiple steps
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
            if vel_magnitude > 5.0:  # Maximum reasonable velocity
                scale_factor = 5.0 / vel_magnitude
                p.resetBaseVelocity(
                    self.box_id, 
                    [vel[0] * scale_factor, vel[1] * scale_factor, vel[2] * scale_factor], 
                    [0, 0, 0]
                )
            
            # Apply force
            force = force_dir * force_magnitude * self.max_force
            force_3d = [force[0], force[1], 0]
            
            p.applyExternalForce(
                self.box_id, 
                -1,
                force_3d, 
                pos,
                p.WORLD_FRAME
            )
            
            # Step simulation
            p.stepSimulation()
            
            if self.gui:
                # Get current position and orientation
                current_pos, current_quat = p.getBasePositionAndOrientation(self.box_id)
                pos_xy = np.array(current_pos[:2])
                
                # Find closest point on path
                dists = np.linalg.norm(self.path_points - pos_xy, axis=1)
                closest_idx = np.argmin(dists)
                closest_point = self.path_points[closest_idx]
                
                # Draw current path connection (red line showing deviation)
                p.addUserDebugLine(
                    [closest_point[0], closest_point[1], 0.05],
                    [current_pos[0], current_pos[1], 0.05],
                    [1, 0, 0], 1, 0.1  # Red color, line width, lifetime
                )
                
                # Get orientation information
                current_angle = p.getEulerFromQuaternion(current_quat)[2]
                
                # Calculate desired orientation (path tangent)
                next_idx = min(closest_idx + 1, len(self.path_points) - 1)
                tangent = self.path_points[next_idx] - self.path_points[closest_idx]
                desired_angle = np.arctan2(tangent[1], tangent[0])
                
                # Draw current orientation vector (red)
                current_end = [
                    current_pos[0] + 0.3 * np.cos(current_angle),
                    current_pos[1] + 0.3 * np.sin(current_angle),
                    0.05
                ]
                p.addUserDebugLine(
                    [current_pos[0], current_pos[1], 0.05],
                    current_end,
                    [1, 0, 0],  # Red
                    2,
                    0.1
                )
                
                # Draw desired orientation vector (blue)
                desired_end = [
                    current_pos[0] + 0.3 * np.cos(desired_angle),
                    current_pos[1] + 0.3 * np.sin(desired_angle),
                    0.05
                ]
                p.addUserDebugLine(
                    [current_pos[0], current_pos[1], 0.05],
                    desired_end,
                    [0, 0, 1],  # Blue
                    2,
                    0.1
                )
                
                # Draw applied force vector (green)
                if np.linalg.norm(force_dir) > 0:
                    force_end = [
                        current_pos[0] + 0.4 * force_dir[0],
                        current_pos[1] + 0.4 * force_dir[1],
                        0.05
                    ]
                    p.addUserDebugLine(
                        [current_pos[0], current_pos[1], 0.05],
                        force_end,
                        [0, 1, 0],  # Green
                        2,
                        0.1
                    )
        
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
        
        # Info dict
        info = {
            "steps": self.steps,
            "progress": progress,
            "deviation": deviation,
            "orientation_error": state_after[2],
        }
        
        # Display state information
        if self.gui and self.steps % 10 == 0:
            # Create a detailed state display
            state_text = (
                f"Position Error: ({state_after[0]:.2f}, {state_after[1]:.2f})\n"
                f"Orient Error: {state_after[2]:.2f}\n"
                f"Progress: {progress:.2f}\n"
                f"Deviation: {deviation:.2f}\n"
                f"Reward: {reward:.2f}"
            )
            
            # Add text at the top of the screen
            p.addUserDebugText(
                state_text,
                [0, 0, 1],  # Position above the scene
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                lifeTime=0.5
            )
        
        return state_after, reward, done, truncated, info
    
    def _calculate_reward(self, state_before, state_after):
        """Calculate a simplified reward based on progress and deviation"""
        # Extract relevant state components
        progress_before = state_before[3]
        progress_after = state_after[3]
        deviation = state_after[4]
        
        # Simple progress reward (direct, not clipped)
        progress_reward = 5.0 * (progress_after - progress_before)
        
        # Simple deviation penalty (linear)
        deviation_penalty = -2.0 * deviation
        
        # Combine rewards
        total_reward = progress_reward + deviation_penalty
        
        # Goal completion bonus (keep this part)
        if progress_after > 0.95 and deviation < self.goal_thresh:
            total_reward += 10.0
        
        return float(total_reward)

    
    def close(self):
        """Close the environment"""
        p.disconnect()