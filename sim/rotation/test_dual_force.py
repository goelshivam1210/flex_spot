#!/usr/bin/env python3
"""
Dual Force Application Test Script (Full Arc Only)

Tests a trained TD3 policy on a full arc path in two modes:
1. Single centroid force application (current method)
2. Distributed contact forces on rear face (sim-to-real method)

The arc path is always drawn in the GUI, and the box's trajectory is visualized as it follows the arc.
"""

import os
import numpy as np
import pandas as pd
import imageio
import mujoco
import mujoco.viewer
import time
import argparse
import yaml
from scipy.spatial.transform import Rotation

from env import SimplePathFollowingEnv
from td3 import TD3


import imageio
import mujoco


class VideoRecorder:
    """
    A class for offscreen video recording of a MuJoCo simulation.
    """
    def __init__(self, model, data, output_path, width=800, height=600, fps=30, path_points=None):
        self.model = model
        self.data = data
        self.path_points_to_draw = path_points
        self.num_path_markers = 0
        self.writer = imageio.get_writer(output_path, fps=fps)
        self.renderer = mujoco.Renderer(model, height, width)

        # Create a dedicated MjvCamera instance for recording
        self.cam = mujoco.MjvCamera()
        
        # Set camera properties for a zoomed-out, angled view
        self.cam.distance = 7.5
        self.cam.azimuth = 90.0
        self.cam.elevation = -30.0
        
        # Set the camera to look at a point in the center of the scene
        self.cam.lookat = np.array([0.0, 0.0, 0.5])

    def capture(self):
        """
        Captures a single frame from the simulation using the custom camera.
        """
        # Tell the renderer to use our specific camera when updating the scene
        self.renderer.update_scene(self.data, camera=self.cam)
        self.renderer.scene.ngeom -= self.num_path_markers

        # Now, add the path markers to the scene
        if self.path_points_to_draw is not None:
            plot_path_markers(self.renderer.scene, self.path_points_to_draw)
        
        # Render the complete scene (robot + markers) and save to video
        pixels = self.renderer.render()
        self.writer.append_data(pixels)

    def close(self):
        """
        Closes the video writer and cleans up the renderer.
        """
        self.writer.close()
        self.renderer.close()

class DualForceTestEnv(SimplePathFollowingEnv):
    """Extended environment that can apply forces in two different ways"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_body_id = self.model.body("box").id
        self.force_mode = "centroid"  # "centroid" or "contact"
        self.contact_points = None
        self.applied_contact_forces = None
        self.current_segment = 0
        self.viewer = None
        self.render_enabled = False
    
    def reset(self, *args, **kwargs):
        self.current_segment = 0
        return super().reset(*args, **kwargs)
    
    def enable_rendering(self):
        """Enable rendering with proper viewer setup"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render_enabled = True
            
            # Set up camera for better view
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            
            print("MuJoCo viewer launched successfully!")
            print("Camera controls:")
            print("- Mouse drag: Rotate view")
            print("- Mouse wheel: Zoom")
            print("- Double-click: Focus on object")
            
        return self.viewer
    
    def render(self):
        """Render the current state"""
        if self.render_enabled and self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.016)  # ~60 FPS
            else:
                print("Viewer was closed")
                self.render_enabled = False
                self.viewer = None
    
    def close_viewer(self):
        """Close the viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            self.render_enabled = False
    

    def set_force_mode(self, mode):
        """Set force application mode: 'centroid' or 'contact'"""
        assert mode in ["centroid", "contact"], f"Invalid mode: {mode}"
        self.force_mode = mode
        print(f"Force mode set to: {mode}")
        if mode == "contact":
            self._setup_contact_points()
    
    def _setup_contact_points(self):
        offset_x = -0.2
        offset_y = 0.2
        offset_z = 0.0
        self.contact_points = np.array([
            [offset_x, +offset_y, offset_z],
            [offset_x, -offset_y, offset_z]
        ])
        print(f"Contact points setup: {self.contact_points}")
    
    def _wrench_to_contact_forces(self, wrench):
        Fx, Fy, tau_z = wrench[0], wrench[1], wrench[2]
        torque_cap = 2 * self.max_force * abs(self.contact_points[0,1])  # 2 pads, arm = |offset_y|
        tau_z = np.clip(tau_z, -torque_cap, torque_cap)
        
        # Build equilibrium matrix A for 2 contact points
        # We have 3 equilibrium equations and 4 unknowns (2 contacts × 2 forces each)
        # A * [f1x, f1y, f2x, f2y]^T = [Fx, Fy, τz]^T
        
        A = np.zeros((3, 4))  # 3 equilibrium equations, 4 force components
        
        # Force balance equations:
        # f1x + f2x = Fx
        # f1y + f2y = Fy
        A[0, 0] = 1.0  # f1x coefficient for Fx equation
        A[0, 2] = 1.0  # f2x coefficient for Fx equation
        A[1, 1] = 1.0  # f1y coefficient for Fy equation  
        A[1, 3] = 1.0  # f2y coefficient for Fy equation
        
        # Moment balance equation about centroid:
        # τz = r1 × f1 + r2 × f2
        # For 2D: τz = (r1x * f1y - r1y * f1x) + (r2x * f2y - r2y * f2x)
        r1 = self.contact_points[0]  # [offset_x, +offset_y, offset_z]
        r2 = self.contact_points[1]  # [offset_x, -offset_y, offset_z]
        
        # τz = r1x*f1y - r1y*f1x + r2x*f2y - r2y*f2x
        A[2, 0] = -r1[1]  # -r1y * f1x
        A[2, 1] = +r1[0]  # +r1x * f1y
        A[2, 2] = -r2[1]  # -r2y * f2x  
        A[2, 3] = +r2[0]  # +r2x * f2y
        
        # Solve using pseudo-inverse
        wrench_2d = np.array([Fx, Fy, tau_z])
        forces_flat = np.linalg.pinv(A) @ wrench_2d
        contact_forces_2d = forces_flat.reshape(2, 2)
        contact_forces = np.zeros((2, 3))
        contact_forces[:, :2] = contact_forces_2d  # Copy x,y forces
        contact_forces[:, 2] = 0  # z forces are zero
        
        return contact_forces
    
    def step(self, action):
        self.steps += 1
        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        torque_z = np.clip(action[2], -1, 1) * self.max_torque
        wrench = np.array([force_x, force_y, torque_z])
        state_before = self._get_state()
        for _ in range(self.sim_steps):
            # Check for numerical instability
            if np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel)):
                print("Warning: Physics instability detected. Resetting velocity.")
                self.data.qvel[:] = 0
                break
            if self.force_mode == "centroid":
                box_quat = self.data.body('box').xquat
                rot_matrix = Rotation.from_quat(box_quat[[1,2,3,0]]).as_matrix()
                force_world = rot_matrix @ np.array([wrench[0], wrench[1], 0])
                torque_world = rot_matrix @ np.array([0, 0, wrench[2]])
                self.data.xfrc_applied[self.box_body_id] = np.concatenate([force_world, torque_world])
            else:  # contact mode
                contact_forces_local = self._wrench_to_contact_forces(wrench)
                max_pad_force = np.max(np.linalg.norm(contact_forces_local, axis=1))
                if max_pad_force > self.max_force:
                    contact_forces_local *= self.max_force / max_pad_force

                self.applied_contact_forces = contact_forces_local.copy()

                box_pos  = self.data.body('box').xpos.copy()
                box_quat = self.data.body('box').xquat
                rot_mat  = Rotation.from_quat(box_quat[[1, 2, 3, 0]]).as_matrix()

                self.data.xfrc_applied[:] = 0
                self.data.qfrc_applied[:] = 0

                for i in range(2):
                    # point / force in world frame
                    world_pt    = box_pos + rot_mat @ self.contact_points[i]
                    world_force = rot_mat @ contact_forces_local[i]

                    mujoco.mj_applyFT(
                        self.model,
                        self.data,
                        world_force.reshape(3,1),
                        np.zeros((3,1)),
                        world_pt.reshape(3,1),
                        self.box_body_id,
                        self.data.qfrc_applied
                    )
            
            mujoco.mj_step(self.model, self.data)
            # Render if enabled
            if self.render_enabled:
                self.render()
        
        # Continue with rest of step method (same as parent)
        state_after = self._get_state()
        angular_velocity_z = self.data.qvel[5]
        reward, reward_comps = self._calculate_reward(state_before, state_after, angular_velocity_z)
        
        progress = state_after[3]
        deviation = state_after[4]
        
        # Check if we need to switch to next segment
        if self.segment_length is not None and progress > 0.95:
            self.current_segment += 1
            print(f"Switching to segment {self.current_segment}")
        
        done = False
        if progress > 0.95 and deviation < self.goal_thresh:
            done = True
            reward += self.goal_reward
            print(f"Goal reached at step {self.steps}! Mode: {self.force_mode}")
        
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        info = {
            "steps": self.steps,
            "progress": progress,
            "deviation": deviation,
            "orientation_error": state_after[2],
            "lateral_error": state_after[0],
            "longitudinal_error": state_after[1],
            "speed_along_path": state_after[5],
            "box_forward_x": state_after[6],
            "box_forward_y": state_after[7],
            "force_mode": self.force_mode,
            "current_segment": self.current_segment,
            "applied_wrench": wrench,
            "applied_forces": getattr(self, 'applied_contact_forces', None),
            "reward_comps": reward_comps,
        }
        
        if self.steps % 20 == 0:
            print(f"Step {self.steps}: Mode={self.force_mode}, Progress={progress:.3f}, "
                  f"Deviation={deviation:.3f}, Wrench=[{force_x:.1f}, {force_y:.1f}, {torque_z:.1f}]")
        
        return state_after, reward, done, truncated, info

def load_trained_model(model_path, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_torque = env.max_torque
    agent = TD3(
        lr=1e-3,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        max_torque=max_torque
    )
    agent.load_actor(os.path.dirname(model_path), os.path.basename(model_path).replace('_actor.pth', ''))
    return agent

def plot_path_markers(scene, path_points):
    """Adds markers to a given MjvScene to visualize the path."""
    z_height = 0.02
    for point in path_points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.015, 0, 0]),
            pos=np.array([point[0], point[1], z_height]),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 0.5])
        )
        scene.ngeom += 1

def test_episode(env, agent, mode, max_steps=500, render=False, recorder=None):
    """Run a single test episode"""
    env.set_force_mode(mode)
    state, _ = env.reset()

    # --- Data Logging Setup ---
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    debug_data = []
    # --------------------------
    
    total_reward = 0
    step_count = 0
    segment_rewards = []  # Track rewards per segment
    current_segment = 0

    print(f"\n=== Testing in {mode.upper()} mode ===")

    viewer = None
    if render:
        viewer = env.enable_rendering()
        if viewer is None:
            print("Failed to create viewer!")
            return None
        time.sleep(1.0) # Give viewer time to initialize

    time.sleep(2.0)

    state, _ = env.reset()

    viewer_is_active = False 
    if render:
        # env.viewer is the persistent object we need to manage
        if env.enable_rendering():
             viewer_is_active = True
        else:
            print("Failed to create viewer!")
            return None
        time.sleep(1.0) 

    
    for step in range(max_steps):
        if render and viewer and not viewer.is_running(): 
            break
        # Get action from trained policy
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track segment rewards
        if info['current_segment'] != current_segment:
            segment_rewards.append(total_reward)
            current_segment = info['current_segment']
            print(f"Completed segment {current_segment-1} with reward {segment_rewards[-1]:.2f}")
        
        # Print progress periodically
        if step_count % 50 == 0:
            print(f"Step {step_count}: Progress={info['progress']:.3f}, "
                f"Deviation={info['deviation']:.3f}, Reward={reward:.2f}")
            
        # --- Log data for this step ---
        log_entry = {
            'step': step_count,
            'progress': info['progress'],
            'deviation': info['deviation'],
            'lat_err': info['lateral_error'],
            'orient_err': info['orientation_error'],
            'action_fx': action[0],
            'action_fy': action[1],
            'action_tz': action[2],
            'reward': reward
        }
        debug_data.append(log_entry)
        # -----------------------------
        
        box_pos = env.data.body('box').xpos[:2]

        # Find the index of the closest point on the path
        dists = np.linalg.norm(env.path_points - box_pos, axis=1)
        closest_idx = np.argmin(dists)
        closest_path_point = env.path_points[closest_idx]

        print(f"Step {step_count:3d}: "
            f"Box Position=({box_pos[0]:.2f}, {box_pos[1]:.2f}) | "
            f"Closest Path Point (idx {closest_idx})=({closest_path_point[0]:.2f}, {closest_path_point[1]:.2f}) | "
            f"Deviation={info['deviation']:.3f}")
        if render:
            viewer.sync()
            time.sleep(0.01)

        if recorder:
            recorder.capture()

        if done or truncated:
            break
    
    # Add final segment reward
    segment_rewards.append(total_reward - sum(segment_rewards))

    print("\n--- Episode Log ---")
    df = pd.DataFrame(debug_data)
    print(df.to_string())
    print("-" * 20)
    
    print(f"Episode finished: Steps={step_count}, Total Reward={total_reward:.2f}")
    print(f"Final Progress: {info['progress']:.3f}, Final Deviation: {info['deviation']:.3f}")
    success = info['progress'] > 0.95 and info['deviation'] < env.goal_thresh
    print(f"Success: {'YES' if success else 'NO'}")

    if render and viewer_is_active:
        print("Episode complete. Closing viewer...")
        env.close_viewer()
    
    return {
        'total_reward': total_reward,
        'steps': step_count,
        'success': success,
        'final_progress': info['progress'],
        'final_deviation': info['deviation'],
        'segment_rewards': segment_rewards,
        'num_segments': len(segment_rewards)
    }

def main():
    parser = argparse.ArgumentParser(description="Test force application methods on a full arc path")

    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model (e.g., runs/run-0-2024.../models/best_model)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of test episodes per mode")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", 
                        help="Render the episodes in a GUI window.")
    parser.add_argument("--record", type=str, default=None,
                help="If set, record offscreen to this mp4 file")

    
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config["env"]
    
    # Create test environments for both short segments and full arcs
    env_short = DualForceTestEnv(
        gui=args.render,
        max_force=env_cfg.get("max_force", 400.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        max_steps=args.max_steps,
        segment_length=env_cfg.get("segment_length", 0.3)  # Use same segment length as training
    )
    
    env_full = DualForceTestEnv(
        gui=args.render,
        max_force=env_cfg.get("max_force", 400.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        max_steps=args.max_steps * 2,  # Longer for full arc
        segment_length=None  # No segment length for full arc testing
    )
    print(f"Loading model from: {args.model_path}")
    agent = load_trained_model(args.model_path, env_short)
    print("Model loaded successfully!")

    # Test both modes on both environments
    modes = ["centroid", "contact"]
    envs = {
        "short": env_short,
        "full": env_full
    }
    
    results = {}
    
    for env_name, env in envs.items():
        print(f"\n{'='*50}")
        print(f"TESTING {env_name.upper()} SEGMENTS")
        print(f"{'='*50}")
        
        env_results = {}
        
        for mode in modes:
            print(f"\n--- Testing {mode.upper()} mode on {env_name} segments ---")
            
            mode_results = []
            
            for episode in range(args.episodes):
                print(f"\nEpisode {episode + 1}/{args.episodes}")
                recorder = None
                if args.record:
                    # Create a unique filename for each episode
                    base_path, ext = os.path.splitext(args.record)
                    output_path = f"{base_path}_{env_name}_{mode}_ep{episode + 1}{ext}"
                    print(f"Recording episode to {output_path}...")
                    recorder = VideoRecorder(
                        env.model,
                        env.data,
                        output_path,
                        width=800, height=600, fps=40,
                        path_points=env.path_points
                    )
                result = test_episode(env, agent, mode, args.max_steps, render=args.render, recorder=recorder)
                mode_results.append(result)
                if recorder:
                    recorder.close()
                
                # Add sleep for visualization
                if args.render:
                    time.sleep(1.0)
            
            # Calculate statistics
            avg_reward = np.mean([r['total_reward'] for r in mode_results])
            success_rate = np.mean([r['success'] for r in mode_results])
            avg_steps = np.mean([r['steps'] for r in mode_results])
            
            # Calculate per-segment statistics
            avg_segments = np.mean([r['num_segments'] for r in mode_results])
            avg_segment_rewards = np.mean([np.mean(r['segment_rewards']) for r in mode_results])
            
            env_results[mode] = {
                'episodes': mode_results,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_segments': avg_segments,
                'avg_segment_reward': avg_segment_rewards
            }
            
            print(f"\n{mode.upper()} MODE SUMMARY ({env_name}):")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Success Rate: {success_rate:.2f}")
            print(f"Average Steps: {avg_steps:.1f}")
            print(f"Average Segments: {avg_segments:.1f}")
            print(f"Average Segment Reward: {avg_segment_rewards:.2f}")
        
        results[env_name] = env_results
    
    # Final comparison
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    
    for env_name in envs.keys():
        print(f"\n{env_name.upper()} SEGMENTS:")
        for mode in modes:
            r = results[env_name][mode]
            print(f"{mode.upper():>10}: Reward={r['avg_reward']:6.1f}, "
                  f"Success={r['success_rate']:4.2f}, Steps={r['avg_steps']:5.1f}, "
                  f"Segments={r['avg_segments']:4.1f}, SegReward={r['avg_segment_reward']:6.1f}")
    
    # Check if performance is similar between modes for each environment type
    for env_name in envs.keys():
        reward_diff = abs(results[env_name]['centroid']['avg_reward'] - 
                         results[env_name]['contact']['avg_reward'])
        success_diff = abs(results[env_name]['centroid']['success_rate'] - 
                          results[env_name]['contact']['success_rate'])
        segment_reward_diff = abs(results[env_name]['centroid']['avg_segment_reward'] - 
                                results[env_name]['contact']['avg_segment_reward'])
        
        print(f"\nPerformance Difference ({env_name}):")
        print(f"Total reward difference: {reward_diff:.2f}")
        print(f"Success rate difference: {success_diff:.2f}")
        print(f"Per-segment reward difference: {segment_reward_diff:.2f}")
        
        if reward_diff < 50 and success_diff < 0.2 and segment_reward_diff < 20:
            print(f"GOOD: Force transformation works well for {env_name} segments")
        else:
            print(f"WARNING: Force transformation may not be working well for {env_name} segments")
    
    env_short.close()
    env_full.close()
    if recorder:
        recorder.close()
        print(f"Wrote video to {args.record}")

if __name__ == "__main__":
    main()