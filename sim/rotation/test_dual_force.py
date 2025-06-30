#!/usr/bin/env python3
"""
Dual Force Application Test Script with Policy Stitching

Tests how well the centroid-to-contact force transformation works with stitched policies:

1. For each short segment:
   - Get centroid forces from micro-policy
   - Transform to contact forces using _wrench_to_contact_forces
   - Compare performance between direct centroid application vs transformed contact forces

2. For full arc (stitched policies):
   - Get centroid forces from sequence of micro-policies
   - Transform each policy's output to contact forces
   - Compare if the transformation works as well with stitched policies as with individual ones

This validates that the mathematical transformation from centroid to contact forces
works correctly both for individual micro-policies and when stitching them together.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
import yaml
import contextlib
from scipy.spatial.transform import Rotation

from env_mujoco import SimplePathFollowingEnv
from td3 import TD3

class DualForceTestEnv(SimplePathFollowingEnv):
    """Extended environment that can apply forces in two different ways"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_body_id = self.model.body("box").id
        self.force_mode = "centroid"  # "centroid" or "contact"
        self.contact_points = None
        self.applied_contact_forces = None
        self.current_segment = 0  # Track which segment we're on for stitching
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
        """Setup two contact points on the rear face at ground level"""
        # Box is scaled by 0.4, so dimensions are ~0.4x0.4x0.4
        # Contact points on rear face left and right sides at ground level
        offset_x = -0.2   # Behind box center (rear face)
        offset_y = 0.2    # Left and right edges (wider spacing)
        offset_z = 0.0   # Ground level (bottom of box)
        # offset_z = -0.02
        
        self.contact_points = np.array([
            [offset_x, +offset_y, offset_z],  # Left rear contact at ground level
            [offset_x, -offset_y, offset_z]   # Right rear contact at ground level
        ])
        print(f"Contact points setup: {self.contact_points}")
    
    def _wrench_to_contact_forces(self, wrench):
        """
        Convert centroid wrench [Fx, Fy, τz] to contact forces using pseudo-inverse method
        
        Args:
            wrench: [Fx, Fy, τz] - force and torque at centroid
            
        Returns:
            contact_forces: (2, 3) array of forces at each contact point
        """
        # For 2D case, we have Fx, Fy, τz
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
        
        # Reshape to contact forces (2 contacts, 3D forces each)
        contact_forces_2d = forces_flat.reshape(2, 2)  # [[f1x, f1y], [f2x, f2y]]
        contact_forces = np.zeros((2, 3))
        contact_forces[:, :2] = contact_forces_2d  # Copy x,y forces
        contact_forces[:, 2] = 0  # z forces are zero
        
        return contact_forces
    
    def step(self, action):
        """Override step to apply forces based on current mode"""
        self.steps += 1
        
        # Parse action to wrench
        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        torque_z = np.clip(action[2], -1, 1) * self.max_torque
        
        wrench = np.array([force_x, force_y, torque_z])
        
        # Get state before applying forces
        state_before = self._get_state()
        
        # Apply forces based on mode
        for _ in range(self.sim_steps):
            # Check for numerical instability
            if np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel)):
                print("Warning: Physics instability detected. Resetting velocity.")
                self.data.qvel[:] = 0
                break

            # Apply forces based on mode
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
        reward = self._calculate_reward(state_before, state_after)
        
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
            "applied_forces": getattr(self, 'applied_contact_forces', None)

        }
        
        if self.steps % 20 == 0:
            print(f"Step {self.steps}: Mode={self.force_mode}, Progress={progress:.3f}, "
                  f"Deviation={deviation:.3f}, Wrench=[{force_x:.1f}, {force_y:.1f}, {torque_z:.1f}]")
        
        return state_after, reward, done, truncated, info
    
    # def _apply_centroid_forces(self, force_x, force_y, torque_z):
    #     """Apply forces at centroid (original method)"""
    #     force_3d = [force_x, force_y, 0]
    #     p.applyExternalForce(
    #         self.box_id, -1, force_3d, [0, 0, 0], p.LINK_FRAME
    #     )
    #     p.applyExternalTorque(
    #         self.box_id, -1, [0, 0, torque_z], p.LINK_FRAME
    #     )
    
    # def _apply_contact_forces(self, wrench, pos, quat):
    #     """Apply distributed contact forces (sim-to-real method)"""
    #     # Convert wrench to contact forces
    #     contact_forces = self._wrench_to_contact_forces(wrench)
    #     self.applied_contact_forces = contact_forces.copy()
        
    #     # Get box orientation
    #     euler = p.getEulerFromQuaternion(quat)
    #     box_angle = euler[2]
        
    #     # Transform contact points from box frame to world frame
    #     cos_a, sin_a = np.cos(box_angle), np.sin(box_angle)
    #     rotation_matrix = np.array([
    #         [cos_a, -sin_a, 0],
    #         [sin_a,  cos_a, 0],
    #         [0,      0,     1]
    #     ])
        
    #     for i, (contact_point, force) in enumerate(zip(self.contact_points, contact_forces)):
    #         # Transform contact point to world frame
    #         world_contact_point = rotation_matrix @ contact_point
            
    #         # Apply force at contact point in world frame
    #         # Transform force from box frame to world frame
    #         world_force = rotation_matrix @ force
            
    #         p.applyExternalForce(
    #             self.box_id, -1, 
    #             force.tolist(),           # Box frame forces
    #             contact_point.tolist(),   # Box frame positions  
    #             p.LINK_FRAME             # Box frame application
    #         )
    
    # def _draw_debug_info(self, pos, quat, force_x, force_y, torque_z):
    #     """Enhanced debug visualization showing both force modes"""
    #     # Call parent method for basic visualization
    #     super()._draw_debug_info(pos, quat, force_x, force_y, torque_z)
        
    #     if self.force_mode == "contact" and self.contact_points is not None:
    #         # Draw contact points and forces
    #         euler = p.getEulerFromQuaternion(quat)
    #         box_angle = euler[2]
            
    #         cos_a, sin_a = np.cos(box_angle), np.sin(box_angle)
    #         rotation_matrix = np.array([
    #             [cos_a, -sin_a, 0],
    #             [sin_a,  cos_a, 0],
    #             [0,      0,     1]
    #         ])
            
    #         for i, contact_point in enumerate(self.contact_points):
    #             # Transform contact point to world frame
    #             world_contact_point = rotation_matrix @ contact_point
    #             world_contact_pos = np.array(pos) + world_contact_point
                
    #             # Draw contact point (yellow sphere)
    #             p.addUserDebugLine(
    #                 world_contact_pos,
    #                 world_contact_pos + [0, 0, 0.05],
    #                 [1, 1, 0], 5, 0.1
    #             )
                
    #             # Draw contact force vector if available
    #             if self.applied_contact_forces is not None:
    #                 force = self.applied_contact_forces[i]
    #                 world_force = rotation_matrix @ force
    #                 force_magnitude = np.linalg.norm(world_force)
                    
    #                 if force_magnitude > 1e-3:
    #                     # Scale force for visualization
    #                     force_scale = 0.5 / max(self.max_force, 1)
    #                     force_end = world_contact_pos + world_force * force_scale
                        
    #                     # Draw force vector (cyan for contact forces)
    #                     p.addUserDebugLine(
    #                         world_contact_pos,
    #                         force_end,
    #                         [0, 1, 1], 4, 0.1
    #                     )
                        
    #                     # Add force magnitude text
    #                     p.addUserDebugText(
    #                         f"F{i+1}: {force_magnitude:.1f}N",
    #                         world_contact_pos + [0, 0, 0.1],
    #                         textColorRGB=[0, 1, 1],
    #                         textSize=0.8,
    #                         lifeTime=0.1
    #                     )

def load_trained_model(model_path, env):
    """Load trained TD3 model"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_torque = env.max_torque
    
    agent = TD3(
        lr=1e-3,  # Not used for inference
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        max_torque=max_torque
    )
    
    # Load only the actor for inference
    agent.load_actor(os.path.dirname(model_path), os.path.basename(model_path).replace('_actor.pth', ''))
    return agent


def test_episode(env, agent, mode, max_steps=500, render=False):
    """Run a single test episode"""
    env.set_force_mode(mode)
    state, _ = env.reset()
    
    total_reward = 0
    step_count = 0
    segment_rewards = []  # Track rewards per segment
    current_segment = 0

    viewer = env.enable_rendering()
    if viewer is None:
        print("Failed to create viewer!")
        return None
    
    print(f"\n=== Testing in {mode.upper()} mode ===")

    time.sleep(2.0)

    state, _ = env.reset()
    
    for step in range(max_steps):
        if render and viewer and not viewer.is_running(): 
            break
        # Get action from trained policy
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        
        # Step environment
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
        
        if render:
            viewer.sync()
        
        if done or truncated:
            break
    
    # Add final segment reward
    segment_rewards.append(total_reward - sum(segment_rewards))
    
    print(f"Episode finished: Steps={step_count}, Total Reward={total_reward:.2f}")
    print(f"Final Progress: {info['progress']:.3f}, Final Deviation: {info['deviation']:.3f}")
    
    success = info['progress'] > 0.95 and info['deviation'] < env.goal_thresh
    print(f"Success: {'YES' if success else 'NO'}")

    print("Episode complete. Viewer will stay open for 3 seconds...")
    for i in range(3):
        if viewer.is_running():
            viewer.sync()
            time.sleep(1.0)
        else:
            break
    
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
    parser = argparse.ArgumentParser(description="Test force transformation with stitched policies")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model (e.g., runs/run-0-2024.../models/best_model)")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of test episodes per mode")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", 
                        help="Render the episodes in a GUI window.")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    env_cfg = config["env"]
    
    # Create test environments for both short segments and full arcs
    env_short = DualForceTestEnv(
        gui=True,  # Always use GUI for testing
        max_force=env_cfg.get("max_force", 300.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        max_steps=args.max_steps,
        segment_length=env_cfg.get("segment_length", 0.3)  # Use same segment length as training
    )
    
    env_full = DualForceTestEnv(
        gui=True,  # Always use GUI for testing
        max_force=env_cfg.get("max_force", 300.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        max_steps=args.max_steps * 2,  # Longer for full arc
        segment_length=None  # No segment length for full arc testing
    )
    
    # Load trained model
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
                
                result = test_episode(env, agent, mode, args.max_steps)
                mode_results.append(result)
                
                # Add sleep for visualization
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
    print(f"FINAL COMPARISON")
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


if __name__ == "__main__":
    main()