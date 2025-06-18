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
import pybullet as p
import time
import argparse
import yaml
from env import SimplePathFollowingEnv
from td3 import TD3

class DualForceTestEnv(SimplePathFollowingEnv):
    """Extended environment that can apply forces in two different ways"""
    
    def set_force_mode(self, mode):
        """Set force application mode: 'centroid' or 'contact'"""
        assert mode in ["centroid", "contact"], f"Invalid mode: {mode}"
        self.force_mode = mode
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
    
    def _wrench_to_contact_forces(self, wrench):
        Fx, Fy, tau_z = wrench[0], wrench[1], wrench[2]
        A = np.zeros((3, 4))
        A[0, 0] = 1.0
        A[0, 2] = 1.0
        A[1, 1] = 1.0
        A[1, 3] = 1.0
        r1 = self.contact_points[0]
        r2 = self.contact_points[1]
        A[2, 0] = -r1[1]
        A[2, 1] = +r1[0]
        A[2, 2] = -r2[1]
        A[2, 3] = +r2[0]
        wrench_2d = np.array([Fx, Fy, tau_z])
        forces_flat = np.linalg.pinv(A) @ wrench_2d
        contact_forces_2d = forces_flat.reshape(2, 2)
        contact_forces = np.zeros((2, 3))
        contact_forces[:, :2] = contact_forces_2d
        contact_forces[:, 2] = 0
        return contact_forces
    
    def step(self, action):
        self.steps += 1
        force_x = np.clip(action[0], -1, 1) * self.max_force
        force_y = np.clip(action[1], -1, 1) * self.max_force
        torque_z = np.clip(action[2], -1, 1) * self.max_torque
        wrench = np.array([force_x, force_y, torque_z])
        state_before = self._get_state()
        for _ in range(self.sim_steps):
            pos, quat = p.getBasePositionAndOrientation(self.box_id)
            vel, ang_vel = p.getBaseVelocity(self.box_id)
            if (np.any(np.isnan(pos)) or np.any(np.isinf(pos)) or
                np.any(np.isnan(vel)) or np.any(np.isinf(vel))):
                print("Warning: Physics instability detected. Resetting object velocity.")
                p.resetBaseVelocity(self.box_id, [0, 0, 0], [0, 0, 0])
                break
            if self.force_mode == "centroid":
                self._apply_centroid_forces(force_x, force_y, torque_z)
            else:
                self._apply_contact_forces(wrench, pos, quat)
            p.stepSimulation()
            if self.gui:
                self._draw_debug_info(pos, quat, force_x, force_y, torque_z)
        state_after = self._get_state()
        reward = self._calculate_reward(state_before, state_after)
        progress = state_after[3]
        deviation = state_after[4]
        done = False
        if progress > 0.95 and deviation < self.goal_thresh:
            done = True
            reward += self.goal_reward
            if self.gui:
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
            "force_mode": self.force_mode
        }
        if self.gui and self.steps % 10 == 0:
            self._display_state_info(state_after, reward)
        return state_after, reward, done, truncated, info
    
    def _apply_centroid_forces(self, force_x, force_y, torque_z):
        force_3d = [force_x, force_y, 0]
        p.applyExternalForce(
            self.box_id, -1, force_3d, [0, 0, 0], p.LINK_FRAME
        )
        p.applyExternalTorque(
            self.box_id, -1, [0, 0, torque_z], p.LINK_FRAME
        )
    
    def _apply_contact_forces(self, wrench, pos, quat):
        contact_forces = self._wrench_to_contact_forces(wrench)
        self.applied_contact_forces = contact_forces.copy()
        euler = p.getEulerFromQuaternion(quat)
        box_angle = euler[2]
        cos_a, sin_a = np.cos(box_angle), np.sin(box_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        for i, (contact_point, force) in enumerate(zip(self.contact_points, contact_forces)):
            world_contact_point = rotation_matrix @ contact_point
            world_force = rotation_matrix @ force
            p.applyExternalForce(
                self.box_id, -1, 
                force.tolist(),
                contact_point.tolist(),
                p.LINK_FRAME
            )
    
    def _draw_debug_info(self, pos, quat, force_x, force_y, torque_z):
        super()._draw_debug_info(pos, quat, force_x, force_y, torque_z)
        if self.force_mode == "contact" and self.contact_points is not None:
            euler = p.getEulerFromQuaternion(quat)
            box_angle = euler[2]
            cos_a, sin_a = np.cos(box_angle), np.sin(box_angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            for i, contact_point in enumerate(self.contact_points):
                world_contact_point = rotation_matrix @ contact_point
                world_contact_pos = np.array(pos) + world_contact_point
                p.addUserDebugLine(
                    world_contact_pos,
                    world_contact_pos + [0, 0, 0.05],
                    [1, 1, 0], 5, 0.1
                )
                if self.applied_contact_forces is not None:
                    force = self.applied_contact_forces[i]
                    world_force = rotation_matrix @ force
                    force_magnitude = np.linalg.norm(world_force)
                    if force_magnitude > 1e-3:
                        force_scale = 0.5 / max(self.max_force, 1)
                        force_end = world_contact_pos + world_force * force_scale
                        p.addUserDebugLine(
                            world_contact_pos,
                            force_end,
                            [0, 1, 1], 4, 0.1
                        )
                        p.addUserDebugText(
                            f"F{i+1}: {force_magnitude:.1f}N",
                            world_contact_pos + [0, 0, 0.1],
                            textColorRGB=[0, 1, 1],
                            textSize=0.8,
                            lifeTime=0.1
                        )

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

def test_episode(env, agent, mode, max_steps=500, render_speed=0.02):
    env.set_force_mode(mode)
    state, _ = env.reset()
    total_reward = 0
    step_count = 0
    print(f"\n=== Testing in {mode.upper()} mode (full arc) ===")
    while step_count < max_steps:
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if step_count % 50 == 0:
            print(f"Step {step_count}: Progress={info['progress']:.3f}, "
                  f"Deviation={info['deviation']:.3f}, Reward={reward:.2f}")
        if done or truncated:
            break
        if env.gui:
            time.sleep(render_speed)
    print(f"Episode finished: Steps={step_count}, Total Reward={total_reward:.2f}")
    print(f"Final Progress: {info['progress']:.3f}, Final Deviation: {info['deviation']:.3f}")
    success = info['progress'] > 0.95 and info['deviation'] < env.goal_thresh
    print(f"Success: {'YES' if success else 'NO'}")
    return {
        'total_reward': total_reward,
        'steps': step_count,
        'success': success,
        'final_progress': info['progress'],
        'final_deviation': info['deviation']
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
    parser.add_argument("--render_speed", type=float, default=0.02,
                       help="Sleep time between steps for visualization")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config["env"]
    # Always use full arc for testing
    env = DualForceTestEnv(
        gui=True,
        max_force=env_cfg.get("max_force", 300.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        max_steps=args.max_steps,
        segment_length=None,  # Use full arc
        test_full_arc=True,   # Always test on full arc
        arc_radius=env_cfg.get("arc_radius", 1.5),
        arc_start=env_cfg.get("arc_start", -np.pi/3),
        arc_end=env_cfg.get("arc_end", np.pi/3)
    )
    print(f"Loading model from: {args.model_path}")
    agent = load_trained_model(args.model_path, env)
    print("Model loaded successfully!")
    modes = ["centroid", "contact"]
    results = {}
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"TESTING {mode.upper()} FORCE MODE ON FULL ARC")
        print(f"{'='*50}")
        mode_results = []
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            result = test_episode(env, agent, mode, args.max_steps, args.render_speed)
            mode_results.append(result)
            time.sleep(1.0)
        avg_reward = np.mean([r['total_reward'] for r in mode_results])
        success_rate = np.mean([r['success'] for r in mode_results])
        avg_steps = np.mean([r['steps'] for r in mode_results])
        results[mode] = {
            'episodes': mode_results,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps': avg_steps
        }
        print(f"\n{mode.upper()} MODE SUMMARY (FULL ARC):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
    print(f"\n{'='*50}")
    print(f"FINAL COMPARISON (FULL ARC)")
    print(f"{'='*50}")
    for mode in modes:
        r = results[mode]
        print(f"{mode.upper():>10}: Reward={r['avg_reward']:6.1f}, "
              f"Success={r['success_rate']:4.2f}, Steps={r['avg_steps']:5.1f}")
    reward_diff = abs(results['centroid']['avg_reward'] - results['contact']['avg_reward'])
    success_diff = abs(results['centroid']['success_rate'] - results['contact']['success_rate'])
    print(f"\nPerformance Difference (FULL ARC):")
    print(f"Total reward difference: {reward_diff:.2f}")
    print(f"Success rate difference: {success_diff:.2f}")
    if reward_diff < 50 and success_diff < 0.2:
        print(f"GOOD: Force transformation works well for full arc")
    else:
        print(f"WARNING: Force transformation may not be working well for full arc")
    env.close()

if __name__ == "__main__":
    main()