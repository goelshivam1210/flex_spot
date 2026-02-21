"""
test_generalization.py — Test a trained TD3 policy on out-of-distribution path shapes.

Tests whether the short-segment trained policy can generalize to:
  - S-shaped paths (alternating curvature)
  - Meandering paths (multi-frequency lateral oscillation)

The policy is not retrained — this is pure inference on novel geometries.
The env's full_path and path_points are monkey-patched directly so nothing
else in the environment changes (physics, domain randomization, state, action
space all remain identical to training).

Usage:
    python test_generalization.py --run_dir runs/run-0-2024-01-01_00-00-00 --model best_model
    python test_generalization.py --run_dir runs/run-0-2024-01-01_00-00-00 --model best_model --num_episodes 5
"""

import os
import sys
import argparse
import yaml
import numpy as np
import imageio
import mujoco
from scipy.spatial.transform import Rotation

from env import SimplePathFollowingEnv
from td3 import TD3


# ---------------------------------------------------------------------------
# Path generators
# ---------------------------------------------------------------------------

def generate_s_path(length=3.0, amplitude=0.5, num_points=100):
    """
    S-shaped path: one full sine cycle over `length` meters forward.
    The box must turn left then right (or right then left) to follow it.
    Amplitude controls how wide the S is.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = (t / (2 * np.pi)) * length
    # y = np.sin(t) * amplitude
    y = np.sin(t) * amplitude + np.sin(1.5 * t) * amplitude * 0.2

    return np.stack([x, y], axis=1)


def generate_meander_path(length=4.0, amplitude=0.4, num_points=150):
    """
    Meandering path: superposition of two sine frequencies to create
    irregular, non-repeating curvature changes. More realistic than
    a clean S-curve — tests robustness to unpredictable turns.
    """
    t = np.linspace(0, 4 * np.pi, num_points)
    x = (t / (4 * np.pi)) * length
    y = np.sin(t) * amplitude + np.sin(2.3 * t) * amplitude * 0.4
    return np.stack([x, y], axis=1)

def generate_triple_s_path(length=4.5, amplitude=0.4, num_points=150):
    """
    Three consecutive S-curves — tests whether the policy can chain
    multiple direction reversals without accumulating drift.
    """
    t = np.linspace(0, 6 * np.pi, num_points)  # 3 full sine cycles
    x = (t / (6 * np.pi)) * length
    y = np.sin(t) * amplitude
    return np.stack([x, y], axis=1)

PATH_GENERATORS = {
    "s_curve":  generate_s_path,
    "meander":  generate_meander_path,
    "triple_s": generate_triple_s_path
}


# ---------------------------------------------------------------------------
# Path marker drawing (same as visualize.py)
# ---------------------------------------------------------------------------

def plot_path_markers(scene, path_points):
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


# ---------------------------------------------------------------------------
# Video Recorder (same as visualize.py)
# ---------------------------------------------------------------------------

class VideoRecorder:
    def __init__(self, model, data, output_path, width=1280, height=720, fps=40):
        self.model = model
        self.data  = data

        self.writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )

        if getattr(model.vis.global_, "offwidth", 0) < width:
            model.vis.global_.offwidth = int(width)
        if getattr(model.vis.global_, "offheight", 0) < height:
            model.vis.global_.offheight = int(height)

        self.renderer = mujoco.Renderer(model, height, width)

        self.cam            = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance   = 8.0
        self.cam.azimuth    = 90.0
        self.cam.elevation  = -35.0
        self.cam.lookat     = np.array([2.0, 0.0, 0.3], dtype=float)

    def capture(self, path_points):
        self.renderer.update_scene(self.data, camera=self.cam)
        if path_points is not None:
            plot_path_markers(self.renderer.scene, path_points)
        return self.renderer.render()

    def close(self):
        self.writer.close()
        self.renderer.close()


# ---------------------------------------------------------------------------
# Env reset with custom path
# ---------------------------------------------------------------------------

def reset_env_with_path(env, path_points, rng):
    """
    Reset the environment but override full_path and path_points with
    a custom geometry. Everything else (mass, friction, physics) resets
    normally so domain randomization still applies.
    """
    # Let env reset normally to randomize mass/friction and reset physics
    env.segment_length = None   # use the full custom path, no sub-sampling
    env.test_full_arc  = True   # prevent internal direction randomization
    env._reverse_path  = False

    # Override path before reset's _make_training_segment runs.
    # We inject directly after reset so the start pose is computed from our path.
    # We call super().reset() equivalent by temporarily replacing full_path.
    env.full_path = path_points.copy()

    # Manually do what reset() does after generating the path
    env.path_points = path_points.copy()

    # Domain randomization
    env.sampled_mass = rng.uniform(env.mass_range[0], env.mass_range[1])
    env.model.body_mass[env.box_body_id] = env.sampled_mass

    env.sampled_friction = rng.uniform(env.friction_range[0], env.friction_range[1])
    box_geom_id   = env.model.geom('box_geom').id
    floor_geom_id = env.model.geom('floor').id
    env.model.geom_friction[box_geom_id][0]   = env.sampled_friction
    env.model.geom_friction[floor_geom_id][0] = env.sampled_friction

    # Compute goal thresh
    seg_len = float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))
    env.goal_thresh = max(env.goal_thresh_pct * seg_len, env.goal_thresh_min)

    # Reset MuJoCo state
    mujoco.mj_resetData(env.model, env.data)

    # Place box at path start, oriented along first tangent
    start_pos = [path_points[0][0], path_points[0][1], 0.2]
    tangent   = path_points[1] - path_points[0]
    angle     = np.arctan2(tangent[1], tangent[0])
    quat      = Rotation.from_euler('xyz', [0, 0, angle]).as_quat()
    quat      = quat / np.linalg.norm(quat)

    qpos         = np.zeros(env.model.nq)
    qpos[0:3]    = start_pos
    qpos[3:7]    = [quat[3], quat[0], quat[1], quat[2]]
    env.data.qpos[:] = qpos
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    env.steps            = 0
    env.prev_position    = np.array(start_pos[:2])
    env.prev_time        = 0.0
    env.last_closest_idx = 0

    state, _ = env._get_state()
    return state


# ---------------------------------------------------------------------------
# Run one episode on a custom path
# ---------------------------------------------------------------------------

def run_episode(env, agent, path_points, recorder, max_steps, label, rng):
    state = reset_env_with_path(env, path_points, rng)

    frames       = []
    total_reward = 0.0
    deviations   = []

    for t in range(max_steps):
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)

        next_state, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        deviations.append(float(info["deviation"]))
        state = next_state

        frames.append(recorder.capture(path_points))

        if done or truncated:
            break

    success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)

    print(f"  [{label}] steps={t+1:4d}  progress={info['progress']:.3f}  "
          f"deviation={info['deviation']:.4f}  avg_dev={np.mean(deviations):.4f}  "
          f"mass={info['sampled_mass']:.1f}kg  friction={info['sampled_friction']:.2f}  "
          f"success={success}  terminal={info['terminal_event']}")

    for frame in frames:
        recorder.writer.append_data(frame)

    return {
        "success":     success,
        "steps":       t + 1,
        "reward":      total_reward,
        "progress":    info["progress"],
        "avg_dev":     float(np.mean(deviations)),
        "final_info":  info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test TD3 policy generalization on S-curve and meandering paths."
    )
    parser.add_argument("--run_dir",       type=str, required=True,
                        help="Path to the run directory")
    parser.add_argument("--model",         type=str, default="best_model",
                        help="Model name: best_model, final_model, etc.")
    parser.add_argument("--num_episodes",  type=int, default=5,
                        help="Episodes per path type (default: 5)")
    parser.add_argument("--max_steps",     type=int, default=1500,
                        help="Max steps per episode (default: 1500)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--width",         type=int, default=1280)
    parser.add_argument("--height",        type=int, default=720)
    parser.add_argument("--fps",           type=int, default=40)
    # S-curve params
    parser.add_argument("--s_length",      type=float, default=3.0,
                        help="Forward length of S-curve in metres (default: 3.0)")
    parser.add_argument("--s_amplitude",   type=float, default=0.5,
                        help="Lateral amplitude of S-curve in metres (default: 0.5)")
    # Meander params
    parser.add_argument("--m_length",      type=float, default=4.0,
                        help="Forward length of meander path in metres (default: 4.0)")
    parser.add_argument("--m_amplitude",   type=float, default=0.4,
                        help="Lateral amplitude of meander path in metres (default: 0.4)")
    args = parser.parse_args()

    run_dir     = os.path.abspath(args.run_dir)
    models_dir  = os.path.join(run_dir, "models")
    config_path = os.path.join(run_dir, "config.yaml")
    videos_dir  = os.path.join(run_dir, "videos", "generalization")
    os.makedirs(videos_dir, exist_ok=True)

    # Validate
    if not os.path.isdir(run_dir):
        print(f"ERROR: run_dir not found: {run_dir}"); sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found: {config_path}"); sys.exit(1)
    actor_file = os.path.join(models_dir, f"{args.model}_actor.pth")
    if not os.path.isfile(actor_file):
        print(f"ERROR: Model not found: {actor_file}"); sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    rng = np.random.default_rng(args.seed)

    print(f"Run dir : {run_dir}")
    print(f"Model   : {args.model}")
    print(f"Episodes per path: {args.num_episodes}")

    # Build env — gui=False, no segment sub-sampling, longer max_steps
    test_cfg                  = env_cfg.copy()
    test_cfg["gui"]           = False
    test_cfg["segment_length"] = None
    test_cfg["test_full_arc"] = True
    test_cfg["max_steps"]     = args.max_steps

    env = SimplePathFollowingEnv(**test_cfg)
    env.reset(seed=args.seed)

    # Load agent
    agent = TD3(
        lr=1e-3,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        max_torque=env.max_torque,
    )
    agent.load_actor(models_dir, args.model)

    # Path configs
    paths = {
        "s_curve": generate_s_path(
            length=args.s_length,
            amplitude=args.s_amplitude,
        ),
        "meander": generate_meander_path(
            length=args.m_length,
            amplitude=args.m_amplitude,
        ),
        "triple_s": generate_triple_s_path(
            length=args.s_length * 1.5,
            amplitude=args.s_amplitude * 0.8,
        )
    }

    # Run episodes for each path type
    all_results = {}
    for path_name, path_points in paths.items():
        print(f"\n{'='*60}")
        print(f"PATH: {path_name.upper()}  "
              f"({len(path_points)} points, "
              f"length≈{np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)):.2f}m)")
        print(f"{'='*60}")

        results   = []
        successes = 0

        for ep in range(args.num_episodes):
            out_path = os.path.join(
                videos_dir,
                f"{path_name}_{args.model}_ep{ep+1:02d}.mp4"
            )
            recorder = VideoRecorder(
                env.model, env.data, out_path,
                width=args.width, height=args.height, fps=args.fps
            )

            result = run_episode(
                env, agent, path_points, recorder,
                max_steps=args.max_steps,
                label=f"{path_name.upper()} ep{ep+1}",
                rng=rng
            )
            recorder.close()
            results.append(result)

            if result["success"]:
                successes += 1

            print(f"    → saved: {out_path}")

        # Summary for this path type
        success_rate = successes / args.num_episodes
        avg_progress = np.mean([r["progress"] for r in results])
        avg_dev      = np.mean([r["avg_dev"]   for r in results])
        avg_reward   = np.mean([r["reward"]    for r in results])

        all_results[path_name] = {
            "success_rate": success_rate,
            "avg_progress": avg_progress,
            "avg_deviation": avg_dev,
            "avg_reward":   avg_reward,
        }

        print(f"\n  [{path_name.upper()}] SUMMARY")
        print(f"    Success rate : {success_rate:.2f}  ({successes}/{args.num_episodes})")
        print(f"    Avg progress : {avg_progress:.3f}")
        print(f"    Avg deviation: {avg_dev:.4f}m")
        print(f"    Avg reward   : {avg_reward:.2f}")

    # Final summary across all path types
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for path_name, stats in all_results.items():
        print(f"  {path_name.upper():12s}  "
              f"success={stats['success_rate']:.2f}  "
              f"progress={stats['avg_progress']:.3f}  "
              f"avg_dev={stats['avg_deviation']:.4f}m")

    print(f"\nVideos saved to: {videos_dir}")
    env.close()


if __name__ == "__main__":
    main()