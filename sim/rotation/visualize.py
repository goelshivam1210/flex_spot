"""
visualize.py — Record videos of a trained TD3 policy on short and full arc.

Usage:
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model best_model
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model converged_model
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model final_model

Records one successful episode each for short segment and full arc.
If no success is found within --max_attempts, saves the best episode seen.
Videos are saved to <run_dir>/videos/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import imageio
import mujoco

from env import SimplePathFollowingEnv
from td3 import TD3


# ---------------------------------------------------------------------------
# Path marker drawing
# ---------------------------------------------------------------------------

def plot_path_markers(scene, path_points):
    """Draw red spheres along the path directly into MuJoCo's scene geometry."""
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
# Video Recorder
# ---------------------------------------------------------------------------

class VideoRecorder:
    def __init__(self, model, data, output_path, width=1280, height=720, fps=40, path_points=None):
        self.model = model
        self.data = data
        self.path_points_to_draw = path_points

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

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 7.5
        self.cam.azimuth = 90.0
        self.cam.elevation = -30.0
        self.cam.lookat = np.array([0.0, 0.0, 0.5], dtype=float)

    def close(self):
        self.writer.close()
        self.renderer.close()


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(models_dir, model_name, env):
    """Load only the actor weights — sufficient for inference."""
    agent = TD3(
        lr=1e-3,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        max_torque=env.max_torque,
    )
    agent.load_actor(models_dir, model_name)
    return agent


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def record_first_success(env, agent, recorder, max_steps, max_attempts, label):
    """
    Run up to max_attempts episodes. Record the first successful one.
    If no success found, saves the episode with the highest progress instead.
    Returns result dict: steps, reward, success, final_info.
    """
    best_frames   = None
    best_progress = -1.0
    best_result   = None

    for attempt in range(1, max_attempts + 1):
        state, _ = env.reset()

        # Keep path markers in sync with the current episode's segment
        recorder.path_points_to_draw = getattr(env, "path_points", None)

        frames = []
        total_reward = 0.0

        for t in range(max_steps):
            action = agent.select_action(np.array(state))
            if action.ndim > 1:
                action = action.squeeze(0)

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            state = next_state

            # Render and capture frame
            recorder.renderer.update_scene(recorder.data, camera=recorder.cam)
            if recorder.path_points_to_draw is not None:
                plot_path_markers(recorder.renderer.scene, recorder.path_points_to_draw)
            frames.append(recorder.renderer.render())

            if done or truncated:
                break

        success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)
        result  = {"steps": t + 1, "reward": total_reward, "success": success, "final_info": info}

        print(f"  [{label}] attempt {attempt:3d}: progress={info['progress']:.3f}  "
              f"deviation={info['deviation']:.4f}  goal_thresh={info['goal_thresh']:.4f}  "
              f"success={success}")

        # Always track the best episode seen so far
        if info["progress"] > best_progress:
            best_progress = info["progress"]
            best_frames   = frames
            best_result   = result

        if success:
            print(f"  [{label}] ✓ Success on attempt {attempt}. Writing video.")
            best_frames = frames
            best_result = result
            break

    if not best_result["success"]:
        print(f"  [{label}] ✗ No success in {max_attempts} attempts. "
              f"Saving best episode (progress={best_progress:.3f}).")

    for frame in best_frames:
        recorder.writer.append_data(frame)

    return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record TD3 policy videos for short and full arc.")
    parser.add_argument("--run_dir",      type=str, required=True,
                        help="Path to the run directory")
    parser.add_argument("--model",        type=str, required=True,
                        help="Model name: best_model, converged_model, or final_model")
    parser.add_argument("--max_attempts", type=int, default=50,
                        help="Max episodes to attempt before giving up (default: 50)")
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--fps",          type=int, default=40)
    parser.add_argument("--short_max_steps", type=int, default=500)
    parser.add_argument("--full_max_steps",  type=int, default=1000)
    args = parser.parse_args()

    run_dir     = os.path.abspath(args.run_dir)
    models_dir  = os.path.join(run_dir, "models")
    config_path = os.path.join(run_dir, "config.yaml")
    videos_dir  = os.path.join(run_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Validate inputs
    if not os.path.isdir(run_dir):
        print(f"ERROR: run_dir not found: {run_dir}")
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found: {config_path}")
        sys.exit(1)
    actor_file = os.path.join(models_dir, f"{args.model}_actor.pth")
    if not os.path.isfile(actor_file):
        print(f"ERROR: Model file not found: {actor_file}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    print(f"Run dir : {run_dir}")
    print(f"Model   : {args.model}")

    # -------------------------------------------------------------------
    # SHORT SEGMENT
    # -------------------------------------------------------------------
    print("\n--- SHORT SEGMENT ---")
    short_cfg = env_cfg.copy()
    short_cfg["gui"]            = False
    short_cfg["segment_length"] = env_cfg.get("short_segment_length", 0.3)
    short_cfg["max_steps"]      = args.short_max_steps

    short_env   = SimplePathFollowingEnv(**short_cfg)
    short_env.reset(seed=42)
    short_agent = load_agent(models_dir, args.model, short_env)

    short_out = os.path.join(videos_dir, f"short_{args.model}.mp4")
    short_rec = VideoRecorder(
        short_env.model, short_env.data, short_out,
        width=args.width, height=args.height, fps=args.fps,
        path_points=getattr(short_env, "path_points", None)
    )

    short_res = record_first_success(
        short_env, short_agent, short_rec,
        max_steps=args.short_max_steps,
        max_attempts=args.max_attempts,
        label="SHORT"
    )
    short_rec.close()
    short_env.close()
    print(f"[SHORT] steps={short_res['steps']}  reward={short_res['reward']:.2f}  "
          f"success={short_res['success']}  → {short_out}")

    # -------------------------------------------------------------------
    # FULL ARC
    # -------------------------------------------------------------------
    print("\n--- FULL ARC ---")
    full_cfg = env_cfg.copy()
    full_cfg["gui"]            = False
    full_cfg["segment_length"] = None
    full_cfg["test_full_arc"]  = True
    full_cfg["max_steps"]      = args.full_max_steps

    full_env   = SimplePathFollowingEnv(**full_cfg)
    full_env.reset(seed=42)
    full_agent = load_agent(models_dir, args.model, full_env)

    full_out = os.path.join(videos_dir, f"full_{args.model}.mp4")
    full_rec = VideoRecorder(
        full_env.model, full_env.data, full_out,
        width=args.width, height=args.height, fps=args.fps,
        path_points=getattr(full_env, "path_points", None)
    )

    full_res = record_first_success(
        full_env, full_agent, full_rec,
        max_steps=args.full_max_steps,
        max_attempts=args.max_attempts,
        label="FULL"
    )
    full_rec.close()
    full_env.close()
    print(f"[FULL]  steps={full_res['steps']}  reward={full_res['reward']:.2f}  "
          f"success={full_res['success']}  → {full_out}")

    print(f"\nDone. Videos saved to: {videos_dir}")


if __name__ == "__main__":
    main()