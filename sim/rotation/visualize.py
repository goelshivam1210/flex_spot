"""
visualize.py — Record videos of a trained TD3 policy on short, full arc, and random arcs.

Usage:
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model best_model
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model converged_model
    python visualize.py --run_dir runs/run-0-2024-01-01_00-00-00 --model final_model

Records:
  - One successful episode for short segment
  - One successful episode for full arc
  - --num_random_arcs episodes of random arcs (one video per arc)

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

def record_first_success(env, agent, recorder, max_steps, max_attempts, label, alternate_direction=False):
    """
    Run up to max_attempts episodes. Record the first successful one.
    If no success found, saves the episode with the highest progress instead.

    BUG FIX #3: _reverse_path is explicitly reset to False after each env.reset()
    call so the flag never bleeds between attempts. Direction for the next attempt
    is always set *before* reset, and cleared *after* reset.

    If alternate_direction=True, alternates arc direction every other attempt
    without permanently mutating env arc params.

    Returns result dict: steps, reward, success, final_info.
    """
    best_frames   = None
    best_progress = -1.0
    best_result   = None

    # Store original arc params so we can restore them each attempt
    orig_start = env.arc_start
    orig_end   = env.arc_end

    for attempt in range(1, max_attempts + 1):
        # Restore arc params and set direction before reset
        env.arc_start = orig_start
        env.arc_end   = orig_end
        env._reverse_path = alternate_direction and (attempt % 2 == 0)

        state, _ = env.reset()

        # BUG FIX #3: clear the flag after reset so it cannot bleed further
        env._reverse_path = False

        # Keep path markers in sync with the current episode's segment
        recorder.path_points_to_draw = getattr(env, "path_points", None)

        frames = []
        total_reward = 0.0
        t = 0

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


def record_single_episode(env, agent, recorder, max_steps, label):
    """
    Run exactly one episode with the current env configuration and record it.
    BUG FIX #3: _reverse_path is cleared after reset to prevent flag bleed.
    Returns result dict: steps, reward, success, final_info.
    """
    state, _ = env.reset()

    # BUG FIX #3: clear after reset — caller sets _reverse_path before calling
    # this function; once reset() has consumed it, clear so it doesn't persist.
    env._reverse_path = False

    recorder.path_points_to_draw = getattr(env, "path_points", None)

    frames       = []
    total_reward = 0.0
    t = 0

    for t in range(max_steps):
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)

        next_state, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        state = next_state

        recorder.renderer.update_scene(recorder.data, camera=recorder.cam)
        if recorder.path_points_to_draw is not None:
            plot_path_markers(recorder.renderer.scene, recorder.path_points_to_draw)
        frames.append(recorder.renderer.render())

        if done or truncated:
            break

    success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)
    result  = {"steps": t + 1, "reward": total_reward, "success": success, "final_info": info}

    print(f"  [{label}] progress={info['progress']:.3f}  "
          f"deviation={info['deviation']:.4f}  "
          f"mass={info['sampled_mass']:.1f}kg  "
          f"friction={info['sampled_friction']:.2f}  "
          f"success={success}")

    for frame in frames:
        recorder.writer.append_data(frame)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record TD3 policy videos.")
    parser.add_argument("--run_dir",          type=str, required=True,
                        help="Path to the run directory")
    parser.add_argument("--model",            type=str, required=True,
                        help="Model name: best_model, converged_model, or final_model")
    parser.add_argument("--max_attempts",     type=int, default=50,
                        help="Max attempts for short/full arc success (default: 50)")
    parser.add_argument("--num_random_arcs",  type=int, default=5,
                        help="Number of random arc videos to record (default: 5)")
    parser.add_argument("--seed",             type=int, default=42,
                        help="Random seed for arc geometry sampling (default: 42)")
    parser.add_argument("--width",            type=int, default=1280)
    parser.add_argument("--height",           type=int, default=720)
    parser.add_argument("--fps",              type=int, default=40)
    parser.add_argument("--short_max_steps",  type=int, default=500)
    parser.add_argument("--full_max_steps",   type=int, default=1000)
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

    rng = np.random.default_rng(args.seed)

    # -------------------------------------------------------------------
    # SHORT SEGMENT
    # -------------------------------------------------------------------
    print("\n--- SHORT SEGMENT ---")
    short_cfg = env_cfg.copy()
    short_cfg["gui"]            = False
    short_cfg["segment_length"] = env_cfg.get("short_segment_length", 0.3)
    short_cfg["max_steps"]      = args.short_max_steps

    short_env   = SimplePathFollowingEnv(**short_cfg)
    short_env.reset(seed=args.seed)
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
        label="SHORT",
        alternate_direction=True   # test both directions during short-seg search
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
    full_env.reset(seed=args.seed)
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
        label="FULL",
        alternate_direction=True
    )
    full_rec.close()
    full_env.close()
    print(f"[FULL]  steps={full_res['steps']}  reward={full_res['reward']:.2f}  "
          f"success={full_res['success']}  → {full_out}")

    # -------------------------------------------------------------------
    # RANDOM ARCS
    # -------------------------------------------------------------------
    if args.num_random_arcs > 0:
        print(f"\n--- RANDOM ARCS ({args.num_random_arcs} episodes) ---")

        gen_cfg = env_cfg.copy()
        gen_cfg["gui"]            = False
        gen_cfg["segment_length"] = None
        gen_cfg["test_full_arc"]  = True
        gen_cfg["max_steps"]      = args.full_max_steps

        gen_env   = SimplePathFollowingEnv(**gen_cfg)
        gen_env.reset(seed=args.seed + 100)
        gen_agent = load_agent(models_dir, args.model, gen_env)

        successes = 0
        for i in range(args.num_random_arcs):
            # Sample random arc geometry
            r      = rng.uniform(1.0, 2.0)
            theta0 = rng.uniform(-np.pi / 2, 0)
            theta1 = rng.uniform(0, np.pi / 2)
            flipped = rng.random() > 0.5
            direction = "rev" if flipped else "fwd"

            # Set arc params and direction before reset; reset() consumes _reverse_path
            gen_env.arc_radius     = r
            gen_env.arc_start      = theta0
            gen_env.arc_end        = theta1
            gen_env.segment_length = None
            gen_env.test_full_arc  = True
            gen_env._reverse_path  = flipped  # consumed by reset()

            label   = f"GEN_{i+1:02d}"
            tmp_out = os.path.join(videos_dir, f"_tmp_gen_{i+1:02d}.mp4")
            gen_rec = VideoRecorder(
                gen_env.model, gen_env.data, tmp_out,
                width=args.width, height=args.height, fps=args.fps
            )

            gen_res = record_single_episode(
                gen_env, gen_agent, gen_rec,
                max_steps=args.full_max_steps,
                label=label
            )
            gen_rec.close()

            # Rename with all params baked into filename
            mass        = gen_res["final_info"]["sampled_mass"]
            friction    = gen_res["final_info"]["sampled_friction"]
            success_tag = "ok" if gen_res["success"] else "fail"
            final_out = os.path.join(
                videos_dir,
                f"gen_{i+1:02d}_{args.model}"
                f"_r{r:.2f}"
                f"_t{np.degrees(theta0):.0f}to{np.degrees(theta1):.0f}"
                f"_{direction}"
                f"_m{mass:.1f}_f{friction:.2f}"
                f"_{success_tag}"
                f".mp4"
            )
            os.rename(tmp_out, final_out)

            if gen_res["success"]:
                successes += 1

            print(f"  [{label}] → {final_out}")

        gen_env.close()
        print(f"\n[GEN] {successes}/{args.num_random_arcs} successful  "
              f"(seed={args.seed}, {args.num_random_arcs} arcs)")

    print(f"\nDone. Videos saved to: {videos_dir}")


if __name__ == "__main__":
    main()