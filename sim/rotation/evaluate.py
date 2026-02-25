"""
evaluate.py — Systematic policy evaluation across mass/friction/direction conditions.

Tests a trained TD3 policy on:
  - Short segment and Full arc
  - Clockwise and Anticlockwise directions
  - In-distribution: (5kg, 15kg) x (0.4, 0.6 CoF)
  - Out-of-distribution: all combinations of 3kg/20kg x 0.5/0.8, plus 10kg+0.8

For each condition: up to max_attempts episodes, saves only the best video.
Per-step action CSV saved alongside each video.
Summary CSV and text report written to <run_dir>/eval/

Usage:
    python evaluate.py --run_dir runs/run-0-2024-01-01_00-00-00 --model best_model
"""

import os
import sys
import csv
import argparse
import yaml
import numpy as np
import imageio
import mujoco

from env import SimplePathFollowingEnv
from td3 import TD3


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

ID_CONDITIONS = [
    {"label": "low_mass_low_friction",   "mass": 5.0,  "friction": 0.4, "ood": False},
    {"label": "low_mass_high_friction",  "mass": 5.0,  "friction": 0.6, "ood": False},
    {"label": "high_mass_low_friction",  "mass": 15.0, "friction": 0.4, "ood": False},
    {"label": "high_mass_high_friction", "mass": 15.0, "friction": 0.6, "ood": False},
]

OOD_CONDITIONS = [
    {"label": "ood_3kg_std_friction",  "mass": 3.0,  "friction": 0.5, "ood": True},
    {"label": "ood_20kg_std_friction", "mass": 20.0, "friction": 0.5, "ood": True},
    {"label": "ood_std_mass_0.8_cof",  "mass": 10.0, "friction": 0.8, "ood": True},
    {"label": "ood_3kg_0.8_cof",       "mass": 3.0,  "friction": 0.8, "ood": True},
    {"label": "ood_20kg_0.8_cof",      "mass": 20.0, "friction": 0.8, "ood": True},
]

ALL_CONDITIONS = ID_CONDITIONS + OOD_CONDITIONS

DIRECTIONS = [
    {"label": "clockwise",     "reverse": False},
    {"label": "anticlockwise", "reverse": True},
]

ARC_TYPES = [
    {"label": "short", "segment_length": 0.3, "test_full_arc": False, "max_steps": 500},
    {"label": "full",  "segment_length": None, "test_full_arc": True,  "max_steps": 1000},
]


# ---------------------------------------------------------------------------
# Path marker drawing
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
# Video Recorder
# ---------------------------------------------------------------------------

class VideoRecorder:
    def __init__(self, model, data, output_path, width=1280, height=720, fps=40):
        self.model = model
        self.data = data
        self.path_points_to_draw = None

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

    def capture_frame(self):
        self.renderer.update_scene(self.data, camera=self.cam)
        if self.path_points_to_draw is not None:
            plot_path_markers(self.renderer.scene, self.path_points_to_draw)
        return self.renderer.render()

    def close(self):
        self.writer.close()
        self.renderer.close()


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(models_dir, model_name, env):
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
# Save action log CSV
# ---------------------------------------------------------------------------

def save_action_csv(action_log, path):
    if not action_log:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(action_log[0].keys()))
        writer.writeheader()
        writer.writerows(action_log)


# ---------------------------------------------------------------------------
# Episode runner: up to max_attempts, saves best video + action CSV
# ---------------------------------------------------------------------------

def run_condition(env, agent, output_path, max_steps, max_attempts, label,
                  width=1280, height=720, fps=40):
    """
    Run up to max_attempts episodes. Save the best (highest progress) video
    and a per-step action CSV alongside it.
    Returns a result dict with success flag and metrics.
    """
    best_frames     = None
    best_progress   = -1.0
    best_result     = None
    best_action_log = None

    # Precompute r_local — fixed geometry, doesn't change per step
    box_geom_id    = env.model.geom('box_geom').id
    half_extents   = env.model.geom_size[box_geom_id]
    corner_local   = np.array([half_extents[0], half_extents[1], 0.0])
    dist_to_corner = np.linalg.norm(corner_local)
    r_local = env.edge_heading_local * dist_to_corner if env.push_from_edge else np.zeros(3)

    for attempt in range(1, max_attempts + 1):
        state, _ = env.reset()
        env._reverse_path = False

        rec = VideoRecorder(env.model, env.data, output_path,
                            width=width, height=height, fps=fps)
        rec.path_points_to_draw = getattr(env, "path_points", None)

        frames      = []
        total_reward = 0.0
        action_log  = []
        t = 0

        for t in range(max_steps):
            action = agent.select_action(np.array(state))
            if action.ndim > 1:
                action = action.squeeze(0)

            # Compute what will be applied this step
            fx_raw = float(np.clip(action[0], -1, 1))
            fy_raw = float(np.clip(action[1], -1, 1))
            fx = fx_raw * env.max_force
            fy = fy_raw * env.max_force
            force_local = np.array([fx, fy, 0.0])

            if env.push_from_edge:
                torque_local = np.cross(r_local, force_local)
            else:
                tz_raw = float(np.clip(action[2], -1, 1)) if len(action) >= 3 else 0.0
                torque_local = np.array([0.0, 0.0, tz_raw * env.max_torque])

            # Per-step action log row
            action_log.append({
                "step":          t,
                "action_fx_raw": round(fx_raw, 5),
                "action_fy_raw": round(fy_raw, 5),
                "fx_N":          round(fx, 3),
                "fy_N":          round(fy, 3),
                "force_mag_N":   round(float(np.linalg.norm(force_local[:2])), 3),
                "r_local_x":     round(float(r_local[0]), 5),
                "r_local_y":     round(float(r_local[1]), 5),
                "torque_x":      round(float(torque_local[0]), 5),
                "torque_y":      round(float(torque_local[1]), 5),
                "torque_z":      round(float(torque_local[2]), 5),
                "lateral_err":   round(float(state[0]), 5),
                "orient_err":    round(float(state[1]), 5),
                "speed_fwd":     round(float(state[2]), 5),
                "speed_lat":     round(float(state[3]), 5),
                "angular_vel":   round(float(state[4]), 5),
                "f_react":       round(float(state[5]), 5),
            })

            # Print every 10 steps to avoid flooding terminal
            # if t % 10 == 0:
            #     print(f"    step {t:4d} | "
            #           f"fx={fx:+7.1f}N  fy={fy:+7.1f}N  "
            #           f"|F|={np.linalg.norm(force_local[:2]):.1f}N  "
            #           f"τz={torque_local[2]:+7.3f}  "
            #           f"r=[{r_local[0]:.3f},{r_local[1]:.3f}]  "
            #           f"lat_err={state[0]:+.4f}  "
            #           f"ori_err={state[1]:+.4f}")

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            state = next_state
            frames.append(rec.capture_frame())

            if done or truncated:
                break

        rec.close()

        success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)
        result = {
            "attempts":         attempt,
            "steps":            t + 1,
            "reward":           total_reward,
            "success":          success,
            "progress":         info["progress"],
            "deviation":        info["deviation"],
            "lateral_error":    info["lateral_error"],
            "orient_error":     info["orientation_error"],
            "terminal_event":   info["terminal_event"],
            "sampled_mass":     info["sampled_mass"],
            "sampled_friction": info["sampled_friction"],
        }

        print(f"  [{label}] attempt {attempt}/{max_attempts}: "
              f"progress={info['progress']:.3f}  "
              f"deviation={info['deviation']:.4f}  "
              f"success={success}  "
              f"terminal={info['terminal_event']}")

        if info["progress"] > best_progress:
            best_progress   = info["progress"]
            best_frames     = frames
            best_result     = result
            best_action_log = action_log

        if success:
            print(f"  [{label}] ✓ Success on attempt {attempt}.")
            break

    if not best_result["success"]:
        print(f"  [{label}] ✗ No success in {max_attempts} attempts. "
              f"Saving best (progress={best_progress:.3f}).")

    # Write best video
    tmp_writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    for frame in best_frames:
        tmp_writer.append_data(frame)
    tmp_writer.close()

    # Write action CSV alongside video
    csv_path = output_path.replace(".mp4", "_actions.csv")
    save_action_csv(best_action_log, csv_path)
    print(f"  [{label}] Action CSV: {os.path.basename(csv_path)}")

    return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Systematic TD3 policy evaluation.")
    parser.add_argument("--run_dir",      type=str, required=True)
    parser.add_argument("--model",        type=str, required=True,
                        help="Model name: best_model, final_model, etc.")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help="Max attempts per condition (default: 5)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--fps",          type=int, default=40)
    args = parser.parse_args()

    run_dir    = os.path.abspath(args.run_dir)
    models_dir = os.path.join(run_dir, "models")
    eval_dir   = os.path.join(run_dir, "eval")
    videos_dir = os.path.join(eval_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    total = len(ALL_CONDITIONS) * len(DIRECTIONS) * len(ARC_TYPES)
    print(f"Run dir : {run_dir}")
    print(f"Model   : {args.model}")
    print(f"Eval dir: {eval_dir}")
    print(f"Total conditions: {len(ALL_CONDITIONS)} x {len(DIRECTIONS)} directions "
          f"x {len(ARC_TYPES)} arc types = {total} videos\n")

    summary_rows = []

    # -------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------
    for arc in ARC_TYPES:
        for condition in ALL_CONDITIONS:
            for direction in DIRECTIONS:

                label = f"{arc['label']}_{condition['label']}_{direction['label']}"
                print(f"\n=== {label} ===")

                test_cfg = env_cfg.copy()
                test_cfg["gui"]            = False
                test_cfg["segment_length"] = arc["segment_length"]
                test_cfg["test_full_arc"]  = arc["test_full_arc"]
                test_cfg["max_steps"]      = arc["max_steps"]
                # Fix mass and friction — disable domain randomization
                test_cfg["mass_range"]     = [condition["mass"], condition["mass"]]
                test_cfg["friction_range"] = [condition["friction"], condition["friction"]]

                env = SimplePathFollowingEnv(**test_cfg)
                env._reverse_path = direction["reverse"]
                env.reset(seed=args.seed)

                agent = load_agent(models_dir, args.model, env)

                video_name = (
                    f"{arc['label']}"
                    f"_{condition['label']}"
                    f"_{direction['label']}"
                    f"_m{condition['mass']:.0f}"
                    f"_f{condition['friction']:.1f}"
                    f"_{'ood' if condition['ood'] else 'id'}"
                    f".mp4"
                )
                output_path = os.path.join(videos_dir, video_name)

                result = run_condition(
                    env, agent, output_path,
                    max_steps=arc["max_steps"],
                    max_attempts=args.max_attempts,
                    label=label,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                )

                env.close()

                summary_rows.append({
                    "arc_type":        arc["label"],
                    "condition":       condition["label"],
                    "direction":       direction["label"],
                    "mass_kg":         condition["mass"],
                    "friction":        condition["friction"],
                    "ood":             condition["ood"],
                    "success":         result["success"],
                    "attempts":        result["attempts"],
                    "progress":        round(result["progress"], 4),
                    "deviation_m":     round(result["deviation"], 4),
                    "lateral_err":     round(result["lateral_error"], 4),
                    "orient_err_rad":  round(result["orient_error"], 4),
                    "steps":           result["steps"],
                    "reward":          round(result["reward"], 2),
                    "terminal_event":  result["terminal_event"],
                    "video":           video_name,
                    "action_csv":      video_name.replace(".mp4", "_actions.csv"),
                })

                status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
                print(f"  → {status} | progress={result['progress']:.3f} | "
                      f"deviation={result['deviation']:.4f} | "
                      f"attempts={result['attempts']}")

    # -------------------------------------------------------------------
    # Write summary CSV
    # -------------------------------------------------------------------
    csv_path = os.path.join(eval_dir, f"results_{args.model}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV: {csv_path}")

    # -------------------------------------------------------------------
    # Write text report
    # -------------------------------------------------------------------
    report_path = os.path.join(eval_dir, f"report_{args.model}.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report — {args.model}\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Max attempts per condition: {args.max_attempts}\n")
        f.write("=" * 70 + "\n\n")

        for arc in ARC_TYPES:
            f.write(f"ARC TYPE: {arc['label'].upper()}\n")
            f.write("-" * 70 + "\n")
            arc_rows = [r for r in summary_rows if r["arc_type"] == arc["label"]]

            for ood_flag, group_label in [(False, "In-Distribution"), (True, "Out-of-Distribution")]:
                group_rows = [r for r in arc_rows if r["ood"] == ood_flag]
                if not group_rows:
                    continue
                successes = sum(1 for r in group_rows if r["success"])
                f.write(f"\n  {group_label}: {successes}/{len(group_rows)} successful\n")
                f.write(f"  {'Condition':<30} {'Dir':<14} {'OK':<5} "
                        f"{'Tries':<7} {'Progress':<10} {'Deviation':<12} {'Terminal'}\n")
                f.write(f"  {'-'*28} {'-'*12} {'-'*4} {'-'*6} {'-'*8} {'-'*10} {'-'*15}\n")
                for r in group_rows:
                    s = "✓" if r["success"] else "✗"
                    f.write(f"  {r['condition']:<30} {r['direction']:<14} "
                            f"{s:<5} {r['attempts']:<7} "
                            f"{r['progress']:<10.3f} {r['deviation_m']:<12.4f} "
                            f"{str(r['terminal_event'])}\n")

            total_success = sum(1 for r in arc_rows if r["success"])
            f.write(f"\n  TOTAL {arc['label'].upper()}: "
                    f"{total_success}/{len(arc_rows)} successful "
                    f"({100*total_success/len(arc_rows):.1f}%)\n\n")

        total_success = sum(1 for r in summary_rows if r["success"])
        f.write("=" * 70 + "\n")
        f.write(f"GRAND TOTAL: {total_success}/{len(summary_rows)} successful "
                f"({100*total_success/len(summary_rows):.1f}%)\n")

    print(f"Report: {report_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for arc in ARC_TYPES:
        arc_rows = [r for r in summary_rows if r["arc_type"] == arc["label"]]
        id_rows  = [r for r in arc_rows if not r["ood"]]
        ood_rows = [r for r in arc_rows if r["ood"]]
        print(f"\n{arc['label'].upper()}:")
        print(f"  ID  : {sum(1 for r in id_rows  if r['success'])}/{len(id_rows)}  successful")
        print(f"  OOD : {sum(1 for r in ood_rows if r['success'])}/{len(ood_rows)} successful")

    total_success = sum(1 for r in summary_rows if r["success"])
    print(f"\nGRAND TOTAL: {total_success}/{len(summary_rows)} "
          f"({100*total_success/len(summary_rows):.1f}%)")
    print(f"\nVideos + Action CSVs: {videos_dir}")


if __name__ == "__main__":
    main()