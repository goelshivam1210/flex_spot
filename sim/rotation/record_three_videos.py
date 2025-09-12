import os
import argparse
import time
import yaml
import numpy as np
import imageio
import mujoco
from scipy.spatial.transform import Rotation

from env import SimplePathFollowingEnv
from td3 import TD3
from test_dual_force import DualForceTestEnv


class VideoRecorder:
    def __init__(self, model, data, output_path, width=800, height=600, fps=40, path_points=None):
        self.model = model
        self.data = data
        self.path_points_to_draw = path_points
        self.num_path_markers = 0
        self.writer = imageio.get_writer(output_path, fps=fps)
        self.renderer = mujoco.Renderer(model, height, width)

        self.cam = mujoco.MjvCamera()
        self.cam.distance = 7.5
        self.cam.azimuth = 90.0
        self.cam.elevation = -30.0
        self.cam.lookat = np.array([0.0, 0.0, 0.5])

    def capture(self):
        self.renderer.update_scene(self.data, camera=self.cam)
        self.renderer.scene.ngeom -= self.num_path_markers
        if self.path_points_to_draw is not None:
            self.num_path_markers = plot_path_markers(self.renderer.scene, self.path_points_to_draw)
        pixels = self.renderer.render()
        self.writer.append_data(pixels)

    def close(self):
        self.writer.close()
        self.renderer.close()


def plot_path_markers(scene, path_points):
    z_height = 0.02
    added = 0
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
        added += 1
    return added


def load_trained_model(model_base_path, env):
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
    agent.load_actor(os.path.dirname(model_base_path), os.path.basename(model_base_path))
    return agent


def pick_model_base(models_dir, prefer="converged_model"):
    candidates = [
        prefer,
        "best_model",
        "final_model",
    ]
    for name in candidates:
        actor_path = os.path.join(models_dir, f"{name}_actor.pth")
        if os.path.isfile(actor_path):
            return os.path.join(models_dir, name), name
    raise FileNotFoundError(
        f"No model found in {models_dir}. Looked for "
        f"{', '.join(n + '_actor.pth' for n in candidates)}"
    )


def run_episode_simple(env, agent, max_steps, recorder):
    state, _ = env.reset()
    recorder.path_points_to_draw = getattr(env, "path_points", None)

    total_reward = 0.0
    for t in range(max_steps):
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        state = next_state
        recorder.capture()
        if done or truncated:
            break

    success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)
    return {"steps": t + 1, "reward": total_reward, "success": success, "final_info": info}


def run_episode_dual(env: DualForceTestEnv, agent, max_steps, recorder, force_mode="contact"):
    env.set_force_mode(force_mode)
    state, _ = env.reset()
    recorder.path_points_to_draw = getattr(env, "path_points", None)

    total_reward = 0.0
    for t in range(max_steps):
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        state = next_state
        recorder.capture()
        if done or truncated:
            break

    success = (info["progress"] > 0.95 and info["deviation"] < env.goal_thresh)
    return {"steps": t + 1, "reward": total_reward, "success": success, "final_info": info}


def main():
    p = argparse.ArgumentParser(description="Record one-episode videos: short, full, dual.")
    p.add_argument("--run_dir", required=True, help="Path to a run directory (e.g., runs/run-41-YYYY-mm-dd_HH-MM-SS)")
    p.add_argument("--prefer", default="converged_model",
                   help="Preferred model base name in models/ (default: converged_model). Fallback to best_model/final_model.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--short_max_steps", type=int, default=500)
    p.add_argument("--full_max_steps", type=int, default=1000)
    p.add_argument("--dual_max_steps", type=int, default=1000)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    models_dir = os.path.join(run_dir, "models")
    config_path = os.path.join(run_dir, "config.yaml")
    videos_dir = os.path.join(run_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    model_base, chosen = pick_model_base(models_dir, prefer=args.prefer)
    print(f"Using model: {chosen} at {model_base}_actor.pth")

    short_env_cfg = env_cfg.copy()
    short_env_cfg["segment_length"] = env_cfg.get("short_segment_length", 0.3)
    short_env_cfg["gui"] = False
    short_env = SimplePathFollowingEnv(**short_env_cfg)
    short_env.reset()
    short_agent = load_trained_model(model_base, short_env)

    short_out = os.path.join(videos_dir, "short.mp4")
    short_rec = VideoRecorder(short_env.model, short_env.data, short_out,
                              width=args.width, height=args.height, fps=args.fps,
                              path_points=getattr(short_env, "path_points", None))
    print("\n[SHORT] recording →", short_out)
    short_res = run_episode_simple(short_env, short_agent, args.short_max_steps, short_rec)
    short_rec.close()
    short_env.close()
    print(f"[SHORT] steps={short_res['steps']} reward={short_res['reward']:.2f} success={short_res['success']}")

    full_env_cfg = env_cfg.copy()
    full_env_cfg["segment_length"] = None
    full_env_cfg["max_steps"] = max(full_env_cfg.get("max_steps", 500), args.full_max_steps)
    full_env_cfg["gui"] = False
    full_env = SimplePathFollowingEnv(**full_env_cfg)
    full_env.reset()
    full_agent = load_trained_model(model_base, full_env)

    full_out = os.path.join(videos_dir, "full.mp4")
    full_rec = VideoRecorder(full_env.model, full_env.data, full_out,
                             width=args.width, height=args.height, fps=args.fps,
                             path_points=getattr(full_env, "path_points", None))
    print("\n[FULL] recording →", full_out)
    full_res = run_episode_simple(full_env, full_agent, args.full_max_steps, full_rec)
    full_rec.close()
    full_env.close()
    print(f"[FULL] steps={full_res['steps']} reward={full_res['reward']:.2f} success={full_res['success']}")

    dual_env_cfg = env_cfg.copy()
    dual_env_cfg["segment_length"] = None
    dual_env_cfg["max_steps"] = max(dual_env_cfg.get("max_steps", 500), args.dual_max_steps)
    dual_env_cfg["gui"] = False
    dual_env = DualForceTestEnv(**dual_env_cfg)
    dual_env.reset()
    dual_agent = load_trained_model(model_base, dual_env)

    dual_out = os.path.join(videos_dir, "dual.mp4")
    dual_rec = VideoRecorder(dual_env.model, dual_env.data, dual_out,
                             width=args.width, height=args.height, fps=args.fps,
                             path_points=getattr(dual_env, "path_points", None))
    print("\n[DUAL] recording (contact mode) →", dual_out)
    dual_res = run_episode_dual(dual_env, dual_agent, args.dual_max_steps, dual_rec, force_mode="contact")
    dual_rec.close()
    dual_env.close()
    print(f"[DUAL] steps={dual_res['steps']} reward={dual_res['reward']:.2f} success={dual_res['success']}")

    print("\nDone. Videos written to:", videos_dir)


if __name__ == "__main__":
    main()
