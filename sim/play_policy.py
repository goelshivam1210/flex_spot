# play_policy.py

import torch
import numpy as np
import argparse
import yaml
import time
import imageio.v2 as imageio  # Ensure this is imageio.v2
import pybullet as p
from pathlib import Path
import math  # For math.pi if used in config defaults

# Ensure this imports the env.py from the same directory
from env import PrismaticEnv
from td3 import TD3


def play(config_path, model_load_path, model_name,
         num_episodes=10, no_gui=False, record_dir=None):
    # Load configuration that was used for the specific training run
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config.get("env", {})
    agent_cfg = config.get("agent", {})

    # Environment parameters from the loaded config
    env_kwargs = {
        # Flip gui flag based on --no-gui
        "gui": not no_gui,
        "max_force_per_pusher": env_cfg.get("max_force_per_pusher", 50.0),
        "friction": env_cfg.get("friction", 0.5),
        "angular_damping_factor": env_cfg.get("angular_damping_factor", 0.2),
        "goal_pos": np.array(env_cfg.get("goal_pos", [2.0, 0.0])),
        "goal_thresh": env_cfg.get("goal_thresh", 0.1),
        "max_steps": env_cfg.get("max_steps", 500),
        "pusher_offset_distance": env_cfg.get("pusher_offset_distance", 0.15),
        "randomize_initial_yaw": env_cfg.get("randomize_initial_yaw", False),
        "initial_yaw_range": np.array(env_cfg.get("initial_yaw_range", [0, 0])),
    }

    env = PrismaticEnv(**env_kwargs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action_val = float(agent_cfg.get("max_action", 1.0))

    agent = TD3(lr=0,  # lr not used for inference
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action_val)

    try:
        agent.load_actor(model_load_path, model_name)
        print(f"Successfully loaded actor model from {model_load_path}/{model_name}_actor.pth")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        terminated = False
        truncated = False
        ep_steps = 0

        print(f"\n--- Episode {ep + 1} ---")
        initial_yaw_rad = math.atan2(state[1], state[0])
        print(f"Initial Box Yaw: {math.degrees(initial_yaw_rad):.1f}°")

        # Prepare recording
        video_writer_instance = None
        if record_dir:
            vid_path = Path(record_dir) / f"episode_{ep+1}.mp4"
            vid_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                video_writer_instance = imageio.get_writer(
                    vid_path, fps=30, codec='libx264', quality=7
                )
                print(f"Recording to {vid_path}")
            except Exception as e:
                print(f"Could not create video writer for {vid_path}: {e}")
                video_writer_instance = None

        while not (terminated or truncated) and ep_steps < env.max_steps:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Record frame if requested
            if video_writer_instance:
                try:
                    img = env.render(mode='rgb_array')
                    if img is not None:
                        video_writer_instance.append_data(img)
                    elif ep_steps == 0:
                        print("Warning: env.render('rgb_array') returned None.")
                except Exception as e:
                    if ep_steps == 0:
                        print(f"Warning: render error: {e}")

            state = next_state
            ep_reward += reward
            ep_steps += 1

            # If GUI is active and not suppressed, slow down for viewing
            if not no_gui and env.gui:
                time.sleep(1.0 / 60.0)

        # Close & report recording
        if video_writer_instance:
            video_writer_instance.close()
            print(f"Finished writing video for episode {ep+1}")

        # Episode summary
        print(f"Episode {ep + 1} finished in {ep_steps} steps, total reward {ep_reward:.2f}")
        final_pos = state[2:4] + env.initial_box_pos_xy
        if np.linalg.norm(final_pos - env.goal_pos_world) < env.goal_thresh:
            print("Goal Reached!")
        elif truncated:
            print("Episode truncated (max steps reached).")
        else:
            print("Terminated before goal.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained TD3 agent.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the training config.yaml")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory where the trained model is saved")
    parser.add_argument("--model_name", type=str, default="best_model",
                        help="Name of the model files to load")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (no PyBullet GUI)")
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Directory to save episode videos (requires GUI)")
    args = parser.parse_args()

    if args.record_dir and args.no_gui:
        print("Warning: Recording requested but --no-gui is set. Recording disabled.")
        args.record_dir = None

    play(
        args.config_path,
        args.model_dir,
        args.model_name,
        num_episodes=args.episodes,
        no_gui=args.no_gui,
        record_dir=args.record_dir
    )
