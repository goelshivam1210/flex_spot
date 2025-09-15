import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from td3 import TD3
from test_dual_force import DualForceTestEnv

def split_model_path(model_path: str):
    """Return (model_dir, prefix) from a path that might be a prefix or an *_actor.pth file."""
    if model_path.endswith(".pth"):
        model_dir = os.path.dirname(model_path)
        base = os.path.basename(model_path)
        if base.endswith("_actor.pth"):
            prefix = base[:-len("_actor.pth")]
        elif base.endswith("_critic.pth"):
            prefix = base[:-len("_critic.pth")]
        else:
            prefix = os.path.splitext(base)[0]
    else:
        model_dir = os.path.dirname(model_path)
        prefix = os.path.basename(model_path)
    return model_dir, prefix

def load_agent_for_env(model_prefix_path: str, env: DualForceTestEnv) -> TD3:
    """Load actor weights for TD3 from a prefix path (e.g., .../models/converged_model)."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(
        lr=1e-3,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        max_torque=env.max_torque
    )
    model_dir, prefix = split_model_path(model_prefix_path)
    # Will look for '<prefix>_actor.pth' inside model_dir
    agent.load_actor(model_dir, prefix)
    return agent

def run_and_record(env: DualForceTestEnv, agent: TD3, mode: str, max_steps: int):
    """Run a single full-arc episode and collect executed XY trajectory."""
    env.set_force_mode(mode)
    state, _ = env.reset()

    exec_xy = []
    done = truncated = False
    steps = 0
    total_reward = 0.0
    last_info = {}

    while not (done or truncated) and steps < max_steps:
        action = agent.select_action(np.array(state))
        if getattr(action, "ndim", 1) > 1:
            action = action.squeeze(0)
        next_state, reward, done, truncated, info = env.step(action)
        exec_xy.append(env.data.body('box').xpos[:2].copy())
        state = next_state
        total_reward += float(reward)
        steps += 1
        last_info = info

    path_xy = env.path_points.copy()
    success = (last_info.get("progress", 0.0) > 0.95 and
               last_info.get("deviation", 1e9) < env.goal_thresh)
    return {
        "path_xy": path_xy,
        "exec_xy": np.array(exec_xy),
        "success": success,
        "final_deviation": float(last_info.get("deviation", np.nan)),
        "final_progress": float(last_info.get("progress", np.nan)),
        "total_reward": total_reward,
        "steps": steps,
    }

def plot_overlay(path_xy, exec_xy, title, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.figure(figsize=(8, 7))
    # planned arc
    plt.plot(path_xy[:, 0], path_xy[:, 1], linestyle='--', linewidth=2, label='Planned Path')
    plt.scatter(path_xy[0, 0], path_xy[0, 1], s=80, marker='o', label='Path Start')
    plt.scatter(path_xy[-1, 0], path_xy[-1, 1], s=80, marker='s', label='Path End')
    # executed trajectory
    if len(exec_xy) > 0:
        plt.plot(exec_xy[:, 0], exec_xy[:, 1], linewidth=2, label='Executed Trajectory')
        plt.scatter(exec_xy[0, 0], exec_xy[0, 1], s=70, marker='^', label='Exec Start')
        plt.scatter(exec_xy[-1, 0], exec_xy[-1, 1], s=70, marker='v', label='Exec End')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {outfile}")

def build_env(env_cfg, max_steps):
    """Full-arc env (segment_length=None)."""
    env = DualForceTestEnv(
        gui=False,
        max_force=env_cfg.get("max_force", 400.0),
        max_torque=env_cfg.get("max_torque", 50.0),
        friction=env_cfg.get("friction", 0.4),
        spinning_friction=env_cfg.get("spinning_friction", 0.05),
        rolling_friction=env_cfg.get("rolling_friction", 0.01),
        linear_damping=env_cfg.get("linear_damping", 0.05),
        angular_damping=env_cfg.get("angular_damping", 0.1),
        goal_thresh=env_cfg.get("goal_thresh", 0.2),
        strict_terminal=env_cfg.get("strict_terminal", True),
        spin_penalty_k=env_cfg.get("spin_penalty_k", 0.0),
        deviation_tolerance=env_cfg.get("deviation_tolerance", 0.15),
        segment_length=None,   # <— FULL ARC
        max_steps=max_steps
    )
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run directories (each should contain models/converged_model*)")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Config file to read env params from")
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load env config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    for run_dir in args.runs:
        models_dir = os.path.join(run_dir, "models")
        prefix = os.path.join(models_dir, "converged_model")
        actor_pth = prefix + "_actor.pth"

        if not os.path.exists(actor_pth):
            raise FileNotFoundError(
                f"Could not find actor file at {actor_pth}. "
                f"Make sure '{prefix}_actor.pth' exists."
            )

        # Output dir inside the run
        outdir = os.path.join(run_dir, "plots")
        os.makedirs(outdir, exist_ok=True)

        # Build and seed env once per run (keeps the same full arc across modes)
        env = build_env(env_cfg, args.max_steps)
        env.reset(seed=args.seed)

        print(f"\n=== {os.path.basename(run_dir)} ===")
        print(f"Arc: radius={env.arc_radius:.2f}, "
              f"start={np.degrees(env.arc_start):.1f}°, end={np.degrees(env.arc_end):.1f}°")

        # Load converged_model
        agent = load_agent_for_env(prefix, env)

        # CENTROID
        res_c = run_and_record(env, agent, mode="centroid", max_steps=args.max_steps)
        title_c = (f"converged_model — CENTROID\n"
                   f"success={res_c['success']}  "
                   f"prog={res_c['final_progress']:.2f}  "
                   f"dev={res_c['final_deviation']:.3f}  "
                   f"steps={res_c['steps']}")
        out_c = os.path.join(outdir, "converged_centroid_overlay.png")
        plot_overlay(res_c["path_xy"], res_c["exec_xy"], title_c, out_c)

        # CONTACT / DUAL
        res_d = run_and_record(env, agent, mode="contact", max_steps=args.max_steps)
        title_d = (f"converged_model — CONTACT (dual)\n"
                   f"success={res_d['success']}  "
                   f"prog={res_d['final_progress']:.2f}  "
                   f"dev={res_d['final_deviation']:.3f}  "
                   f"steps={res_d['steps']}")
        out_d = os.path.join(outdir, "converged_contact_overlay.png")
        plot_overlay(res_d["path_xy"], res_d["exec_xy"], title_d, out_d)

        env.close()

if __name__ == "__main__":
    main()
