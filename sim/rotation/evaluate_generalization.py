"""
evaluate_generalization.py — Systematic generalization evaluation with RMSE reporting.

Evaluates trained TD3 policies on:
  - Table 2: Path adherence sweeps (ID, Mass OOD Low/High, Friction OOD Low/High, Radius OOD Low/High)
  - Table 3: Trajectory type generalization (Arc 60°/120°/180°, S-path, Meandering path)

RMSE is computed over path adherence error (deviation = distance from box to closest path point)
across all steps in all episodes per condition. Uses config.yaml as source of truth for ID ranges.

Usage (from repo root):
    # Single checkpoint (run_dir + model name)
    python sim/rotation/evaluate_generalization.py --checkpoint sim/rotation/runs/run-0 --model best_model

    # Single checkpoint (explicit path to model)
    python sim/rotation/evaluate_generalization.py --checkpoint sim/rotation/runs/run-0/models/best_model

    # All checkpoints in a run directory
    python sim/rotation/evaluate_generalization.py --checkpoints_dir sim/rotation/runs/run-0

    # Reduce episodes for faster run
    python sim/rotation/evaluate_generalization.py --checkpoint <run_dir> --model best_model --episodes 10

    # Run only Table 2 or only Table 3
    python sim/rotation/evaluate_generalization.py --checkpoint <run_dir> --model best_model --no_table3
    python sim/rotation/evaluate_generalization.py --checkpoint <run_dir> --model best_model --no_table2

Output:
  - Printed summary (human readable)
  - results_generalization.csv: model, suite, condition, rmse (machine-readable)
  - results_generalization.json: full results plus metadata
"""

import os
import sys
import csv
import json
import argparse
import yaml
import glob
import numpy as np

# Ensure imports work when run from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from env import SimplePathFollowingEnv
from td3 import TD3

# Import path generators for Table 3 (S-path, meandering)
try:
    from test_generalize import (
        generate_s_path,
        generate_meander_path,
        reset_env_with_path,
    )
except ImportError:
    generate_s_path = None
    generate_meander_path = None
    reset_env_with_path = None


# ---------------------------------------------------------------------------
# Constants: ID vs OOD ranges
# ---------------------------------------------------------------------------

# Train / ID defaults (overridden from config.yaml when available)
ID_MASS_LO, ID_MASS_HI = 5.0, 15.0
ID_FRICTION_LO, ID_FRICTION_HI = 0.4, 0.6
ID_RADIUS = 1.5


def _update_id_from_config(env_cfg):
    """Update module-level ID constants from config.yaml (source of truth for ID)."""
    global ID_MASS_LO, ID_MASS_HI, ID_FRICTION_LO, ID_FRICTION_HI, ID_RADIUS
    if "mass_range" in env_cfg and len(env_cfg["mass_range"]) >= 2:
        ID_MASS_LO, ID_MASS_HI = env_cfg["mass_range"][0], env_cfg["mass_range"][1]
    if "friction_range" in env_cfg and len(env_cfg["friction_range"]) >= 2:
        ID_FRICTION_LO, ID_FRICTION_HI = env_cfg["friction_range"][0], env_cfg["friction_range"][1]
    if "arc_radius" in env_cfg:
        ID_RADIUS = float(env_cfg["arc_radius"])

# OOD ranges (Table 1)
OOD_MASS_LOW = (2.5, 5.0)
OOD_MASS_HIGH = (15.0, 17.5)
OOD_FRICTION_LOW = (0.3, 0.4)
OOD_FRICTION_HIGH = (0.6, 0.7)
OOD_RADIUS_LOW = (1.0, 1.5)
OOD_RADIUS_HIGH = (1.5, 2.0)

# Arc spans in radians (Table 3)
ARC_60_RAD = 1.0471975512   # π/3
ARC_120_RAD = 2.0943951024  # 2π/3
ARC_180_RAD = 3.1415926536  # π


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(models_dir, model_name, env, device=None):
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
# Rollout and RMSE computation
# ---------------------------------------------------------------------------

def run_rollout(env, agent, max_steps, rng, episode_seed=None):
    """Run one episode, return list of per-step deviations (path adherence errors)."""
    seed_val = int(episode_seed) if episode_seed is not None else int(rng.integers(0, 2**31))
    state, _ = env.reset(seed=seed_val)
    deviations = []
    for _ in range(max_steps):
        action = agent.select_action(np.array(state))
        if action.ndim > 1:
            action = action.squeeze(0)
        next_state, reward, done, truncated, info = env.step(action)
        deviations.append(float(info["deviation"]))
        state = next_state
        if done or truncated:
            break
    return deviations


def compute_rmse(deviations_list):
    """RMSE over all deviations (path adherence error)."""
    if not deviations_list:
        return float("nan")
    all_dev = np.concatenate([np.array(d) for d in deviations_list if len(d) > 0])
    if len(all_dev) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(all_dev ** 2)))


# ---------------------------------------------------------------------------
# Table 2: Path adherence sweeps
# ---------------------------------------------------------------------------

def build_env_for_table2(env_cfg, condition_name, rng, arc_radius_override=None, max_steps=1000):
    """
    Build env config with mass_range, friction_range, arc_radius set per condition.
    - ID: all params in ID range
    - Mass_OOD_Low/High: only mass in OOD range
    - Friction_OOD_Low/High: only friction in OOD range
    - Radius_OOD_Low/High: arc_radius_override must be provided (sampled per episode)
    """
    cfg = env_cfg.copy()
    cfg["gui"] = False
    cfg["segment_length"] = None  # full arc
    cfg["test_full_arc"] = True
    cfg["max_steps"] = max_steps

    if condition_name == "ID":
        cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
        cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
        cfg["arc_radius"] = ID_RADIUS
    elif condition_name == "Mass_OOD_Low":
        cfg["mass_range"] = list(OOD_MASS_LOW)
        cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
        cfg["arc_radius"] = ID_RADIUS
    elif condition_name == "Mass_OOD_High":
        cfg["mass_range"] = list(OOD_MASS_HIGH)
        cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
        cfg["arc_radius"] = ID_RADIUS
    elif condition_name == "Friction_OOD_Low":
        cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
        cfg["friction_range"] = list(OOD_FRICTION_LOW)
        cfg["arc_radius"] = ID_RADIUS
    elif condition_name == "Friction_OOD_High":
        cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
        cfg["friction_range"] = list(OOD_FRICTION_HIGH)
        cfg["arc_radius"] = ID_RADIUS
    elif condition_name == "Radius_OOD_Low":
        cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
        cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
        cfg["arc_radius"] = arc_radius_override if arc_radius_override is not None else rng.uniform(
            OOD_RADIUS_LOW[0], OOD_RADIUS_LOW[1]
        )
    elif condition_name == "Radius_OOD_High":
        cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
        cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
        cfg["arc_radius"] = arc_radius_override if arc_radius_override is not None else rng.uniform(
            OOD_RADIUS_HIGH[0], OOD_RADIUS_HIGH[1]
        )
    else:
        raise ValueError(f"Unknown condition: {condition_name}")

    return cfg


def run_table2(env_cfg, models_dir, model_name, episodes, max_steps, seed, rng):
    """Run Table 2: path adherence sweeps. Returns dict condition -> RMSE."""
    conditions = [
        "ID",
        "Mass_OOD_Low", "Mass_OOD_High",
        "Friction_OOD_Low", "Friction_OOD_High",
        "Radius_OOD_Low", "Radius_OOD_High",
    ]
    results = {}
    arc_start = env_cfg.get("arc_start", -np.pi / 3)
    arc_end = env_cfg.get("arc_end", np.pi / 3)

    for cond in conditions:
        deviations_list = []
        is_radius_cond = cond.startswith("Radius_OOD")
        for ep in range(episodes):
            radius_override = None
            if is_radius_cond:
                if cond == "Radius_OOD_Low":
                    radius_override = rng.uniform(OOD_RADIUS_LOW[0], OOD_RADIUS_LOW[1])
                else:
                    radius_override = rng.uniform(OOD_RADIUS_HIGH[0], OOD_RADIUS_HIGH[1])
            cfg = build_env_for_table2(
                env_cfg, cond, rng, arc_radius_override=radius_override, max_steps=max_steps
            )
            env = SimplePathFollowingEnv(**cfg)
            env._reverse_path = False
            env.arc_start = arc_start
            env.arc_end = arc_end
            agent_loaded = load_agent(models_dir, model_name, env)
            devs = run_rollout(env, agent_loaded, max_steps, rng, episode_seed=seed + ep)
            deviations_list.append(devs)
            env.close()
        results[cond] = compute_rmse(deviations_list)
    return results


# ---------------------------------------------------------------------------
# Table 3: Trajectory type generalization
# ---------------------------------------------------------------------------

def arc_start_end_for_span(span_rad, center_rad=0.0):
    """Return (arc_start, arc_end) for a given angular span (radians)."""
    half = span_rad / 2
    return center_rad - half, center_rad + half


def run_table3_arc(env_cfg, models_dir, model_name, span_rad, episodes, max_steps, seed, rng):
    """Run eval on an arc of given span (radians), ID physical params."""
    cfg = env_cfg.copy()
    cfg["gui"] = False
    cfg["segment_length"] = None
    cfg["test_full_arc"] = True
    cfg["max_steps"] = max_steps
    cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
    cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
    cfg["arc_radius"] = ID_RADIUS
    start, end = arc_start_end_for_span(span_rad)
    cfg["arc_start"] = start
    cfg["arc_end"] = end

    env = SimplePathFollowingEnv(**cfg)
    env._reverse_path = False
    agent_loaded = load_agent(models_dir, model_name, env)
    deviations_list = []
    for ep in range(episodes):
        devs = run_rollout(env, agent_loaded, max_steps, rng, episode_seed=seed + ep)
        deviations_list.append(devs)
    env.close()
    return compute_rmse(deviations_list)


def run_table3_custom_path(env_cfg, models_dir, model_name, path_points, episodes, max_steps, seed, rng):
    """Run eval on custom path (S-path or meandering), ID physical params."""
    if reset_env_with_path is None:
        return float("nan")  # test_generalize not available
    cfg = env_cfg.copy()
    cfg["gui"] = False
    cfg["segment_length"] = None
    cfg["test_full_arc"] = True
    cfg["mass_range"] = [ID_MASS_LO, ID_MASS_HI]
    cfg["friction_range"] = [ID_FRICTION_LO, ID_FRICTION_HI]
    cfg["arc_radius"] = ID_RADIUS
    cfg["max_steps"] = max_steps

    env = SimplePathFollowingEnv(**cfg)
    env._reverse_path = False
    env.reset(seed=seed)
    agent_loaded = load_agent(models_dir, model_name, env)
    deviations_list = []
    for ep in range(episodes):
        state = reset_env_with_path(env, path_points, rng)
        ep_devs = []
        for _ in range(max_steps):
            action = agent_loaded.select_action(np.array(state))
            if action.ndim > 1:
                action = action.squeeze(0)
            next_state, reward, done, truncated, info = env.step(action)
            ep_devs.append(float(info["deviation"]))
            state = next_state
            if done or truncated:
                break
        deviations_list.append(ep_devs)
    env.close()
    return compute_rmse(deviations_list)


def run_table3(env_cfg, models_dir, model_name, episodes, max_steps, seed, rng):
    """Run Table 3: trajectory type generalization."""
    results = {}
    # Arc 60°, 120°, 180°
    results["Arc_60deg"] = run_table3_arc(
        env_cfg, models_dir, model_name, ARC_60_RAD, episodes, max_steps, seed, rng
    )
    results["Arc_120deg"] = run_table3_arc(
        env_cfg, models_dir, model_name, ARC_120_RAD, episodes, max_steps, seed, rng
    )
    results["Arc_180deg"] = run_table3_arc(
        env_cfg, models_dir, model_name, ARC_180_RAD, episodes, max_steps, seed, rng
    )
    # S-path and Meandering
    if generate_s_path is not None and generate_meander_path is not None:
        s_path = generate_s_path(length=3.0, amplitude=0.5, num_points=100)
        meander_path = generate_meander_path(length=4.0, amplitude=0.4, num_points=150)
        results["S_path"] = run_table3_custom_path(
            env_cfg, models_dir, model_name, s_path, episodes, max_steps, seed, rng
        )
        results["Meandering_path"] = run_table3_custom_path(
            env_cfg, models_dir, model_name, meander_path, episodes, max_steps, seed, rng
        )
    else:
        results["S_path"] = float("nan")
        results["Meandering_path"] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def resolve_checkpoints(checkpoint_path=None, checkpoints_dir=None, model_name=None):
    """
    Resolve (run_dir, models_dir, list of model names).
    - checkpoint_path: path to run_dir/models/model_name OR run_dir (then model_name required)
    - checkpoints_dir: path to run_dir, evaluate all models in models/
    """
    if checkpoints_dir:
        run_dir = os.path.abspath(checkpoints_dir)
        config_path = os.path.join(run_dir, "config.yaml")
        models_dir = os.path.join(run_dir, "models")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.yaml not found: {config_path}")
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"models dir not found: {models_dir}")
        patterns = glob.glob(os.path.join(models_dir, "*_actor.pth"))
        model_names = []
        for p in patterns:
            base = os.path.basename(p)
            name = base.replace("_actor.pth", "")
            model_names.append(name)
        if not model_names:
            raise FileNotFoundError(f"No *_actor.pth in {models_dir}")
        return run_dir, models_dir, sorted(model_names)

    if checkpoint_path:
        path = os.path.abspath(checkpoint_path)
        if os.path.isdir(path):
            run_dir = path
            models_dir = os.path.join(run_dir, "models")
            config_path = os.path.join(run_dir, "config.yaml")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"config.yaml not found: {config_path}")
            if not model_name:
                raise ValueError("--model required when --checkpoint is a run directory")
            actor_path = os.path.join(models_dir, f"{model_name}_actor.pth")
            if not os.path.isfile(actor_path):
                raise FileNotFoundError(f"Model not found: {actor_path}")
            return run_dir, models_dir, [model_name]
        else:
            # path is run_dir/models/model_name (prefix to model files)
            models_dir = os.path.dirname(path)
            model_name_from_path = os.path.basename(path)
            run_dir = os.path.dirname(models_dir)
            config_path = os.path.join(run_dir, "config.yaml")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"config.yaml not found for checkpoint: {path}")
            actor_path = os.path.join(models_dir, f"{model_name_from_path}_actor.pth")
            if not os.path.isfile(actor_path):
                raise FileNotFoundError(f"Model not found: {actor_path}")
            return run_dir, models_dir, [model_name_from_path]
    raise ValueError("Provide --checkpoint or --checkpoints_dir")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generalization evaluation with RMSE (path adherence error)."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to run_dir or run_dir/models/model_name")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Path to run_dir; evaluate all models in models/")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name when --checkpoint is run_dir (e.g. best_model)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per condition (default: 20)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory; default: run_dir/eval_generalization")
    parser.add_argument("--run_table2", action="store_true", default=True,
                        help="Run Table 2 path adherence sweeps (default: True)")
    parser.add_argument("--run_table3", action="store_true", default=True,
                        help="Run Table 3 trajectory type generalization (default: True)")
    parser.add_argument("--no_table2", action="store_true", dest="no_table2",
                        help="Skip Table 2")
    parser.add_argument("--no_table3", action="store_true", dest="no_table3",
                        help="Skip Table 3")
    args = parser.parse_args()

    if args.checkpoint and args.checkpoints_dir:
        print("ERROR: Provide --checkpoint OR --checkpoints_dir, not both.")
        sys.exit(1)
    if not args.checkpoint and not args.checkpoints_dir:
        print("ERROR: Provide --checkpoint or --checkpoints_dir.")
        sys.exit(1)

    run_dir, models_dir, model_names = resolve_checkpoints(
        args.checkpoint, args.checkpoints_dir, args.model
    )
    out_dir = args.out_dir or os.path.join(run_dir, "eval_generalization")
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config.get("env", {})
    if "model_path" not in env_cfg:
        env_cfg["model_path"] = os.path.join(_SCRIPT_DIR, "scene.xml")
    _update_id_from_config(env_cfg)

    rng = np.random.default_rng(args.seed)
    run_table2_flag = args.run_table2 and not args.no_table2
    run_table3_flag = args.run_table3 and not args.no_table3

    print("=" * 70)
    print("GENERALIZATION EVALUATION")
    print("=" * 70)
    print(f"Run dir  : {run_dir}")
    print(f"Models   : {model_names}")
    print(f"Episodes : {args.episodes} per condition")
    print(f"Seed     : {args.seed}")
    print(f"Table 2  : {run_table2_flag}")
    print(f"Table 3  : {run_table3_flag}")
    print(f"Out dir  : {out_dir}")
    print()

    all_rows = []
    for model_name in model_names:
        print(f"\n--- Model: {model_name} ---")
        table2_results = {}
        table3_results = {}
        if run_table2_flag:
            table2_results = run_table2(
                env_cfg, models_dir, model_name,
                args.episodes, args.max_steps, args.seed, rng
            )
            for cond, rmse in table2_results.items():
                all_rows.append({
                    "model": model_name,
                    "suite": "table2",
                    "condition": cond,
                    "rmse": round(rmse, 6),
                })
                print(f"  Table2 {cond}: RMSE = {rmse:.6f} m")
        if run_table3_flag:
            table3_results = run_table3(
                env_cfg, models_dir, model_name,
                args.episodes, args.max_steps, args.seed, rng
            )
            for cond, rmse in table3_results.items():
                all_rows.append({
                    "model": model_name,
                    "suite": "table3",
                    "condition": cond,
                    "rmse": round(rmse, 6),
                })
                print(f"  Table3 {cond}: RMSE = {rmse:.6f} m")

    # Save machine-readable CSV
    csv_path = os.path.join(out_dir, "results_generalization.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "suite", "condition", "rmse"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults CSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(out_dir, "results_generalization.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_dir": run_dir,
            "models": model_names,
            "episodes": args.episodes,
            "seed": args.seed,
            "results": all_rows,
        }, f, indent=2)
    print(f"Results JSON: {json_path}")

    # Printed summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model_name in model_names:
        model_rows = [r for r in all_rows if r["model"] == model_name]
        print(f"\n{model_name}:")
        t2 = [r for r in model_rows if r["suite"] == "table2"]
        t3 = [r for r in model_rows if r["suite"] == "table3"]
        if t2:
            print("  Table 2 (Path adherence):")
            for r in t2:
                print(f"    {r['condition']:20s} RMSE = {r['rmse']:.6f} m")
        if t3:
            print("  Table 3 (Trajectory type):")
            for r in t3:
                print(f"    {r['condition']:20s} RMSE = {r['rmse']:.6f} m")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
