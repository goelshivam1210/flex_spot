"""
evaluate_generalization_batch.py — Run generalization eval across many runs and aggregate.

Discovers run directories under a parent folder (e.g. sim/rotation/runs_for_real/*),
runs the same RMSE suites as `evaluate_generalization.py` for each run, then writes:

- `per_run_results.csv`: RMSE per run/suite/condition
- `summary_mean_std.csv`: RMSE mean and std across runs, per suite/condition

Usage (from repo root):
    python sim/rotation/evaluate_generalization_batch.py --runs_dir sim/rotation/runs_for_real --model best_model --episodes 20

Output defaults to:
    <runs_dir>/eval_generalization_aggregate/
"""

import os
import sys
import csv
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

# Ensure imports work when run from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import evaluate_generalization as eg  # noqa: E402


def _discover_run_dirs(runs_dir: str) -> List[str]:
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    run_dirs: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if not os.path.isdir(d):
            continue
        if not os.path.isfile(os.path.join(d, "config.yaml")):
            continue
        if not os.path.isdir(os.path.join(d, "models")):
            continue
        run_dirs.append(d)
    return run_dirs


def _parse_seed_from_run_name(run_name: str) -> Optional[int]:
    """
    Parse seed from run directory names like: run-31-2026-02-28_14-29-50
    Returns None if parsing fails.
    """
    if not run_name.startswith("run-"):
        return None
    parts = run_name.split("-")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _eval_one_run(
    run_dir: str,
    model_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    run_table2: bool,
    run_table3: bool,
) -> Tuple[Tuple[Dict[str, float], Dict[str, List[float]]], Tuple[Dict[str, float], Dict[str, List[float]]]]:
    """
    Return:
      ((table2_rmse, table2_episode_rmses), (table3_rmse, table3_episode_rmses))
    """
    config_path = os.path.join(run_dir, "config.yaml")
    models_dir = os.path.join(run_dir, "models")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config.get("env", {})
    if "model_path" not in env_cfg:
        env_cfg["model_path"] = os.path.join(_SCRIPT_DIR, "scene.xml")

    eg._update_id_from_config(env_cfg)

    actor_path = os.path.join(models_dir, f"{model_name}_actor.pth")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"Actor not found: {actor_path}")

    # Make each run's evaluation RNG independent of ordering, but reproducible.
    rng = np.random.default_rng(seed)

    table2_rmse: Dict[str, float] = {}
    table2_episode_rmses: Dict[str, List[float]] = {}
    table3_rmse: Dict[str, float] = {}
    table3_episode_rmses: Dict[str, List[float]] = {}
    if run_table2:
        table2_rmse, table2_episode_rmses = eg.run_table2_with_episode_rmses(
            env_cfg=env_cfg,
            models_dir=models_dir,
            model_name=model_name,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            rng=rng,
        )
    if run_table3:
        table3_rmse, table3_episode_rmses = eg.run_table3_with_episode_rmses(
            env_cfg=env_cfg,
            models_dir=models_dir,
            model_name=model_name,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            rng=rng,
        )

    return (table2_rmse, table2_episode_rmses), (table3_rmse, table3_episode_rmses)


def _aggregate_mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    if len(arr) <= 1:
        std = 0.0
    else:
        std = float(np.std(arr, ddof=1))
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generalization evaluation across many runs + aggregate mean±std RMSE."
    )
    parser.add_argument("--runs_dir", type=str, required=True,
                        help="Directory containing multiple run subdirectories (each with config.yaml, models/)")
    parser.add_argument("--model", type=str, default="best_model",
                        help="Model prefix to evaluate inside each run's models/ (default: best_model)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per condition per run (default: 100)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fallback seed if not parsed from run name (default: 42)")
    parser.add_argument("--seed_from_run_name", action="store_true", default=True,
                        help="Use seed parsed from run dir name like run-<seed>-... (default: True)")
    parser.add_argument("--no_seed_from_run_name", action="store_true",
                        help="Disable parsing seed from run name (use --seed for all runs)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <runs_dir>/eval_generalization_aggregate)")
    parser.add_argument("--no_table2", action="store_true", help="Skip Table 2")
    parser.add_argument("--no_table3", action="store_true", help="Skip Table 3")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip runs missing the requested model (default: error)")
    parser.add_argument("--write_summary", action="store_true",
                        help="Also write summary_mean_std.csv across runs (analysis).")
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(
        runs_dir, "eval_generalization_aggregate"
    )
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = _discover_run_dirs(runs_dir)
    if not run_dirs:
        print(f"ERROR: No run directories found under: {runs_dir}")
        sys.exit(1)

    run_table2 = not args.no_table2
    run_table3 = not args.no_table3
    seed_from_run = bool(args.seed_from_run_name) and (not args.no_seed_from_run_name)

    print("=" * 70)
    print("BATCH GENERALIZATION EVALUATION")
    print("=" * 70)
    print(f"Runs dir : {runs_dir}")
    print(f"Found    : {len(run_dirs)} runs")
    print(f"Model    : {args.model}")
    print(f"Episodes : {args.episodes} per condition per run")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed mode: {'from run name' if seed_from_run else 'fixed'}")
    if not seed_from_run:
        print(f"Seed     : {args.seed}")
    print(f"Table 2  : {run_table2}")
    print(f"Table 3  : {run_table3}")
    print(f"Out dir  : {out_dir}")
    print()

    per_run_rows: List[Dict[str, object]] = []
    per_episode_rows: List[Dict[str, object]] = []
    used_runs: List[str] = []
    skipped_runs: List[Tuple[str, str]] = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"--- {run_name} ---")
        run_seed = _parse_seed_from_run_name(run_name) if seed_from_run else None
        if run_seed is None:
            run_seed = int(args.seed)
        try:
            (t2_rmse, t2_ep), (t3_rmse, t3_ep) = _eval_one_run(
                run_dir=run_dir,
                model_name=args.model,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=run_seed,
                run_table2=run_table2,
                run_table3=run_table3,
            )
        except FileNotFoundError as e:
            if args.skip_missing:
                msg = str(e)
                skipped_runs.append((run_name, msg))
                print(f"  SKIP: {msg}")
                continue
            raise

        used_runs.append(run_name)
        for cond, rmse in t2_rmse.items():
            per_run_rows.append({
                "run": run_name,
                "run_seed": run_seed,
                "model": args.model,
                "suite": "table2",
                "condition": cond,
                "rmse": float(rmse),
            })
            print(f"  Table2 {cond}: RMSE={rmse:.6f} m")
        for cond, rmse in t3_rmse.items():
            per_run_rows.append({
                "run": run_name,
                "run_seed": run_seed,
                "model": args.model,
                "suite": "table3",
                "condition": cond,
                "rmse": float(rmse),
            })
            print(f"  Table3 {cond}: RMSE={rmse:.6f} m")

        for cond, ep_rmses in t2_ep.items():
            for i, v in enumerate(ep_rmses):
                per_episode_rows.append({
                    "run": run_name,
                    "run_seed": run_seed,
                    "model": args.model,
                    "suite": "table2",
                    "condition": cond,
                    "episode": i,
                    "episode_rmse": float(v),
                })
        for cond, ep_rmses in t3_ep.items():
            for i, v in enumerate(ep_rmses):
                per_episode_rows.append({
                    "run": run_name,
                    "run_seed": run_seed,
                    "model": args.model,
                    "suite": "table3",
                    "condition": cond,
                    "episode": i,
                    "episode_rmse": float(v),
                })

    if not per_run_rows:
        print("ERROR: No results collected (all runs skipped or failed).")
        sys.exit(1)

    # Write per-run CSV
    per_run_csv = os.path.join(out_dir, "per_run_results.csv")
    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "run_seed", "model", "suite", "condition", "rmse"])
        writer.writeheader()
        writer.writerows(per_run_rows)

    per_episode_csv = os.path.join(out_dir, "per_episode_results.csv")
    with open(per_episode_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "run_seed", "model", "suite", "condition", "episode", "episode_rmse"],
        )
        writer.writeheader()
        writer.writerows(per_episode_rows)

    summary_csv = None
    if args.write_summary:
        by_key: Dict[Tuple[str, str], List[float]] = {}
        for r in per_run_rows:
            key = (str(r["suite"]), str(r["condition"]))
            by_key.setdefault(key, []).append(float(r["rmse"]))

        summary_rows: List[Dict[str, object]] = []
        for (suite, condition), vals in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
            mean, std = _aggregate_mean_std(vals)
            summary_rows.append({
                "suite": suite,
                "condition": condition,
                "n_runs": len(vals),
                "rmse_mean": mean,
                "rmse_std": std,
            })

        summary_csv = os.path.join(out_dir, "summary_mean_std.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["suite", "condition", "n_runs", "rmse_mean", "rmse_std"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    # Metadata JSON (nice-to-have)
    meta_json = os.path.join(out_dir, "aggregate_metadata.json")
    with open(meta_json, "w") as f:
        json.dump({
            "runs_dir": runs_dir,
            "model": args.model,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed_mode": "from run name" if seed_from_run else "fixed",
            "seed_fallback": args.seed,
            "included_runs": used_runs,
            "skipped_runs": [{"run": n, "reason": r} for n, r in skipped_runs],
            "outputs": {
                "per_run_csv": per_run_csv,
                "per_episode_csv": per_episode_csv,
                "summary_csv": summary_csv,
            },
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Included runs: {len(used_runs)}/{len(run_dirs)}")
    print(f"Per-run CSV  : {per_run_csv}")
    print(f"Per-episode CSV: {per_episode_csv}")
    if summary_csv:
        print(f"Summary CSV  : {summary_csv}")
    if skipped_runs:
        print(f"Skipped runs : {len(skipped_runs)} (see {meta_json})")


if __name__ == "__main__":
    main()

