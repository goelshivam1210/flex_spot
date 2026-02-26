"""
query_actions.py — Query per-step action logs for a specific evaluation video.

Usage:
    # List all available videos
    python query_actions.py --eval_dir runs/run-0-.../eval

    # Print actions for a specific video
    python query_actions.py --eval_dir runs/run-0-.../eval --video short_low_mass_low_friction_clockwise_m5_f0.4_id

    # Print only first N steps
    python query_actions.py --eval_dir runs/run-0-.../eval --video <name> --steps 20

    # Filter by step range
    python query_actions.py --eval_dir runs/run-0-.../eval --video <name> --from_step 0 --to_step 50
"""

import os
import sys
import csv
import argparse
import glob


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_actions(rows, from_step=None, to_step=None):
    """Pretty print action rows with aligned columns."""
    if from_step is not None:
        rows = [r for r in rows if int(r["step"]) >= from_step]
    if to_step is not None:
        rows = [r for r in rows if int(r["step"]) <= to_step]

    if not rows:
        print("No rows found for the given step range.")
        return

    # Header
    print(f"\n{'Step':>5}  "
          f"{'fx_raw':>8}  {'fy_raw':>8}  "
          f"{'fx(N)':>8}  {'fy(N)':>8}  {'|F|(N)':>8}  "
          f"{'τz':>9}  "
          f"{'r_x':>7}  {'r_y':>7}  "
          f"{'lat_err':>8}  {'ori_err':>8}  "
          f"{'spd_fwd':>8}  {'spd_lat':>8}  "
          f"{'ang_vel':>8}  {'f_react':>7}")
    print("-" * 145)

    for r in rows:
        print(f"{int(r['step']):>5}  "
              f"{float(r['action_fx_raw']):>+8.4f}  "
              f"{float(r['action_fy_raw']):>+8.4f}  "
              f"{float(r['fx_N']):>+8.2f}  "
              f"{float(r['fy_N']):>+8.2f}  "
              f"{float(r['force_mag_N']):>8.2f}  "
              f"{float(r['torque_z']):>+9.4f}  "
              f"{float(r['r_local_x']):>7.4f}  "
              f"{float(r['r_local_y']):>7.4f}  "
              f"{float(r['lateral_err']):>+8.4f}  "
              f"{float(r['orient_err']):>+8.4f}  "
              f"{float(r['speed_fwd']):>+8.4f}  "
              f"{float(r['speed_lat']):>+8.4f}  "
              f"{float(r['angular_vel']):>+8.4f}  "
              f"{float(r['f_react']):>7.4f}")

    print(f"\nTotal steps shown: {len(rows)}")


def print_summary(rows):
    """Print aggregate stats for an episode."""
    import statistics

    def col(key):
        return [float(r[key]) for r in rows]

    print("\n--- Episode Summary ---")
    print(f"  Total steps      : {len(rows)}")
    print(f"  fx_N   mean/std  : {statistics.mean(col('fx_N')):+.2f} / {statistics.stdev(col('fx_N')):.2f}")
    print(f"  fy_N   mean/std  : {statistics.mean(col('fy_N')):+.2f} / {statistics.stdev(col('fy_N')):.2f}")
    print(f"  |F|    mean/max  : {statistics.mean(col('force_mag_N')):.2f} / {max(col('force_mag_N')):.2f}")
    print(f"  τz     mean/std  : {statistics.mean(col('torque_z')):+.4f} / {statistics.stdev(col('torque_z')):.4f}")
    print(f"  lat_err mean/max : {statistics.mean(col('lateral_err')):+.4f} / {max(col('lateral_err'), key=abs):.4f}")
    print(f"  ori_err mean/max : {statistics.mean(col('orient_err')):+.4f} / {max(col('orient_err'), key=abs):.4f}")
    print(f"  ang_vel mean/std : {statistics.mean(col('angular_vel')):+.4f} / {statistics.stdev(col('angular_vel')):.4f}")
    print(f"  spd_fwd mean     : {statistics.mean(col('speed_fwd')):+.4f}")
    print(f"  r_local          : [{rows[0]['r_local_x']}, {rows[0]['r_local_y']}]")


def list_videos(videos_dir):
    """List all available action CSVs."""
    csvs = sorted(glob.glob(os.path.join(videos_dir, "*_actions.csv")))
    if not csvs:
        print(f"No action CSVs found in: {videos_dir}")
        return
    print(f"\nAvailable videos ({len(csvs)}):")
    print("-" * 80)
    for i, path in enumerate(csvs):
        name = os.path.basename(path).replace("_actions.csv", "")
        rows = load_csv(path)
        print(f"  [{i+1:2d}] {name}  ({len(rows)} steps)")


def main():
    parser = argparse.ArgumentParser(description="Query per-step action logs.")
    parser.add_argument("--eval_dir",  type=str, required=True,
                        help="Path to eval directory (contains videos/ subfolder)")
    parser.add_argument("--video",     type=str, default=None,
                        help="Video name (without .mp4 or _actions.csv)")
    parser.add_argument("--steps",     type=int, default=None,
                        help="Print only first N steps")
    parser.add_argument("--from_step", type=int, default=None,
                        help="Start from this step")
    parser.add_argument("--to_step",   type=int, default=None,
                        help="End at this step (inclusive)")
    parser.add_argument("--summary",   action="store_true",
                        help="Print aggregate stats instead of per-step table")
    parser.add_argument("--index",     type=int, default=None,
                        help="Select video by index from the list")
    args = parser.parse_args()

    videos_dir = os.path.join(os.path.abspath(args.eval_dir), "videos")
    if not os.path.isdir(videos_dir):
        print(f"ERROR: videos directory not found: {videos_dir}")
        sys.exit(1)

    # No video specified — list available and exit
    if args.video is None and args.index is None:
        list_videos(videos_dir)
        print("\nUse --video <name> or --index <N> to query a specific video.")
        return

    # Select by index
    if args.index is not None:
        csvs = sorted(glob.glob(os.path.join(videos_dir, "*_actions.csv")))
        if args.index < 1 or args.index > len(csvs):
            print(f"ERROR: index {args.index} out of range (1-{len(csvs)})")
            sys.exit(1)
        csv_path = csvs[args.index - 1]
    else:
        # Strip extensions if user included them
        name = args.video.replace("_actions.csv", "").replace(".mp4", "")
        csv_path = os.path.join(videos_dir, f"{name}_actions.csv")

    if not os.path.isfile(csv_path):
        print(f"ERROR: Action CSV not found: {csv_path}")
        print("\nAvailable videos:")
        list_videos(videos_dir)
        sys.exit(1)

    rows = load_csv(csv_path)
    video_name = os.path.basename(csv_path).replace("_actions.csv", "")
    print(f"\nVideo : {video_name}")
    print(f"CSV   : {csv_path}")
    print(f"Steps : {len(rows)}")

    if args.summary:
        print_summary(rows)
    else:
        # Apply --steps as a shorthand for --to_step
        to_step = args.to_step
        if args.steps is not None:
            to_step = args.steps - 1
        print_actions(rows, from_step=args.from_step, to_step=to_step)
        # Always print summary at the end
        print_summary(rows)


if __name__ == "__main__":
    main()