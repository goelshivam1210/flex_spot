import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

SMOOTH_WINDOW = 11
TITLE_FONTSIZE = 36
AXIS_FONTSIZE = 24
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 24
LINE_WIDTH = 2.5
FILL_ALPHA = 0.20
FULL_REWARD_XMAX = 25_000

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

def best_run_for_tag(run_dirs, all_run_data, tag):
    """
    Returns (run_dir, run_name, max_val, step_at_max) for the run that has the highest value for `tag`.
    Looks across the full training trace in each run's data.json.
    """
    best = (None, None, -np.inf, None)
    for run_dir, run_data in zip(run_dirs, all_run_data):
        if tag not in run_data or run_data[tag].empty:
            continue
        df = run_data[tag].drop_duplicates(subset='steps', keep='last')
        idx = df['values'].idxmax()
        if pd.isna(idx):
            continue
        max_val = df.loc[idx, 'values']
        step_at_max = int(df.loc[idx, 'steps'])
        if max_val > best[2]:
            best = (run_dir, os.path.basename(run_dir), float(max_val), step_at_max)
    return best


def load_run_data(run_dir):
    """Loads data.json file from single run directory"""
    json_path = os.path.join(run_dir, "data.json")
    if not os.path.exists(json_path):
        print(f"Warning: No data.json found in {run_dir}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for tag, values in data.items():
            data[tag] = pd.DataFrame(values)
        return data
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def align_and_average_runs(all_run_data, tag, num_points=200):
    max_step = 0
    for run_data in all_run_data:
        if tag in run_data and not run_data[tag].empty:
            max_step = max(max_step, run_data[tag]['steps'].max())
    
    if max_step == 0:
        return None, None, None
        
    common_steps = np.linspace(0, max_step, num_points)
    
    interpolated_runs = []
    for run_data in all_run_data:
        if tag in run_data and not run_data[tag].empty:
            run_df = run_data[tag].drop_duplicates(subset='steps', keep='last').set_index('steps')
            resampled = run_df.reindex(run_df.index.union(common_steps)).interpolate('index').reindex(common_steps)
            interpolated_runs.append(resampled['values'])
            
    if not interpolated_runs:
        return None, None, None

    combined_df = pd.concat(interpolated_runs, axis=1)
    mean = combined_df.mean(axis=1)
    std = combined_df.std(axis=1)
    
    return common_steps, mean, std

def create_plots(parent_dir, output_dir, num_points=200):
    """
    Main function to find all runs, parse their JSON logs, and generate plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    run_dirs = []
    for d in os.listdir(parent_dir):
        run_path = os.path.join(parent_dir, d)
        if os.path.isdir(run_path) and 'data.json' in os.listdir(run_path):
            run_dirs.append(run_path)
    
    print(f"Found {len(run_dirs)} run directories with data.json. Parsing logs...")
    
    all_run_data = [load_run_data(run_dir) for run_dir in run_dirs]
    all_run_data = [data for data in all_run_data if data is not None]
    
    print(f"Successfully parsed {len(all_run_data)} log files.")
    
    metrics_to_plot = {
        "Success Rate (Full Arc)": "Eval/FullSuccess",
        "Average Reward (Full Arc)": "Eval/FullReward",
        "Success Rate (Short Arc)": "Eval/ShortSuccess",
        "Average Reward (Short Arc)": "Eval/ShortReward",
        "Success Rate (Dual Force Full Arc)": "Eval/DualSuccess",
        "Average Reward (Dual Force Full Arc)": "Eval/DualReward",
        "Elapsed Seconds": "Time/ElapsedSec",
        "Throughput (Steps/s, Window=10k)": "Time/StepsPerSecWindow",
        "Throughput (Steps/s, Cumulative)": "Time/StepsPerSecCumulative",
    }

    for plot_title, tag in metrics_to_plot.items():
        print(f"Generating plot for: {plot_title}")
        steps, mean, std = align_and_average_runs(all_run_data, tag, num_points)
        
        if steps is None:
            print(f"--> Skipping '{plot_title}' - no data found for this tag.")
            continue

        if plot_title == "Average Reward (Full Arc)":
            clip_mask = steps <= FULL_REWARD_XMAX
            steps = steps[clip_mask]
            mean = mean.iloc[clip_mask]
            std  = std.iloc[clip_mask]

        mean_s = mean.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        std_s  = std.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

        is_success_plot = "Success Rate" in plot_title
        if is_success_plot:
            mean_plot = mean_s.clip(lower=0.0, upper=1.0).values
            lower = (mean_s - std_s).clip(lower=0.0, upper=1.0).values
            upper = (mean_s + std_s).clip(lower=0.0, upper=1.0).values
        else:
            mean_plot = mean_s.values
            lower = (mean_s - std_s).values
            upper = (mean_s + std_s).values

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(steps, mean_s.values, label='Mean', linewidth=LINE_WIDTH)
        ax.fill_between(
            steps,
            lower, 
            upper,
            alpha=FILL_ALPHA,
            label='Std. Dev.'
        )

        xsf = ScalarFormatter(useMathText=True)
        xsf.set_scientific(True)
        xsf.set_powerlimits((0, 0))
        xsf.set_useOffset(False)
        ax.xaxis.set_major_formatter(xsf)

        if "Reward" in plot_title:
            ysf = ScalarFormatter(useMathText=True)
            ysf.set_scientific(True)
            ysf.set_powerlimits((0, 0))
            ysf.set_useOffset(False)
            ax.yaxis.set_major_formatter(ysf)

        ax.set_title(f"{plot_title}")
        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel(plot_title.split('(')[0].strip())

        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

        ax.margins(x=0, y=0.02)
        ax.set_xlim(float(steps.min()), float(steps.max()))

        if is_success_plot:
            ax.set_ylim(0.0, 1.0)

        fig.tight_layout(pad=0.1)
        filename = f"{plot_title.replace(' ', '_').replace('/', '-')}.png"
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"--> Saved plot to {output_path}")
    
    tags_of_interest = {
        "Short": "Eval/ShortSuccess",
        "Full":  "Eval/FullSuccess",
        "Dual":  "Eval/DualSuccess",
    }

    print("\n=== Best policies (max success over training) ===")
    for name, tag in tags_of_interest.items():
        run_dir, run_name, max_val, step_at_max = best_run_for_tag(run_dirs, all_run_data, tag)
        if run_dir is None:
            print(f"{name}: no data found for tag '{tag}'")
            continue
        print(f"{name}: {run_name}  |  max={max_val:.3f} at step {step_at_max}  |  models: {os.path.join(run_dir, 'models')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot aggregated results from multiple TensorBoard JSON exports.")
    parser.add_argument("parent_dir", type=str, help="Path to the parent directory containing all run folders (e.g., 'runs/').")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the generated plots.")
    parser.add_argument("--points", type=int, default=200, help="Number of points to interpolate for the x-axis.")
    args = parser.parse_args()
    
    create_plots(args.parent_dir, args.output_dir, args.points)
