#!/usr/bin/env python3
"""
compute_rmse.py

Compute RMSE between robot trajectory and planned path from saved experiment data.

Usage:
    python compute_rmse.py                                    # Compute for all experiments
    python compute_rmse.py --experiment small_box_no_handle_push  # Compute for specific experiment
    python compute_rmse.py --run-dir experiment_logs/small_box_no_handle_push/20250113_143022  # Compute for specific run

Author: Shivam Goel
Date: September 2025
"""

import argparse
import os
import json
import csv
import numpy as np
from pathlib import Path


def load_trajectory_data(csv_file):
    """Load trajectory data from CSV file."""
    positions = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'x' in row and 'y' in row:
                positions.append([float(row['x']), float(row['y'])])
    return np.array(positions)


def load_planned_path(csv_file):
    """Load planned path data from CSV file."""
    positions = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'x' in row and 'y' in row:
                positions.append([float(row['x']), float(row['y'])])
    return np.array(positions)


def compute_rmse(robot_traj, planned_path):
    """Compute Root Mean Square Error between robot trajectory and planned path."""
    if len(robot_traj) == 0 or len(planned_path) == 0:
        return float('inf')
    
    # Interpolate to same length (use shorter length)
    min_len = min(len(robot_traj), len(planned_path))
    robot_traj = robot_traj[:min_len]
    planned_path = planned_path[:min_len]
    
    # Compute RMSE for x, y coordinates
    mse = np.mean((robot_traj - planned_path) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def compute_final_position_error(robot_traj, planned_path):
    """Compute final position error."""
    if len(robot_traj) == 0 or len(planned_path) == 0:
        return float('inf')
    
    robot_final = robot_traj[-1]
    planned_final = planned_path[-1]
    
    error = np.linalg.norm(robot_final - planned_final)
    return error


def analyze_single_run(run_dir):
    """Analyze a single experiment run."""
    run_dir = Path(run_dir)
    
    # Check if required files exist
    params_file = run_dir / "params.json"
    robot_traj_file = run_dir / "robot_trajectory.csv"
    planned_path_file = run_dir / "planned_path.csv"
    
    if not all(f.exists() for f in [params_file, robot_traj_file, planned_path_file]):
        print(f"❌ Missing files in {run_dir}")
        return None
    
    # Load data
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    robot_traj = load_trajectory_data(robot_traj_file)
    planned_path = load_planned_path(planned_path_file)
    
    # Compute metrics
    rmse = compute_rmse(robot_traj, planned_path)
    final_error = compute_final_position_error(robot_traj, planned_path)
    
    # Calculate total distance traveled
    if len(robot_traj) > 1:
        distances = np.sqrt(np.sum(np.diff(robot_traj, axis=0) ** 2, axis=1))
        total_distance = np.sum(distances)
    else:
        total_distance = 0.0
    
    # Calculate average speed
    if params.get('policy_duration_seconds', 0) > 0:
        avg_speed = total_distance / params['policy_duration_seconds']
    else:
        avg_speed = 0.0
    
    results = {
        'run_dir': str(run_dir),
        'experiment': params['experiment'],
        'timestamp': params['timestamp'],
        'arc_angle_degrees': params.get('arc_angle_degrees', 'N/A'),
        'target_distance': params.get('target_distance', 'N/A'),
        'action_scale': params.get('action_scale', 'N/A'),
        'max_steps': params.get('max_steps', 'N/A'),
        'steps_taken': params.get('steps_taken', 0),
        'success': params.get('success', False),
        'policy_duration': params.get('policy_duration_seconds', 0),
        'rmse': rmse,
        'final_position_error': final_error,
        'total_distance_traveled': total_distance,
        'average_speed': avg_speed
    }
    
    return results


def print_results(results):
    """Print analysis results in a nice format."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT ANALYSIS: {results['experiment']}")
    print(f"Run: {results['timestamp']}")
    print(f"{'='*80}")
    
    print(f"Parameters:")
    print(f"  Arc Angle: {results['arc_angle_degrees']}°")
    print(f"  Target Distance: {results['target_distance']}m")
    print(f"  Action Scale: {results['action_scale']}")
    print(f"  Max Steps: {results['max_steps']}")
    
    print(f"\nResults:")
    print(f"  Success: {'✅' if results['success'] else '❌'}")
    print(f"  Steps Taken: {results['steps_taken']}")
    print(f"  Policy Duration: {results['policy_duration']:.2f}s")
    
    print(f"\nAccuracy Metrics:")
    print(f"  RMSE: {results['rmse']:.4f}m")
    print(f"  Final Position Error: {results['final_position_error']:.4f}m")
    
    print(f"\nMovement Metrics:")
    print(f"  Total Distance Traveled: {results['total_distance_traveled']:.2f}m")
    print(f"  Average Speed: {results['average_speed']:.4f} m/s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compute RMSE from experiment data')
    parser.add_argument('--experiment', help='Analyze specific experiment type')
    parser.add_argument('--run-dir', help='Analyze specific run directory')
    parser.add_argument('--experiment-logs-dir', default='experiment_logs', 
                        help='Directory containing experiment logs')
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Analyze specific run
        results = analyze_single_run(args.run_dir)
        if results:
            print_results(results)
        else:
            print("Failed to analyze run")
    
    elif args.experiment:
        # Analyze specific experiment type
        experiment_dir = Path(args.experiment_logs_dir) / args.experiment
        if not experiment_dir.exists():
            print(f"Experiment directory not found: {experiment_dir}")
            return
        
        print(f"Analyzing all runs for experiment: {args.experiment}")
        
        runs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if not runs:
            print("No runs found for this experiment")
            return
        
        all_results = []
        for run_dir in sorted(runs):
            results = analyze_single_run(run_dir)
            if results:
                all_results.append(results)
        
        if not all_results:
            print("No valid runs found")
            return
        
        # Print summary for all runs
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {args.experiment.upper()}")
        print(f"Total Runs: {len(all_results)}")
        print(f"{'='*80}")
        
        successful_runs = [r for r in all_results if r['success']]
        print(f"Successful Runs: {len(successful_runs)}/{len(all_results)}")
        
        if successful_runs:
            rmse_values = [r['rmse'] for r in successful_runs]
            final_errors = [r['final_position_error'] for r in successful_runs]
            durations = [r['policy_duration'] for r in successful_runs]
            
            print(f"\nSuccessful Run Statistics:")
            print(f"  Average RMSE: {np.mean(rmse_values):.4f}m ± {np.std(rmse_values):.4f}m")
            print(f"  Average Final Error: {np.mean(final_errors):.4f}m ± {np.std(final_errors):.4f}m")
            print(f"  Average Duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
        
        # Print individual results
        for results in all_results:
            print_results(results)
    
    else:
        # Analyze all experiments
        experiment_logs_dir = Path(args.experiment_logs_dir)
        if not experiment_logs_dir.exists():
            print(f"Experiment logs directory not found: {experiment_logs_dir}")
            return
        
        print("Analyzing all experiments...")
        
        all_results = []
        for experiment_dir in experiment_logs_dir.iterdir():
            if experiment_dir.is_dir():
                runs = [d for d in experiment_dir.iterdir() if d.is_dir()]
                for run_dir in runs:
                    results = analyze_single_run(run_dir)
                    if results:
                        all_results.append(results)
        
        if not all_results:
            print("No experiment data found")
            return
        
        # Group by experiment type
        experiments = {}
        for results in all_results:
            exp_name = results['experiment']
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(results)
        
        # Print summary for each experiment type
        for exp_name, exp_results in experiments.items():
            print(f"\n{'='*80}")
            print(f"EXPERIMENT: {exp_name.upper()}")
            print(f"Total Runs: {len(exp_results)}")
            print(f"{'='*80}")
            
            successful = [r for r in exp_results if r['success']]
            print(f"Success Rate: {len(successful)}/{len(exp_results)} ({len(successful)/len(exp_results)*100:.1f}%)")
            
            if successful:
                rmse_values = [r['rmse'] for r in successful]
                print(f"Average RMSE: {np.mean(rmse_values):.4f}m ± {np.std(rmse_values):.4f}m")
                print(f"RMSE Range: {np.min(rmse_values):.4f}m - {np.max(rmse_values):.4f}m")


if __name__ == '__main__':
    main()


