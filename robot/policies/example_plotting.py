#!/usr/bin/env python3
"""
Example script demonstrating the trajectory plotting functionality in push_drag.py

This script shows how to use the plotting functions independently for testing
or for analyzing saved trajectory data.
"""

import numpy as np
import matplotlib.pyplot as plt
from push_drag import plot_trajectory, plot_position_over_time

def generate_sample_data():
    """Generate sample robot trajectory and path data for demonstration."""
    
    # Generate a curved path (arc)
    t = np.linspace(0, np.pi/2, 20)
    path_x = 2 * np.cos(t)
    path_y = 2 * np.sin(t)
    path_z = np.ones_like(t) * 0.5
    path_points = np.column_stack([path_x, path_y, path_z])
    
    # Generate robot trajectory with some noise and deviation
    robot_x = path_x + np.random.normal(0, 0.1, len(t))
    robot_y = path_y + np.random.normal(0, 0.1, len(t))
    robot_yaw = t + np.random.normal(0, 0.1, len(t))
    robot_positions = list(zip(robot_x, robot_y, robot_yaw))
    
    return robot_positions, path_points

def main():
    """Demonstrate plotting functionality with sample data."""
    
    print("Generating sample trajectory data...")
    robot_positions, path_points = generate_sample_data()
    
    print("Creating trajectory plots...")
    
    # Plot trajectory comparison
    plot_trajectory(
        robot_positions=robot_positions,
        path_points=path_points,
        experiment_name="Sample Experiment",
        save_path="sample_trajectory.png"
    )
    
    # Plot position over time
    plot_position_over_time(
        robot_positions=robot_positions,
        experiment_name="Sample Experiment", 
        save_path="sample_position_over_time.png"
    )
    
    print("Sample plots created: sample_trajectory.png and sample_position_over_time.png")

if __name__ == "__main__":
    main()
