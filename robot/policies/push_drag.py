"""
push_drag.py

Push/drag manipulation with Spot robot using path-following policy

The system works in 3 s:
1. Object detection, grasp, and path definition
2. Execute path-following policy for push/drag
3. Release and cleanup

Example usage:
    python push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_1robot --target-distance 1.0

other usage examples:
python push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_1robot
python push_drag.py --hostname 192.168.1.100 --experiment small_box_no_handle_1robot --autonomous-detection
python push_drag.py --hostname 192.168.1.100 --experiment large_box_no_handle_2robots --autonomous-detection --robot-side right

Author: Shivam Goel
Date: September 2025
"""

import argparse
import sys
import time
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from datetime import datetime  

# Import Spot SDK modules
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose

# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from flex.interactive_perception import InteractivePerception
from flex.policy_manager import PolicyManager

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "small_box_no_handle_push": {
        "description": "Small box without handle, push task",
        "grasp_strategy": "edge_grasp",
        "task_type": "push",
        "expected_distance": 1.0,
        "max_force_scale": 0.8
    },
    "small_box_no_handle_drag": {
        "description": "Small box without handle, drag task", 
        "grasp_strategy": "edge_grasp",
        "task_type": "drag",
        "expected_distance": 1.0,
        "max_force_scale": 0.8
    },
    "small_box_handle_push": {
        "description": "Small box with handle, push task",
        "grasp_strategy": "handle_grasp", 
        "task_type": "push",
        "expected_distance": 1.0,
        "max_force_scale": 0.8
    },
    "small_box_handle_drag": {
        "description": "Small box with handle, drag task",
        "grasp_strategy": "handle_grasp",
        "task_type": "drag",
        "expected_distance": 1.0,
        "max_force_scale": 0.6
    }
}


def plot_trajectory(robot_positions, path_points, experiment_name, save_path=None):
    """
    Plot robot trajectory and planned path in vision frame.
    
    Args:
        robot_positions: List of (x, y, yaw) tuples representing robot positions
        path_points: Array of planned path points (Nx3 array with x, y, z)
        experiment_name: Name of the experiment for title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Convert robot positions to arrays
    robot_positions = np.array(robot_positions)
    robot_x = robot_positions[:, 0]
    robot_y = robot_positions[:, 1]
    robot_yaw = robot_positions[:, 2]
    
    # Plot planned path
    plt.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2, 
             label='Planned Path', alpha=0.7)
    plt.scatter(path_points[0, 0], path_points[0, 1], color='green', 
                s=100, marker='o', label='Start Point', zorder=5)
    plt.scatter(path_points[-1, 0], path_points[-1, 1], color='red', 
                s=100, marker='s', label='End Point', zorder=5)
    
    # Plot robot trajectory
    plt.plot(robot_x, robot_y, 'r-', linewidth=2, 
             label='Robot Trajectory', alpha=0.8)
    plt.scatter(robot_x[0], robot_y[0], color='darkgreen', 
                s=100, marker='^', label='Robot Start', zorder=5)
    plt.scatter(robot_x[-1], robot_y[-1], color='darkred', 
                s=100, marker='v', label='Robot End', zorder=5)
    
    # Add arrows to show robot orientation at key points
    step_size = max(1, len(robot_x) // 10)  # Show arrows every 10% of trajectory
    for i in range(0, len(robot_x), step_size):
        dx = 0.1 * np.cos(robot_yaw[i])
        dy = 0.1 * np.sin(robot_yaw[i])
        plt.arrow(robot_x[i], robot_y[i], dx, dy, 
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange', alpha=0.7)
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title(f'Robot Trajectory vs Planned Path - {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add text box with statistics
    total_distance = np.sum(np.sqrt(np.diff(robot_x)**2 + np.diff(robot_y)**2))
    path_length = np.sum(np.sqrt(np.diff(path_points[:, 0])**2 + np.diff(path_points[:, 1])**2))
    final_error = np.sqrt((robot_x[-1] - path_points[-1, 0])**2 + (robot_y[-1] - path_points[-1, 1])**2)
    
    stats_text = f'Total Distance: {total_distance:.2f}m\n'
    stats_text += f'Path Length: {path_length:.2f}m\n'
    stats_text += f'Final Error: {final_error:.2f}m\n'
    stats_text += f'Steps: {len(robot_x)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_position_over_time(robot_positions, experiment_name, save_path=None):
    """
    Plot robot position components over time.
    
    Args:
        robot_positions: List of (x, y, yaw) tuples representing robot positions
        experiment_name: Name of the experiment for title
        save_path: Optional path to save the plot
    """
    robot_positions = np.array(robot_positions)
    time_steps = np.arange(len(robot_positions))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # X position
    ax1.plot(time_steps, robot_positions[:, 0], 'b-', linewidth=2)
    ax1.set_ylabel('X Position (m)')
    ax1.set_title(f'Robot Position Over Time - {experiment_name}')
    ax1.grid(True, alpha=0.3)
    
    # Y position
    ax2.plot(time_steps, robot_positions[:, 1], 'g-', linewidth=2)
    ax2.set_ylabel('Y Position (m)')
    ax2.grid(True, alpha=0.3)
    
    # Yaw angle
    ax3.plot(time_steps, np.degrees(robot_positions[:, 2]), 'r-', linewidth=2)
    ax3.set_ylabel('Yaw Angle (degrees)')
    ax3.set_xlabel('Time Step')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position plot saved to: {save_path}")
    
    plt.show()


def user_confirm_step(step_description):
    """Ask user to confirm before executing each step. Return False if user wants to quit."""
    print(f"\n{'='*60}")
    print(f"READY FOR: {step_description}")
    print("Press ENTER to continue, 's' + ENTER to skip, or 'q' + ENTER to quit safely")
    print('='*60)
    
    user_input = input().strip().lower()
    
    if user_input == 'q':
        print("User requested safe shutdown...")
        return False
    elif user_input == 's':
        print("User skipped this step...")
        return 'skip'
    
    return True


def cleanup_and_exit(spot, dock_id=521):
    """Safely cleanup and exit the program with full shutdown."""
    try:
        print('Performing safe shutdown...')
        
        # Release object and stow arm
        try:
            print('Releasing object and stowing arm...')
            spot.open_gripper()
            time.sleep(1)
            spot.stow_arm()
            print('Arm stowed successfully')
        except Exception as e:
            print(f'Could not stow arm: {e}')
        
        # Dock robot
        try:
            print('Attempting to dock robot...')
            spot.dock(dock_id=dock_id)
            print('Robot docked successfully')
        except Exception as e:
            print(f'Could not dock robot: {e}')
        
        print('Robot safely shut down.')
        
    except Exception as e:
        print(f'Error during cleanup: {e}')


def detect_and_grasp_object(spot, config, experiment_config):
    """Detect target object and execute appropriate grasp strategy."""
    print(f'Starting object detection for: {experiment_config["description"]}')
    
    # Take picture for object detection
    print(f'Taking picture with camera: {config.image_source}')
    
    color_img, depth_img = spot.take_picture(
        color_src=config.image_source,
        depth_src=config.depth_source,
        save_images=True
    )
    
    if color_img is None:
        raise Exception('Failed to capture image')
    
    # Choose detection method based on flag
    if config.autonomous_detection:
        print('Using autonomous box detection with OWL-v2 + SAM...')
        try:
            # target_pixel = SpotPerception.find_grasp_sam(
            #     color_img, depth_img, 
            #     left=(config.robot_side == 'left'),
            #     conf=0.15,
            #     max_distance_m=3.0
            # )
            target_pixel = SpotPerception.get_red_object_center_of_mass(color_img)
            if target_pixel is None:
                print('Autonomous detection failed, falling back to manual selection')
                target_pixel = SpotPerception.get_target_from_user(color_img)
        except Exception as e:
            print(f'Autonomous detection error: {e}, using manual selection')
            target_pixel = SpotPerception.get_target_from_user(color_img)
    else:
        print('Using manual target selection...')
        # Get user to select target object or grasp point
        if experiment_config["grasp_strategy"] == "handle_grasp":
            print('Click on the handle you want to grasp...')
        else:
            print('Click on the edge/side of the box you want to grasp...')
        
        target_pixel = SpotPerception.get_target_from_user(color_img)
    
    if target_pixel is None:
        raise Exception('No target selected')
    
    print(f'Target selected at pixel: {target_pixel}')
    
    # Execute grasp
    spot.open_gripper()
    success = spot.grasp_edge(target_pixel, img_src=config.image_source)
    
    if not success:
        raise Exception('Failed to grasp object')

    if not spot.check_grip():
        raise Exception('Failed to grasp object')
    else:
        print('Object grasped successfully')
    
    print('Holding object for 3 seconds...')
    time.sleep(3)
    
    return target_pixel

def define_path_and_direction(spot, config, experiment_config):
    """Define curved arc path for push/drag operation."""
    print('Defining curved arc path for push/drag operation')
    
    # Get current hand position as start point
    robot_state = spot._client._state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
    
    if vision_T_hand is None:
        raise Exception("Could not get hand pose for path definition")
    
    start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
    current_x, current_y, current_yaw = spot.get_current_pose()
    
    # Define arc parameters based on target distance
    arc_radius = config.target_distance  # Use target distance as radius
    
    if experiment_config["task_type"] == "push":
        # Push: forward arc curving right
        # start_angle = -np.pi/6  # -30 degrees
        start_angle = -np.pi/4
        start_angle = 0 
        # end_angle = np.pi/6     # +30 degrees
        end_angle = np.pi/3 
        arc_center = start_position[:2]  # Center at current position
    else:  # drag
        # Drag: backward arc curving left  
        start_angle = np.pi - np.pi/6   # 150 degrees
        end_angle = np.pi + np.pi/6     # 210 degrees
        arc_center = start_position[:2]
    
    # Generate arc path points
    interactive_perception = InteractivePerception()
    path_points_2d = interactive_perception.generate_arc_path(
        center=arc_center,
        radius=arc_radius,
        start_angle=start_angle,
        end_angle=end_angle,
        num_points=max(10, int(config.target_distance * 20))
    )
    
    # Add z-coordinate (keep constant height)
    path_points = np.column_stack([path_points_2d, np.full(len(path_points_2d), start_position[2])])
    
    end_position = path_points[-1]
    direction_vector = (end_position - start_position)[:2]  # 2D direction
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    print(f'Arc path defined: {len(path_points)} points from {start_position} to {end_position}')
    print(f'Arc radius: {arc_radius:.2f}m, Angles: {np.degrees(start_angle):.1f}° to {np.degrees(end_angle):.1f}°')
    print(f'Direction: {experiment_config["task_type"]} (arc curve)')
    
    return {
        'start_position': start_position,
        'end_position': end_position,
        'path_points': path_points,
        'direction_vector': direction_vector
    }

def execute_path_following_policy(spot, config, experiment_config, path_info):
    """Execute path-following policy for push/drag operation."""
    print('Executing path-following policy')
    
    # Load path-following policy (rotation policy)
    policy_manager = PolicyManager()
    
    try:
        policy = policy_manager.load_path_following_policy(config.policy_path, config.model_name)
    except FileNotFoundError as e:
        raise Exception(f"Path-following policy loading failed: {e}")
    
    # Initialize path-following state tracking
    interactive_perception = InteractivePerception()
    path_points = path_info['path_points']
    start_position = path_info['start_position']
    
    # Initialize position tracking
    robot_positions = []
    
    print(f"Starting path-following execution")
    print(f"Max steps: {config.max_steps}, Action scale: {config.action_scale}")
    
    # Policy execution loop
    manipulation_success = False
    closest_path_idx = 0
    
    for step in range(config.max_steps):
        # Get current hand position
        robot_state = spot._client._state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        current_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        current_hand_pos = np.array([current_hand_pose.x, current_hand_pose.y, current_hand_pose.z])
        
        # Get current robot orientation
        current_x, current_y, current_yaw = spot.get_current_pose()
        
        # Track robot position in vision frame
        robot_positions.append((current_x, current_y, current_yaw))
        
        # estimate the box center from grasp
        box_center = interactive_perception.estimate_box_center_from_grasp(
            current_hand_pos, 
            experiment_config["grasp_strategy"], 
            box_dimensions={"width": 0.6, "depth": 0.4, "height": 0.8}, 
            current_yaw=current_yaw
        )

        # Construct state vector for path-following policy
        state_vector = interactive_perception.construct_path_following_state(
            current_hand_pos, path_points, current_yaw, closest_path_idx
        )
        
        # Get action from policy
        action = policy.select_action(state_vector)
        if len(action.shape) > 1:
            action = action.flatten()
        
        # Scale and apply action
        scaled_action = action * config.action_scale * experiment_config["max_force_scale"]
        
        print(f"Step {step+1}/{config.max_steps}")
        print(f"State: {state_vector}")
        print(f"Action: {scaled_action}")
        
        # Execute movement based on task type
        try:
            if experiment_config["task_type"] == "push":                
                # For pushing: move robot body while maintaining arm position
                print('Pushing object by moving robot...')
                dx = scaled_action[0]
                dy = scaled_action[1]
                d_yaw = scaled_action[2] if len(scaled_action) > 2 else 0
                # d_yaw = 0
                dt = 2
                vx = abs(dx/dt)
                vy = abs(dy/dt)
                v_yaw = abs(d_yaw/dt)

                spot.push_object_from_sim(
                    dx=dx, 
                    dy=dy, 
                    d_yaw=d_yaw,
                    vx=vx,
                    vy=vy,
                    v_yaw=v_yaw,
                    dt=2.0
                )

                # print('Pushing object by moving robot...')
                # target_position = current_hand_pos - scaled_action[:3]  # Note the minus sign
                # target_pose = SE3Pose(
                #     x=target_position[0],
                #     y=target_position[1], 
                #     z=target_position[2],
                #     rot=current_hand_pose.rot
                # )
                
                # arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                #     target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
                # )
                # cmd_id = spot._client._command_client.robot_command(arm_cmd)
                # block_until_arm_arrives(spot._client._command_client, cmd_id, timeout_sec=3.0)
                # For pushing: move robot body while maintaining arm position
            else:  # drag
                # For dragging: move arm to pull object
                print('Moving arm to drag object...')
                target_position = current_hand_pos + scaled_action[:3]  # Note the minus sign
                target_pose = SE3Pose(
                    x=target_position[0],
                    y=target_position[1], 
                    z=target_position[2],
                    rot=current_hand_pose.rot
                )
                
                arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                    target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
                )
                cmd_id = spot._client._command_client.robot_command(arm_cmd)
                block_until_arm_arrives(spot._client._command_client, cmd_id, timeout_sec=3.0)
            
            # Update closest path index for state computation
            closest_path_idx = interactive_perception.update_closest_path_index(box_center, path_points, closest_path_idx)
            
            # Check success
            if check_path_following_success(current_hand_pos, path_info, config.success_distance):
                print(f"Path following completed successfully in {step+1} steps!")
                manipulation_success = True
                break
                
        except Exception as e:
            print(f"Error in step {step+1}: {e}")
            break
    
    print(f"Policy execution completed. Success: {manipulation_success}")
    time.sleep(2)  # Hold position to observe
    
    # Generate plots (unless disabled)
    if not hasattr(config, 'no_plots') or not config.no_plots:
        print("Generating trajectory plots...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config.experiment}_{timestamp}"
        
        # Create plots directory if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot trajectory comparison
        trajectory_plot_path = os.path.join(plots_dir, f"trajectory_{experiment_name}.png")
        plot_trajectory(robot_positions, path_points, experiment_name, trajectory_plot_path)
        
        # Plot position over time
        position_plot_path = os.path.join(plots_dir, f"position_over_time_{experiment_name}.png")
        plot_position_over_time(robot_positions, experiment_name, position_plot_path)
        
        print(f"Plots saved to {plots_dir}/ directory")
    else:
        print("Plotting disabled by user")
    
    return manipulation_success


def check_path_following_success(current_pos, path_info, success_threshold):
    """Check if path following succeeded."""
    end_position = path_info['end_position']
    distance_to_goal = np.linalg.norm(current_pos - end_position)
    
    success = distance_to_goal <= success_threshold
    print(f"Distance to goal: {distance_to_goal:.3f}m (threshold: {success_threshold:.3f}m)")
    
    return success


def push_drag_main(config):
    """Main function for push/drag manipulation."""
    
    # Validate experiment configuration
    if config.experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {config.experiment}")
    
    experiment_config = EXPERIMENT_CONFIGS[config.experiment]
    print(f"Starting experiment: {experiment_config['description']}")
    
    # Initialize robot
    spot = Spot(id="PushDragger", hostname=config.hostname)
    spot.start()
    
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Initialize
            print('Initializing robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            saved_yaw = spot.save_initial_yaw()
            
            #  1: Object detection and grasp
            step_result = user_confirm_step("Detect and grasp target object")
            if step_result == False:
                return cleanup_and_exit(spot, config.dock_id)
            elif step_result != 'skip':
                detect_and_grasp_object(spot, config, experiment_config)
            
            #  2: Path definition
            step_result = user_confirm_step("Define push/drag path")
            if step_result == False:
                return cleanup_and_exit(spot, config.dock_id)
            elif step_result != 'skip':
                path_info = define_path_and_direction(spot, config, experiment_config)
            else:
                # Default path info for skip
                path_info = {'start_position': np.array([0,0,0]), 'end_position': np.array([1,0,0])}
            
            #  3: Policy execution
            step_result = user_confirm_step("Execute path-following policy")
            if step_result == False:
                return cleanup_and_exit(spot, config.dock_id)
            elif step_result != 'skip':
                policy_success = execute_path_following_policy(spot, config, experiment_config, path_info)
            else:
                policy_success = True
            
            #  4: Cleanup
            step_result = user_confirm_step("Release object and dock")
            if step_result == False:
                return cleanup_and_exit(spot, config.dock_id)
            cleanup_and_exit(spot, config.dock_id)
            
            if policy_success:
                print('Push/drag operation completed successfully!')
            else:
                print('Push/drag operation completed with issues')
            
        except KeyboardInterrupt:
            print('Interrupted by user. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)
        except Exception as e:
            print(f'Error occurred: {e}. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Push/drag manipulation with Spot robot')
    parser.add_argument('--hostname', required=True, help='Spot robot hostname or IP')
    parser.add_argument('--experiment', required=True, choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='Experiment configuration to run')
    parser.add_argument('--image-source', 
                        default='hand_color_image',
                        help='Camera source for color images')
    parser.add_argument('--depth-source',
                        default='hand_depth_in_hand_color_frame', 
                        help='Camera source for depth images')
    parser.add_argument('--dock-id',
                        type=int,
                        default=521,
                        help='Docking station ID')
    parser.add_argument('--max-steps',
                        type=int,
                        default=15,
                        help='Maximum policy execution steps')
    parser.add_argument('--action-scale', 
                        type=float,
                        default=0.05,
                        help='Scale factor for policy actions')
    parser.add_argument('--target-distance',
                        type=float,
                        default=1.0,
                        help='Target distance to push/drag (meters)')
    parser.add_argument('--success-distance',
                        type=float, 
                        default=0.2,
                        help='Success threshold distance from target (meters)')
    parser.add_argument('--policy-path',
                        default='models/rotation',
                        help='Path to trained path-following policy')
    parser.add_argument('--model-name',
                        default='best_model',
                        help='Name of the model to load (best_model, final_model, etc.)')
    
    parser.add_argument('--autonomous-detection', action='store_true',
                        help='Use autonomous box detection instead of manual selection')
    parser.add_argument('--robot-side', choices=['left', 'right'], default='left',
                        help='Robot side for multi-robot coordination (affects grasp point selection)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable trajectory plotting')
    
    options = parser.parse_args()
    
    try:
        push_drag_main(options)
        return True
    except Exception as exc:
        print(f'Push/drag operation failed: {exc}')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)



