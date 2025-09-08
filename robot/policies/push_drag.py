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
    "small_box_no_handle_1robot": {
        "description": "Small box without handle, single robot",
        "grasp_strategy": "edge_grasp",
        "task_type": "push",
        "expected_distance": 1.0,
        "max_force_scale": 0.8
    },
    "large_box_no_handle_2robots": {
        "description": "Large box without handle, dual robot (coordinated)",
        "grasp_strategy": "edge_grasp", 
        "task_type": "push",
        "expected_distance": 1.5,
        "max_force_scale": 1.0
    },
    "small_box_handle_1robot": {
        "description": "Small box with handle, single robot",
        "grasp_strategy": "handle_grasp",
        "task_type": "drag", 
        "expected_distance": 1.0,
        "max_force_scale": 0.6
    },
    "large_box_handle_2robots": {
        "description": "Large box with handle, dual robot (coordinated)",
        "grasp_strategy": "handle_grasp",
        "task_type": "drag",
        "expected_distance": 1.5,
        "max_force_scale": 0.8
    }
}


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
            target_pixel = SpotPerception.find_grasp_sam(
                color_img, depth_img, 
                left=(config.robot_side == 'left'),
                conf=0.15,
                max_distance_m=3.0
            )
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
    
    print('Object grasped successfully')
    print('Holding object for 3 seconds...')
    time.sleep(3)
    
    return target_pixel

def define_path_and_direction(spot, config, experiment_config):
    """Define straight-line path for push/drag operation."""
    print('Defining path for push/drag operation')
    
    # Get current hand position as start point
    robot_state = spot._client._state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
    
    if vision_T_hand is None:
        raise Exception("Could not get hand pose for path definition")
    
    start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
    
    # Get current robot orientation to define forward direction
    current_x, current_y, current_yaw = spot.get_current_pose()
    
    # Define path direction based on experiment type and robot orientation
    if experiment_config["task_type"] == "push":
        # Push: move forward relative to robot
        direction_vector = np.array([
            math.cos(current_yaw),
            math.sin(current_yaw), 
            0.0
        ])
    else:  # drag
        # Drag: move backward relative to robot
        direction_vector = np.array([
            -math.cos(current_yaw),
            -math.sin(current_yaw),
            0.0
        ])
    
    # Define target end position
    target_distance = config.target_distance
    end_position = start_position + (direction_vector * target_distance)
    
    # Use InteractivePerception to generate path points
    interactive_perception = InteractivePerception()
    path_points = interactive_perception.generate_straight_line_path(
        start_position, end_position, num_points=max(10, int(target_distance * 20))
    )
    
    print(f'Path defined: {len(path_points)} points from {start_position} to {end_position}')
    print(f'Total distance: {target_distance:.2f}m')
    print(f'Direction: {experiment_config["task_type"]} ({direction_vector[:2]})')
    
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
                spot.push_object(
                    dx=scaled_action[0], 
                    dy=scaled_action[1], 
                    d_yaw=scaled_action[2] if len(scaled_action) > 2 else 0,
                    dt=2.0
                )
            else:  # drag
                # For dragging: move arm to pull object
                target_position = current_hand_pos + scaled_action[:3]
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
            closest_path_idx = interactive_perception.update_closest_path_index(current_hand_pos, path_points, closest_path_idx)
            
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
                        default=50,
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