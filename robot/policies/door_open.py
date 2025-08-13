"""
Open door with Spot robot using intelligent policy-based manipulation

The system works in 3 phases:
1. Phase 1: Walk to door, grasp handle, perform wiggle analysis
2. Phase 2: Execute policy-based door opening
3. Phase 3: Release and cleanup

Example usage:
    python door_open.py --hostname 192.168.1.100 --max-steps 15 --action-scale 0.05

Author: Shivam Goel
Date: July 2025
"""

import argparse
import sys
import time
import numpy as np
import math

# Import Spot SDK modules
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose

# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from flex.interactive_perception import InteractivePerception
from flex.policy_manager import PolicyManager


def user_confirm_step(step_description):
    """Ask user to confirm before executing each step. Return False if user wants to quit."""
    print(f"\n{'='*60}")
    print(f"READY FOR: {step_description}")
    print("Press ENTER to continue, or 'q' + ENTER to quit safely")
    print('='*60)
    
    user_input = input().strip().lower()
    
    if user_input == 'q':
        print("User requested safe shutdown...")
        return False
    
    return True


def cleanup_and_exit(spot, dock_id=521):
    """Safely cleanup and exit the program with full shutdown."""
    try:
        print('Performing safe shutdown...')
        
        # Release handle and stow arm
        try:
            print('Releasing handle and stowing arm...')
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


def walk_to_door_and_grasp(spot, config):
    """Walk to door location and grasp handle."""
    print('Walking to door and grasping handle')
    
    # Take picture for door/handle selection
    print(f'Taking picture with camera: {config.image_source}')
    
    color_img, depth_img = spot.take_picture(
        color_src=config.image_source,
        depth_src=config.depth_source,
        save_images=True
    )
    
    if color_img is None:
        raise Exception('Failed to capture image')
    
    # Get user to click on handle location
    print('Click on the door handle you want to grasp...')
    target_pixel = SpotPerception.get_target_from_user(color_img)
    
    if target_pixel is None:
        raise Exception('No target selected')
    
    print(f'Handle target selected at pixel: {target_pixel}')
    
    # Use existing grasp method
    spot.open_gripper()
    success = spot.grasp_edge(target_pixel, img_src=config.image_source)
    
    if not success:
        raise Exception('Failed to grasp handle')
    
    print('Handle grasped successfully')
    print('Holding handle for 3 seconds...')
    time.sleep(3)
    
    return target_pixel


def analyze_door_joint(spot, config):
    """Perform wiggle analysis for joint and state estimation for policy selection and execution."""
    print('Performing wiggle analysis to determine door joint type')
    
    # Use the existing InteractiveSpotController wiggle logic
    interactive_perception = InteractivePerception()
    
    # Get current hand position
    robot_state_client = spot._client._state_client
    robot_state = robot_state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
    if vision_T_hand is None:
        raise Exception("Could not get hand pose for wiggle analysis")
        
    start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
    print(f"Starting wiggle from position: {start_position}")
    
    # Generate wiggle positions
    wiggle_positions = interactive_perception.generate_wiggle_positions(start_position)
    print(f"Generated {len(wiggle_positions)} wiggle positions")
    
    # Execute wiggle movements (using existing logic from original code)
    trajectory = []
    
    for i, target_pos in enumerate(wiggle_positions):
        print(f"Moving to position {i+1}/{len(wiggle_positions)}")
        # spot.close_gripper()
        
        try:
            # Create target pose (keep original orientation)
            target_pose = SE3Pose(
                x=target_pos[0], 
                y=target_pos[1], 
                z=target_pos[2],
                rot=vision_T_hand.rot
            )
            
            # Send arm command
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
            )
            cmd_id = spot._client._command_client.robot_command(arm_cmd)
            block_until_arm_arrives(spot._client._command_client, cmd_id, timeout_sec=3.0)
            
            # Get actual achieved position
            time.sleep(0.5)
            robot_state = robot_state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            actual_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            
            if actual_hand_pose:
                actual_position = np.array([actual_hand_pose.x, actual_hand_pose.y, actual_hand_pose.z])
                trajectory.append(actual_position)
            else:
                trajectory.append(target_pos)
                
        except Exception as e:
            print(f"Error during movement {i+1}: {e}")
            trajectory.append(target_pos)
    
    trajectory = np.array(trajectory)
    print(f"Collected trajectory with {len(trajectory)} points")
    spot.close_gripper()
    
    # Analyze trajectory
    joint_type, joint_params = interactive_perception.analyze_trajectory_and_estimate_joint(trajectory)

    # We add this overiding logic as the prismatic joint on the box is not totally prosmatic hence we need to work more on the estimation.
    if hasattr(config, 'force_joint_type') and config.force_joint_type:
        print(f"\nOverriding detected joint type with user preference: {config.force_joint_type}")
        
        if config.force_joint_type == "prismatic":
            # Use prismatic analysis results
            prismatic_error, prismatic_axis = interactive_perception.prismatic_error_analysis(trajectory)
            interactive_perception.joint_type = "prismatic"
            interactive_perception.joint_params = {"axis": prismatic_axis, "error": prismatic_error}
            joint_type = "prismatic"
            joint_params = interactive_perception.joint_params    
    print(f"Final joint type: {interactive_perception.joint_type}")
    
    print(f"\nJoint Analysis Results:")
    print(f"Estimated joint type: {joint_type}")
    print(f"Joint parameters: {joint_params}")
    
    return interactive_perception


def execute_policy_opening(spot, config, interactive_perception):
    """Execute FLEX policy-based door opening."""
    print('Executing intelligent policy-based door opening')
    
    # Load policy (existing logic)
    policy_manager = PolicyManager()
    joint_type = interactive_perception.joint_type
    
    try:
        policy = policy_manager.load_policy(joint_type)
    except FileNotFoundError as e:
        raise Exception(f"Policy loading failed: {e}")
    
    # Get initial hand position
    robot_state = spot._client._state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    initial_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
    initial_hand_pos = np.array([initial_hand_pose.x, initial_hand_pose.y, initial_hand_pose.z])
    
    current_yaw = spot.get_current_pose()[2]
    
    print(f"Starting policy execution with {joint_type} joint")
    print(f"Max steps: {config.max_steps}, Action scale: {config.action_scale}")
    
    # Policy execution loop (existing logic)
    manipulation_success = False
    
    for step in range(config.max_steps):
        # Get current hand position
        robot_state = spot._client._state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        current_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        current_hand_pos = np.array([current_hand_pose.x, current_hand_pose.y, current_hand_pose.z])
        
        # Get state and action
        state_vector = interactive_perception.construct_state_vector(current_hand_pos, initial_hand_pos)
        action = policy.select_action(state_vector)
        
        if len(action.shape) > 1:
            action = action.flatten()
        action[0] = -action[0]  # Invert x action direction for Spot
        target_position = None
        if interactive_perception.joint_type == "prismatic":
            sliding_axis = interactive_perception.joint_params["axis"]
            sliding_axis = sliding_axis / np.linalg.norm(sliding_axis)  # Normalize
            
            # Project the 3D action onto the 1D sliding direction
            action_magnitude = np.dot(action, sliding_axis)
            constrained_action = action_magnitude * sliding_axis
            
            print(f"Original action: {action}")
            print(f"Sliding axis: {sliding_axis}")
            print(f"Constrained action: {constrained_action}")
            
            target_position = current_hand_pos + (constrained_action * config.action_scale)
        else:
            # For revolute joints, use original 3D action
            target_position = current_hand_pos + (action * config.action_scale)
        
        # target_position = current_hand_pos + (action * config.action_scale)
        
        print(f"Step {step+1}/{config.max_steps}: Target={target_position}")
        
        # Execute movement
        try:
            target_pose = SE3Pose(
                x=target_position[0],
                y=target_position[1], 
                z=target_position[2],
                rot=current_hand_pose.rot
            )
            
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
            )
            if interactive_perception.joint_type == "prismatic":
                follow_arm_command = RobotCommandBuilder.follow_arm_command()
                command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_cmd)
                cmd_id = spot._client._command_client.robot_command(command)
                block_until_arm_arrives(spot._client._command_client, cmd_id, timeout_sec=3.0)
            else:
                deltas = action * config.action_scale
                d_yaw = np.radians(config.success_angle)/config.max_steps
                print(f"deltas = {deltas} dyaws = {d_yaw}")
                spot.push_object(dx=deltas[0],dy=deltas[1],d_yaw=d_yaw,dt=3)

            # Check success (existing logic)
            if check_manipulation_success(interactive_perception, initial_hand_pos, current_hand_pos, 
                                        {'distance': config.success_distance, 'angle': config.success_angle}):
                print(f"Door opening completed successfully in {step+1} steps!")
                manipulation_success = True
                break
                
        except Exception as e:
            print(f"Error in step {step+1}: {e}")
            break
    
    print(f"Policy execution completed. Success: {manipulation_success}")
    time.sleep(2)  # Hold position to observe

    return True

def check_manipulation_success(interactive_perception, initial_hand_pos, current_hand_pos, success_threshold):
    """Check if manipulation succeeded (existing logic)."""
    if interactive_perception.joint_type == "prismatic":
        distance = np.linalg.norm(current_hand_pos - initial_hand_pos)
        success = distance >= success_threshold['distance']
        print(f"Prismatic distance: {distance:.3f}m (threshold: {success_threshold['distance']:.3f}m)")
        return success
    
    elif interactive_perception.joint_type == "revolute":
        joint_center = interactive_perception.joint_params['center']
        initial_vec = initial_hand_pos - joint_center
        current_vec = current_hand_pos - joint_center
        cos_angle = np.dot(initial_vec, current_vec) / (np.linalg.norm(initial_vec) * np.linalg.norm(current_vec))
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        angle_deg = np.degrees(angle_rad)
        success = angle_deg >= success_threshold['angle']
        print(f"Revolute angle: {angle_deg:.1f}° (threshold: {success_threshold['angle']:.1f}°)")
        return success
    
    return False


def execute_door_clearance(spot, interactive_perception):
    if interactive_perception.joint_type == "revolute":
        print("Executing door clearance maneuver...")
        radius = interactive_perception.joint_params.get('radius', 0.3)

        # Move backward-left diagonal (away from door swing)
        spot.push_object(dx=-0.4, dy=0.6, dt=4)  # Back 40cm, left 60cm
        
        # # Move backward and LEFT to clear door swing
        # spot.push_object(dx=-radius, dy=radius)  # Both should be same sign for diagonal movement
        # # OR: Move more distinctly left
        # spot.push_object(dx=0.2, dy=radius)      # Forward slightly, left significantly
        
        print("Door clearance completed")
        return True

def door_open_pixel_location(x_pix, y_pix, config):
    print(f"door_open called with pixel coordinates: {x_pix}, {y_pix}, {config}")

    # Initialize robot
    spot = Spot(id="DoorOpener", hostname=config.hostname)
    spot.start()

    # display target pixel and wait for user approval -- for debugging
    result = spot.take_picture(color_src=config.image_source, save_images=True)
    if isinstance(result, tuple):
        color_img = result[0]
    else:
        color_img = result

    should_continue = SpotPerception.display_pixel_selection(color_img, x_pix, y_pix, True)
    if not should_continue:
        return False

    # get spot lease and execute policy
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Initialize
            print('Initializing robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            saved_yaw = spot.save_initial_yaw()

            # Walk to door and grasp handle
            spot.open_gripper()
            target_pixel = (x_pix, y_pix)
            success = spot.grasp_edge(target_pixel, img_src=config.image_source)
            if not success:
                print('Handle not grasped successfully')
                return False

            # Analyze door joint
            interactive_perception = analyze_door_joint(spot, config)
            spot.return_to_saved_yaw(saved_yaw)

            # Execute policy-based door opening
            policy_success = execute_policy_opening(spot, config, interactive_perception)

            # move the robot body aside
            execute_door_clearance(spot, interactive_perception)

            # Cleanup

            if not policy_success:
                print('Door opening completed with issues')
                return False

        except Exception as e:
            print(f'Error occurred: {e}')
            return False

    return True

def door_open(config):
    """Main function to open door with Spot."""
    
    # Initialize robot
    spot = Spot(id="DoorOpener", hostname=config.hostname)
    spot.start()
    
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Initialize
            print('Initializing robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            saved_yaw = spot.save_initial_yaw()
            # Walk to door and grasp handle
            if not user_confirm_step("Walk to door and grasp handle"):
                return cleanup_and_exit(spot, config.dock_id)
            walk_to_door_and_grasp(spot, config)
            
            # Analyze door joint
            if not user_confirm_step("Perform wiggle analysis"):
                return cleanup_and_exit(spot, config.dock_id)
            interactive_perception = analyze_door_joint(spot, config)
            spot.return_to_saved_yaw(saved_yaw)
            
            # Execute policy-based door opening
            if not user_confirm_step("Execute intelligent door opening"):
                return cleanup_and_exit(spot, config.dock_id)
            policy_success = execute_policy_opening(spot, config, interactive_perception)

            # move the robot body aside 
            if not user_confirm_step("Execute scene clearance by moving body aside"):
                return cleanup_and_exit(spot, config.dock_id)
            execute_door_clearance(spot, interactive_perception)
            
            # Cleanup
            if not user_confirm_step("Release handle and dock"):
                return cleanup_and_exit(spot, config.dock_id)
            cleanup_and_exit(spot, config.dock_id)
            
            if policy_success:
                print('Door opening completed successfully!')
            else:
                print('Door opening completed with issues')
            
        except KeyboardInterrupt:
            print('Interrupted by user. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)
        except Exception as e:
            print(f'Error occurred: {e}. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Open door with Spot robot')
    parser.add_argument('--hostname', required=True, help='Spot robot hostname or IP')
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
                        default=0.02,
                        help='Scale factor for policy actions (meters)')
    parser.add_argument('--success-distance',
                        type=float, 
                        default=0.1,
                        help='Success threshold distance for prismatic joints (meters)')
    parser.add_argument('--success-angle',
                        type=float,
                        default=60.0,
                        help='Success threshold angle for revolute joints (degrees)')
    parser.add_argument('--force-joint-type', choices=['prismatic', 'revolute'], 
                       help='Force specific joint type (overrides detection)')
    
    options = parser.parse_args()
    
    try:
        door_open(options)
        return True
    except Exception as exc:
        print(f'Door opening failed: {exc}')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)