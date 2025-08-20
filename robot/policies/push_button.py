"""
Push light switch button with Spot robot

Push a button with Spot robot by:
1. Walking to the button location (user clicks on image)
2. Unstow arm 
3. Apply downward force to press button
4. Return to ready position
5. Stow arm

run command with arguments:
python push_button.py --username <username> --password <password> --hostname <ip-address> --approach-distance 1 --press-force-percentage 0.05

Author: Shivam Goel
Date: July 2025
"""

import argparse
import sys
import time

from bosdyn.api import (arm_surface_contact_pb2, arm_surface_contact_service_pb2, 
                        geometry_pb2, trajectory_pb2)
from bosdyn.client import math_helpers
from bosdyn.client.arm_surface_contact import ArmSurfaceContactClient
from bosdyn.client.frame_helpers import (GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, 
                                        get_a_tform_b)
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration

# Import custom Spot modules
from spot.spot import Spot
from spot.spot_perception import SpotPerception


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
        
        # Try to stow arm if it's extended
        try:
            print('Attempting to stow arm...')
            spot.stow_arm()
            print('Arm stowed successfully')
        except Exception as e:
            print(f'Could not stow arm: {e}')
        
        # Try to dock the robot
        try:
            print('Attempting to dock robot...')
            spot.dock(dock_id=dock_id)
            print('Robot docked successfully')
        except Exception as e:
            print(f'Could not dock robot: {e}')
        
        print('Robot safely shut down.')
        
    except Exception as e:
        print(f'Error during cleanup: {e}')


def walk_to_button_location(spot, config):
    """Walk to the button location based on user selection."""
    print('Walking to button location')
    
    # Take picture using existing method
    print(f'Taking picture with camera: {config.image_source}')

    result = spot.take_picture(color_src=config.image_source, save_images=True)
    if isinstance(result, tuple):
        color_img = result[0]
    else:
        color_img = result
    
    if color_img is None:
        raise Exception('Failed to capture image')
    
    # Get user to click on button location using your existing method
    print('Click on the button you want to push...')
    target_pixel = SpotPerception.get_target_from_user(color_img)
    
    if target_pixel is None:
        raise Exception('No target selected')
    
    print(f'Target selected at pixel: {target_pixel}')
    
    # Walk to button location using the new method
    spot.walk_to_pixel(target_pixel, 
                      img_src=config.image_source, 
                      offset_distance=config.approach_distance, timeout=config.walk_timeout)
    
    print('Walk to button location completed')


def execute_surface_contact_push(spot, config):
    """Use arm surface contact to press down with controlled force."""
    print('Executing arm surface contact sequence')
    
    # Create the clients we need
    robot_state_client = spot._client._spot.ensure_client(RobotStateClient.default_service_name)
    arm_surface_contact_client = spot._client._spot.ensure_client(ArmSurfaceContactClient.default_service_name)
    
    # Get current robot state and current arm position
    robot_state = robot_state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    
    # Get current hand pose (where arm is after gazing)
    current_hand_pose = get_a_tform_b(snapshot, 'body', 'hand')
    if current_hand_pose is None:
        print('Could not get current hand pose')
        return
    
    print(f'Current hand position: x={current_hand_pose.x:.3f}, y={current_hand_pose.y:.3f}, z={current_hand_pose.z:.3f}')
    
    # Keep the same X,Y position
    hand_x = current_hand_pose.x  # Keep current X position
    hand_y = current_hand_pose.y  # Keep current Y position  
    hand_z = current_hand_pose.z  # Start from current Z position
    
    # Create downward-facing orientation (pointing straight down)
    qw = 0.707  # cos(45°) 
    qx = 0      # No rotation around X
    qy = 0.707  # sin(45°) for 90° rotation around Y
    qz = 0      # No rotation around Z
    body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    
    # Get transform to odom frame
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    
    # STEP 1: Orient gripper downward at current position
    print('Orienting gripper downward at current position...')
    
    # Create pose with current position but downward orientation
    current_vec3 = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z)
    downward_pose_body = geometry_pb2.SE3Pose(position=current_vec3, rotation=body_Q_hand)
    
    # Transform to odom frame
    odom_T_downward = odom_T_flat_body * math_helpers.SE3Pose.from_proto(downward_pose_body)
    
    # Move to downward orientation
    orient_command = RobotCommandBuilder.arm_pose_command(
        odom_T_downward.x, odom_T_downward.y, odom_T_downward.z,
        odom_T_downward.rot.w, odom_T_downward.rot.x, odom_T_downward.rot.y, odom_T_downward.rot.z,
        ODOM_FRAME_NAME, seconds=1.5
    )
    
    # Execute orientation
    command_client = spot._client._command_client
    orient_cmd_id = command_client.robot_command(orient_command)
    block_until_arm_arrives(command_client, orient_cmd_id, timeout_sec=2.0)
    print('Downward orientation complete')
    
    # STEP 2: Now do the press (from current position down)
    print('Preparing downward press trajectory...')
    
    # Define start and end positions - press down from current position
    press_distance = 0.08  # Press down 8cm
    hand_vec3_start = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z)
    hand_vec3_end = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z - press_distance)
    
    # Create poses with downward orientation
    body_T_hand_start = geometry_pb2.SE3Pose(position=hand_vec3_start, rotation=body_Q_hand)
    body_T_hand_end = geometry_pb2.SE3Pose(position=hand_vec3_end, rotation=body_Q_hand)
    
    # Transform to odom frame
    odom_T_hand_start = odom_T_flat_body * math_helpers.SE3Pose.from_proto(body_T_hand_start)
    odom_T_hand_end = odom_T_flat_body * math_helpers.SE3Pose.from_proto(body_T_hand_end)
    
    # Create trajectory
    trajectory_time = config.press_duration
    
    traj_point_start = trajectory_pb2.SE3TrajectoryPoint(
        pose=odom_T_hand_start.to_proto(),
        time_since_reference=seconds_to_duration(0)
    )
    traj_point_end = trajectory_pb2.SE3TrajectoryPoint(
        pose=odom_T_hand_end.to_proto(),
        time_since_reference=seconds_to_duration(trajectory_time)
    )
    
    hand_trajectory = trajectory_pb2.SE3Trajectory(points=[traj_point_start, traj_point_end])
    
    # Set up force parameters
    press_force_percentage = config.press_force_percentage
    percentage_press = geometry_pb2.Vec3(x=0, y=0, z=-press_force_percentage)
    
    print(f'Applying {press_force_percentage*100}% downward force for {trajectory_time}s')
    
    # Close gripper slightly
    gripper_cmd_packed = RobotCommandBuilder.claw_gripper_open_fraction_command(0.2)
    gripper_command = gripper_cmd_packed.synchronized_command.gripper_command.claw_gripper_command
    
    # Create arm surface contact command
    cmd = arm_surface_contact_pb2.ArmSurfaceContact.Request(
        pose_trajectory_in_task=hand_trajectory,
        root_frame_name=ODOM_FRAME_NAME,
        press_force_percentage=percentage_press,
        x_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
        y_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
        z_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_FORCE,
        z_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_LOOSE,
        xy_to_z_cross_term_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_VERY_STIFF,
        gripper_command=gripper_command
    )
    
    cmd.is_robot_following_hand = False
    bias_force_x = -10
    cmd.bias_force_ewrt_body.CopyFrom(geometry_pb2.Vec3(x=bias_force_x, y=0, z=0))
    
    # Execute the command
    proto = arm_surface_contact_service_pb2.ArmSurfaceContactCommand(request=cmd)
    print('Executing arm surface contact - pressing down...')
    arm_surface_contact_client.arm_surface_contact_command(proto)
    
    # Wait for completion
    time.sleep(trajectory_time + 1.0)
    
    # Return to ready position
    print('Moving arm back up to ready position...')
    # spot.unstow_arm()
    
    print('Button press completed')

def push_button_pixel_location(x_pix, y_pix, config):
    print(f"push_button_pixel_location called with pixel coordinates: {x_pix}, {y_pix}, {config}")

    # Initialize robot using your existing class
    spot = Spot(id="ButtonPush", username=config.username, password=config.password, hostname=config.hostname)
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
            # Power on and stand up
            print('Powering on robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()

            #  Walk to button location
            target_pixel = (x_pix, y_pix)
            spot.walk_to_pixel(target_pixel,
                               img_src=config.image_source,
                               offset_distance=config.approach_distance, timeout=config.walk_timeout)

            # open gripper
            print(f"opening gripper now")
            spot.open_gripper()
            spot.unstow_arm()

            spot.close_gripper()

            # Execute surface contact push
            execute_surface_contact_push(spot, config)

            print('Button push sequence completed successfully!')

            spot.stow_arm()

        except Exception as e:
            print(f'Error occurred: {e}')
            return False

    return True

def push_button(config):
    """Main function to push a button with Spot."""
    
    # Initialize robot using your existing class
    spot = Spot(id="ButtonPush", username=config.username, password=config.password, hostname=config.hostname)
    spot.start()
    
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Power on and stand up
            print('Powering on robot...')
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            
            #  Walk to button location
            if not user_confirm_step("Walk to button location"):
                return cleanup_and_exit(spot, config.dock_id)
            walk_to_button_location(spot, config)

            # open gripper
            print (f"opening gripper now")
            spot.open_gripper()
            spot.unstow_arm()

            # # Gaze at the clicked pixel
            # if not user_confirm_step("Gaze at point"):
            #     return cleanup_and_exit(spot, config.dock_id)
            # gaze_target = get_button_target(spot, config)
            # spot.gaze_at_pixel(gaze_target, img_src=config.gaze_image_source)

            spot.close_gripper()
            
            # Execute surface contact push
            if not user_confirm_step("Execute button push (down → push → up)"):
                return cleanup_and_exit(spot, config.dock_id)
            execute_surface_contact_push(spot, config)

            print('Button push sequence completed successfully!')
            
            cleanup_and_exit(spot, config.dock_id)

        except KeyboardInterrupt:
            print('Interrupted by user. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)
        except Exception as e:
            print(f'Error occurred: {e}. Cleaning up...')
            return cleanup_and_exit(spot, config.dock_id)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Push a button with Spot robot')
    parser.add_argument("--username", required=True, help="Spot username for authentication")
    parser.add_argument("--password", required=True, help="Spot password for authentication")
    parser.add_argument('--hostname', required=True, help='Spot robot hostname or IP')
    parser.add_argument('-i', '--image-source', 
                        help='Camera source for walk to object detection',
                        default='hand_color_image')
    parser.add_argument('-g', '--gaze-image-source', 
                        help='Camera source for gazing detection',
                        default='hand_color_image')
    parser.add_argument('-a', '--approach-distance', 
                        help='Distance to stop from target (meters)',
                        default=1, type=float)
    parser.add_argument('-f', '--press-force-percentage', 
                        help='Percentage of max downward force (0.01-0.20, be careful!)',
                        default=0.05, type=float)
    parser.add_argument('-t', '--press-duration', 
                        help='Duration to maintain downward force (seconds)',
                        default=1.0, type=float)
    parser.add_argument('--dock-id', 
                    help='Docking station ID',
                    default=521, type=int)
    parser.add_argument('--walk-timeout', 
                    help='time out for walking to a pixel',
                    default=15.0, type=float)
    options = parser.parse_args()
    
    try:
        push_button(options)
        return True
    except Exception as exc:
        print(f'Button push failed: {exc}')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)