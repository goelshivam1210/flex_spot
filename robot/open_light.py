#!/usr/bin/env python3

"""
Push a button with Spot robot by:
1. Walking to the button location (user clicks on image)
2. Extending arm forward toward button
3. Moving arm down to button level
4. Applying forward force to push the button
"""

import argparse
import sys
import time
import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import (arm_surface_contact_pb2, arm_surface_contact_service_pb2, geometry_pb2, 
                        image_pb2, manipulation_api_pb2, trajectory_pb2)
from bosdyn.client import math_helpers
from bosdyn.client.arm_surface_contact import ArmSurfaceContactClient
from bosdyn.client.frame_helpers import (GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, 
                                        VISION_FRAME_NAME, get_a_tform_b)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.util import seconds_to_duration

# Global variables for image clicking
g_image_click = None
g_image_display = None


def push_button(config):
    """Main function to push a button with Spot."""
    
    # Setup logging and robot connection
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ButtonPushClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client.'

    # Create clients
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    arm_surface_contact_client = robot.ensure_client(ArmSurfaceContactClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Power on and stand up
            robot.logger.info('Powering on robot...')
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), 'Robot power on failed.'
            
            robot.logger.info('Standing up...')
            blocking_stand(command_client, timeout_sec=10)
            
            # Step 1: Walk to button location
            if not user_confirm_step("Step 1: Walk to button location"):
                return cleanup_and_exit(robot, command_client)
            walk_to_button_location(robot, image_client, manipulation_api_client, config)
            
            # Step 2: Unstow arm
            if not user_confirm_step("Step 2: Unstow arm to ready position"):
                return cleanup_and_exit(robot, command_client)
            prepare_arm_for_button_push(robot, command_client)
            
            # Step 3: Execute arm movement sequence (face down → move down → come up)
            if not user_confirm_step("Step 3: Execute arm movement sequence (face down → move down → come up)"):
                return cleanup_and_exit(robot, command_client)
            execute_button_push_sequence(robot, command_client, config)
            
            # Step 4: Stow arm
            if not user_confirm_step("Step 4: Stow arm and finish"):
                return cleanup_and_exit(robot, command_client)
            retract_arm_and_sit(robot, command_client)
            
            robot.power_off(cut_immediately=False, timeout_sec=20)
            robot.logger.info('Robot safely powered off.')
            
        except KeyboardInterrupt:
            robot.logger.info('Interrupted by user. Cleaning up...')
            return cleanup_and_exit(robot, command_client)
        except Exception as e:
            robot.logger.error(f'Error occurred: {e}. Cleaning up...')
            return cleanup_and_exit(robot, command_client)


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


def cleanup_and_exit(robot, command_client):
    """Safely cleanup and exit the program."""
    try:
        robot.logger.info('Performing safe shutdown...')
        
        # Try to stow arm if it's extended
        try:
            robot.logger.info('Attempting to stow arm...')
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            stow_command_id = command_client.robot_command(stow_cmd)
            block_until_arm_arrives(command_client, stow_command_id, 5.0)
            robot.logger.info('Arm stowed successfully')
        except Exception as e:
            robot.logger.warning(f'Could not stow arm: {e}')
        
        # Power off safely
        robot.logger.info('Powering off robot...')
        robot.power_off(cut_immediately=False, timeout_sec=20)
        robot.logger.info('Robot safely powered off.')
        
    except Exception as e:
        robot.logger.error(f'Error during cleanup: {e}')
        robot.logger.info('Attempting emergency power off...')
        try:
            robot.power_off(cut_immediately=True, timeout_sec=5)
        except:
            pass


def walk_to_button_location(robot, image_client, manipulation_api_client, config):
    """Step 1: Walk to the button location based on user selection."""
    robot.logger.info('Step 1: Walking to button location')
    
    # Take image for user selection
    robot.logger.info(f'Getting image from: {config.image_source}')
    image_responses = image_client.get_image_from_sources([config.image_source])
    
    if len(image_responses) != 1:
        raise Exception(f'Got invalid number of images: {len(image_responses)}')
    
    image = image_responses[0]
    
    # Convert image to OpenCV format
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    
    img = np.fromstring(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)
    
    # Get user to click on button location
    robot.logger.info('Click on the button you want to push...')
    image_title = 'Click on the button to push'
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)
    
    global g_image_click, g_image_display
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print('"q" pressed during image selection, exiting.')
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User quit during image selection")
    
    cv2.destroyAllWindows()
    
    robot.logger.info(f'Walking to button at pixel ({g_image_click[0]}, {g_image_click[1]})')
    
    # Create walk-to command
    walk_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
    
    # Set offset distance (how close to get to the target)
    offset_distance = wrappers_pb2.FloatValue(value=1.2)  # Default approach distance
    
    # Build walk-to request
    walk_to = manipulation_api_pb2.WalkToObjectInImage(
        pixel_xy=walk_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
        offset_distance=offset_distance
    )
    
    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
        walk_to_object_in_image=walk_to
    )
    
    # Execute walk command
    cmd_response = manipulation_api_client.manipulation_api_command(walk_to_request)
    
    # Wait for completion with timeout and movement detection
    robot.logger.info('Monitoring walk progress...')
    start_time = time.time()
    timeout_duration = 15.0  # 15 second timeout
    last_position = None
    stationary_count = 0
    stationary_threshold = 8  # If stationary for 8 checks (2 seconds), consider done
    
    while True:
        time.sleep(0.25)
        current_time = time.time()
        
        # Check timeout
        if current_time - start_time > timeout_duration:
            robot.logger.warning(f'Walk command timed out after {timeout_duration}s')
            break
        
        # Get feedback
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )
        response = manipulation_api_client.manipulation_api_feedback_command(feedback_request)
        
        print(f'Walk state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')
        
        # Check if officially done
        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            robot.logger.info('Walk command completed successfully')
            break
        
        # Check if robot has stopped moving (alternative completion detection)
        try:
            robot_state_client = robot.ensure_client('robot-state')
            robot_state = robot_state_client.get_robot_state()
            current_position = [
                robot_state.kinematic_state.ko_tform_body.position.x,
                robot_state.kinematic_state.ko_tform_body.position.y
            ]
            
            if last_position is not None:
                # Calculate distance moved since last check
                distance_moved = ((current_position[0] - last_position[0])**2 + 
                                (current_position[1] - last_position[1])**2)**0.5
                
                if distance_moved < 0.02:  # Less than 2cm movement
                    stationary_count += 1
                    if stationary_count >= stationary_threshold:
                        robot.logger.info(f'Robot appears stationary for {stationary_threshold * 0.25}s, assuming arrival')
                        break
                else:
                    stationary_count = 0  # Reset if robot is moving
            
            last_position = current_position
            
        except Exception as e:
            robot.logger.warning(f'Could not check robot position: {e}')
    
    robot.logger.info('Walk to button location completed')


def prepare_arm_for_button_push(robot, command_client):
    """Step 2: Unstow arm to ready position."""
    robot.logger.info('Step 2: Unstowing arm to ready position')
    
    # Unstow the arm to standard ready position
    robot.logger.info('Unstowing arm...')
    unstow_cmd = RobotCommandBuilder.arm_ready_command()
    unstow_command_id = command_client.robot_command(unstow_cmd)
    block_until_arm_arrives(command_client, unstow_command_id, 3.0)
    
    robot.logger.info('Arm unstowed and ready')


def execute_button_push_sequence(robot, command_client, config):
    """Step 3: Go straight down from unstow position to press button."""
    robot.logger.info('Step 3: Moving arm straight down from unstow position')
    
    # Create the clients we need inside the function
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    arm_surface_contact_client = robot.ensure_client(ArmSurfaceContactClient.default_service_name)
    
    # Get current robot state and current arm position
    robot_state = robot_state_client.get_robot_state()
    snapshot = robot_state.kinematic_state.transforms_snapshot
    
    # Get current hand pose (where arm is after unstowing)
    current_hand_pose = get_a_tform_b(snapshot, 'body', 'hand')
    if current_hand_pose is None:
        robot.logger.error('Could not get current hand pose')
        return
    
    robot.logger.info(f'Current hand position: x={current_hand_pose.x:.3f}, y={current_hand_pose.y:.3f}, z={current_hand_pose.z:.3f}')
    
    # Keep the same X,Y position, just go straight down in Z
    hand_x = current_hand_pose.x  # Keep current X position
    hand_y = current_hand_pose.y  # Keep current Y position  
    hand_z = current_hand_pose.z  # Start from current Z position
    
    robot.logger.info(f'Will press button straight down from current position')
    
    # Create downward-facing orientation (pointing straight down)
    # Quaternion for pointing straight down: 90 degree rotation around Y-axis
    qw = 0.707  # cos(45°) 
    qx = 0      # No rotation around X
    qy = 0.707  # sin(45°) for 90° rotation around Y
    qz = 0      # No rotation around Z
    body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    
    # Define start and end positions (same X,Y, maintain current Z)
    hand_vec3_start = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z)
    hand_vec3_end = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z)  # Same position
    
    # Create poses with downward orientation
    body_T_hand_start = geometry_pb2.SE3Pose(position=hand_vec3_start, rotation=body_Q_hand)
    body_T_hand_end = geometry_pb2.SE3Pose(position=hand_vec3_end, rotation=body_Q_hand)
    
    # Transform to odom frame
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    odom_T_hand_start = odom_T_flat_body * math_helpers.SE3Pose.from_proto(body_T_hand_start)
    odom_T_hand_end = odom_T_flat_body * math_helpers.SE3Pose.from_proto(body_T_hand_end)
    
    # Create trajectory (stationary position, force will handle the pressing)
    trajectory_time = config.press_duration  # How long to maintain the press
    
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
    # Negative Z force to press downward
    press_force_percentage = config.press_force_percentage  # Percentage of max force
    percentage_press = geometry_pb2.Vec3(x=0, y=0, z=-press_force_percentage)
    
    robot.logger.info(f'Applying {press_force_percentage*100}% downward force for {trajectory_time}s')
    
    # Close gripper slightly (not fully closed to avoid damage)
    gripper_cmd_packed = RobotCommandBuilder.claw_gripper_open_fraction_command(0.2)  # 20% closed
    gripper_command = gripper_cmd_packed.synchronized_command.gripper_command.claw_gripper_command
    
    # Create arm surface contact command
    cmd = arm_surface_contact_pb2.ArmSurfaceContact.Request(
        pose_trajectory_in_task=hand_trajectory,
        root_frame_name=ODOM_FRAME_NAME,
        press_force_percentage=percentage_press,
        # Position control in X and Y, force control in Z
        x_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
        y_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
        z_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_FORCE,
        # Set admittance (compliance) - loose for safer interaction
        z_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_LOOSE,
        # Cross-term admittance for safety (if arm gets stuck, it will retract)
        xy_to_z_cross_term_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_VERY_STIFF,
        gripper_command=gripper_command
    )
    
    # Keep robot stationary
    cmd.is_robot_following_hand = False
    
    # Optional: Add bias force for stability (small forward lean)
    bias_force_x = -10  # Small forward bias
    cmd.bias_force_ewrt_body.CopyFrom(geometry_pb2.Vec3(x=bias_force_x, y=0, z=0))
    
    # Create the service request
    proto = arm_surface_contact_service_pb2.ArmSurfaceContactCommand(request=cmd)
    
    # Execute the arm surface contact command
    robot.logger.info('Executing arm surface contact - pressing straight down...')
    arm_surface_contact_client.arm_surface_contact_command(proto)
    
    # Wait for the trajectory to complete
    time.sleep(trajectory_time + 2.0)  # Extra time for completion

    # Step 3b: Explicitly move arm back up to unstow/ready position
    robot.logger.info('Moving arm back up to ready position...')

    # Return to standard unstow position
    return_to_ready_cmd = RobotCommandBuilder.arm_ready_command()
    return_command_id = command_client.robot_command(return_to_ready_cmd)
    block_until_arm_arrives(command_client, return_command_id, 3.0)

    robot.logger.info('Arm returned to ready position')
    
    robot.logger.info('Button press completed - arm will return to unstow position')

def retract_arm_and_sit(robot, command_client):
    """Step 4: Stow arm."""
    robot.logger.info('Step 4: Stowing arm...')
    
    # Stow the arm
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow_cmd)
    block_until_arm_arrives(command_client, stow_command_id, 3.0)
    
    robot.logger.info('Arm stowed successfully')


def cv_mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks on the image."""
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
        print(f'Button location selected: ({x}, {y})')
    else:
        # Draw crosshairs for visual feedback
        color = (0, 255, 0)  # Green crosshairs
        thickness = 2
        height, width = clone.shape[:2]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow('Click on the button to push', clone)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Push a button with Spot robot')
    bosdyn.client.util.add_base_arguments(parser)
    
    parser.add_argument('-i', '--image-source', 
                        help='Camera source for button detection',
                        default='frontleft_fisheye_image')
    parser.add_argument('-f', '--press-force-percentage', 
                        help='Percentage of max downward force (0.01-0.20, be careful!)',
                        default=0.05, type=float)
    parser.add_argument('-t', '--press-duration', 
                        help='Duration to maintain downward force (seconds)',
                        default=2.0, type=float)
    
    options = parser.parse_args()
    
    try:
        push_button(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception('Button push failed')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)