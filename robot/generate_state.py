import argparse
import sys
import time
import traceback


import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.client.math_helpers import Quat
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME, get_vision_tform_body, get_a_tform_b, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

# Global variables for image click handling
g_image_click = None
g_image_display = None

def cv_mouse_callback(event, x, y, flags, param):
    """Callback function for mouse events in the OpenCV window."""
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw crosshairs on the image
        color = (30, 30, 30)
        thickness = 2
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow('Click on center of box front panel', clone)

def verify_estop(robot):
    """Verify the robot is not estopped"""
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)

def add_grasp_constraint(config, grasp, robot_state_client):
    """Add grasp constraints to the manipulation API request."""

    # Set the frame name for grasp parameters
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
    
    # Add standoff distance to avoid collisions with protrusions
    grasp.grasp_params.grasp_palm_to_fingertip = 0.18  # Increase this value as needed
    
    # For top-down grasp
    if hasattr(config, 'force_top_down_grasp') and config.force_top_down_grasp:
    # if config.get('force_top_down_grasp', False):
        # Add a constraint that requests that the x-axis of the gripper is pointing in the
        # negative-z direction in the vision frame.
        axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
        axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
        
        # Add the vector constraint to our proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)
        
        # Allow about 10 degrees of tolerance
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    
    # Set the frame name for grasp parameters
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

def arm_object_grasp(config):
    """Use Boston Dynamics API to grasp object and return initial state."""
    
    # Setup logging and SDK
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    # Check for arm
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    
    # Verify the robot is not estopped
    verify_estop(robot)
    
    # Get clients
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    # This will hold our state information
    state_info = {
        'success': False,
        'initial_position': None,
        'joint_axis': None,
        'displacement': np.zeros(2),  # Initialize with zero displacement
        'box_dimensions': {
            'length': 27 * 0.0254,  # Convert inches to meters
            'width': 22 * 0.0254,
            'height': 13 * 0.0254
        }
    }
    
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        try:
            # Power on robot
            robot.logger.info('Powering on robot...')
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), 'Robot power on failed.'
            
            # Stand up
            robot.logger.info('Commanding robot to stand...')
            blocking_stand(command_client, timeout_sec=10)
            robot.logger.info('Robot standing.')
            
            # Get an image for user to click on
            robot.logger.info('Getting an image from: %s', config.image_source)
            image_responses = image_client.get_image_from_sources([config.image_source])
            
            # Process image
            if len(image_responses) != 1:
                robot.logger.error(f'Got invalid number of images: {len(image_responses)}')
                return state_info
                
            image = image_responses[0]
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
            
            # Show image and get user click
            robot.logger.info('Click on the center of the front panel of the box...')
            image_title = 'Click on center of box front panel'
            cv2.namedWindow(image_title)
            cv2.setMouseCallback(image_title, cv_mouse_callback)
            
            global g_image_click, g_image_display
            g_image_display = img.copy()
            g_image_click = None
            cv2.imshow(image_title, g_image_display)
            
            while g_image_click is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    robot.logger.info('"q" pressed, exiting.')
                    return state_info
            
            # Prepare grasp command
            robot.logger.info(f'Grasping at image location ({g_image_click[0]}, {g_image_click[1]})')
            pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
            
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec, 
                transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor,
                camera_model=image.source.pinhole
            )
            
            # Add a top-down grasp constraint
            modified_config = argparse.Namespace()
            modified_config.force_top_down_grasp = True
            add_grasp_constraint(modified_config, grasp, robot_state_client)
            
            # Send grasp command
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
            cmd_response = manipulation_api_client.manipulation_api_command(
                manipulation_api_request=grasp_request)
            
            # Monitor grasp execution
            robot.logger.info('Executing grasp...')
            while True:
                feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_response.manipulation_cmd_id)
                
                response = manipulation_api_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request)
                
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
                robot.logger.info(f'Current state: {state_name}')
                
                if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                    state_info['success'] = True
                    robot.logger.info('Grasp succeeded!')
                    break
                elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                    robot.logger.error('Grasp failed!')
                    break
                    
                time.sleep(0.25)
            
            # If grasp succeeded, extract state information
            if state_info['success']:
                # Get current robot state
                robot_state = robot_state_client.get_robot_state()
                
                try:
                    # Get hand transform in vision frame
                    # Note: The frame name might be different based on your robot configuration
                    # Common names include "hand", "arm_end_effector", or "gripper"
                    hand_frame_names = ["hand", "arm_end_effector", "gripper"]
                    hand_tform = None
                    
                    for frame_name in hand_frame_names:
                        try:
                            hand_tform = get_a_tform_b(
                                robot_state.kinematic_state.transforms_snapshot,
                                VISION_FRAME_NAME,
                                frame_name
                            )
                            if hand_tform:
                                robot.logger.info(f"Found transform for frame: {frame_name}")
                                break
                        except Exception as e:
                            robot.logger.warning(f"Couldn't get transform for {frame_name}: {e}")
                    
                    if hand_tform:
                        robot.logger.info(f"Hand transform type: {type(hand_tform)}")
                        robot.logger.info(f"Rotation type: {type(hand_tform.rotation)}")
                        robot.logger.info(f"Rotation value: {hand_tform.rotation}")

                        # Debug the rotation more deeply
                        rotation = hand_tform.rotation
                        robot.logger.info(f"Rotation dir: {dir(rotation)}")

                        # Extract attributes from rotation to understand its structure
                        for attr in dir(rotation):
                            if not attr.startswith('__'):
                                try:
                                    value = getattr(rotation, attr)
                                    robot.logger.info(f"Rotation.{attr}: {value}")
                                except Exception as e:
                                    robot.logger.info(f"Failed to get rotation.{attr}: {e}")
        

                        # Extract position of the hand (end effector)
                        # The hand matrix contains both position and orientation information
                        hand_position = np.array([
                            hand_tform.position.x,
                            hand_tform.position.y,
                            hand_tform.position.z
                        ])
                        robot.logger.info(f"Hand position: {hand_position}")

                        # Box dimensions
                        box_length = state_info['box_dimensions']['length']
                        robot.logger.info(f"Box length: {box_length} meters")
                        
                        try:
                            # If rotation has x, y, z vectors as objects (SE3Pose format)
                            if hasattr(rotation, 'x') and hasattr(rotation, 'y') and hasattr(rotation, 'z'):
                                hand_x_axis = np.array([
                                    rotation.x.x,
                                    rotation.x.y,
                                    rotation.x.z
                                ])
                                
                                hand_y_axis = np.array([
                                    rotation.y.x,
                                    rotation.y.y,
                                    rotation.y.z
                                ])
                                
                                robot.logger.info(f"Using rotation.x/y vectors format")
                                
                            # If rotation is a quaternion (check if it has w component)
                            elif hasattr(rotation, 'w'):
                                # quat = Quat(w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z)
                                rotation_matrix = rotation.to_matrix()

                                robot.logger.info(f"Rotation matrix from quaternion: {rotation_matrix}")

                                hand_x_axis = np.array([rotation_matrix[0, 0], rotation_matrix[1, 0], rotation_matrix[2, 0]])
                                hand_y_axis = np.array([rotation_matrix[0, 1], rotation_matrix[1, 1], rotation_matrix[2, 1]])
                                
                                robot.logger.info(f"Using quaternion's to_matrix() to create rotation matrix")
                                
                            # If rotation is a 3x3 matrix directly
                            elif hasattr(rotation, 'r00'):
                                # If it has r00, r01, etc. naming convention
                                hand_x_axis = np.array([rotation.r00, rotation.r10, rotation.r20])
                                hand_y_axis = np.array([rotation.r01, rotation.r11, rotation.r21])
                                
                                robot.logger.info(f"Using r00 matrix format")
                                
                            # If it's a float (indicating we need to convert it differently)
                            elif isinstance(rotation, float):
                                # This is a special case - if rotation is just a float, we might need to
                                # use a different approach or default orientation
                                robot.logger.warning(f"Rotation is a float value: {rotation}")
                                # Use default axes
                                hand_x_axis = np.array([1.0, 0.0, 0.0])
                                hand_y_axis = np.array([0.0, 1.0, 0.0])
                                
                            else:
                                # If all else fails, use default values
                                robot.logger.warning("Unknown rotation format, using default orientation")
                                hand_x_axis = np.array([1.0, 0.0, 0.0])
                                hand_y_axis = np.array([0.0, 1.0, 0.0])
                                
                        except Exception as e:
                            robot.logger.error(f"Error processing rotation: {e}")
                            # Use default axes as fallback
                            hand_x_axis = np.array([1.0, 0.0, 0.0])
                            hand_y_axis = np.array([0.0, 1.0, 0.0])
                        
                        robot.logger.info(f"Hand X axis: {hand_x_axis}")
                        robot.logger.info(f"Hand Y axis: {hand_y_axis}")
                        
                        # Calculate the box centroid by moving half the length along the hand's x-axis
                        box_centroid = hand_position + (hand_x_axis * (box_length/2))
                        robot.logger.info(f"Box centroid: {box_centroid}")
                        
                        # Store the initial position (x,y coordinates) for the simulation
                        state_info['initial_position'] = np.array([box_centroid[0], box_centroid[1]])
                        
                        # Set the joint axis based on the hand's orientation
                        # For the simulation, we need a 2D joint axis in the x-y plane
                        # Project the hand's y-axis onto the x-y plane and normalize
                        hand_y_axis_2d = np.array([
                            hand_y_axis[0],
                            hand_y_axis[1],
                            0  # Ignore z component for 2D simulation
                        ])
                        norm = np.linalg.norm(hand_y_axis_2d)
                        if norm > 0:
                            state_info['joint_axis'] = hand_y_axis_2d / norm
                        else:
                            # Fallback to default if projection is too small
                            state_info['joint_axis'] = np.array([1.0, 0.0])
                        
                        robot.logger.info(f"Joint axis (2D normalized): {state_info['joint_axis']}")
                        robot.logger.info(f"Generated state information: {state_info}")
                        
                        # Hold the object for a moment to verify grasp
                        robot.logger.info("Holding object for 3 seconds...")
                        time.sleep(3.0)
                    else:
                        robot.logger.error("Could not find hand transform!")
                except Exception as e:
                    robot.logger.error(f"Error extracting state information: {e}")
                    # Print the full traceback for better debugging
                    robot.logger.error(f"Traceback: {traceback.format_exc()}")
            
        except Exception as e:
            robot.logger.error(f"Error during operation: {e}")
        finally:
            # Power off robot
            robot.logger.info('Sitting down and powering off...')
            try:
                robot.power_off(cut_immediately=False, timeout_sec=20)
                robot.logger.info('Robot powered off.')
            except Exception as e:
                robot.logger.error(f"Error powering off: {e}")
    
    return state_info

def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    options = parser.parse_args()
    
    state_info = arm_object_grasp(options)
    
    if state_info['success']:
        print("\n== Final State Information ==")
        print(f"Initial position: {state_info['initial_position']}")
        print(f"Joint axis: {state_info['joint_axis']}")
        print(f"Box dimensions: {state_info['box_dimensions']}")
        return True
    else:
        print("Failed to generate state information.")
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)