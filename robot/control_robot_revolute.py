import os
import argparse
import sys
import time
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation
# import matplotlib.pyplot as plt

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.client.estop
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2, estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient
from bosdyn.client.math_helpers import SE3Pose, Quat
import bosdyn.client.math_helpers as math_helpers

# Import TD3 model
from td3 import Actor, TD3
# import DISPROD
from disprod_path_plan import generate_waypoints_dubins, DUBINS_CAR_CONFIG_PATH


# Global variables for image click handling
g_image_click = None
g_image_display = None

# After init_visualization(), these four will be set once and used every update:
viz_background = None     # The static image (waypoints + start point)
viz_scale = None          # pixels per meter
viz_x_offset = None       # translation in pixels so that x_min maps to left margin
viz_y_offset = None       # translation in pixels so that y_min maps to bottom margin
viz_canvas_size = None    # integer, e.g. 500 (px)

# for the ESTOP client
def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)

def world_to_pixel(xy, scale, x_offset, y_offset, canvas_size):
    """Convert a world‐coordinate (x, y) in meters to a pixel (px, py)."""
    x_w, y_w = xy
    # X → right; Y → up. We want (0,0) in world to appear near the bottom-left of the image.
    px = int((x_w * scale) + x_offset)
    # Flip Y so that larger Y is “up” on the screen:
    py = int(canvas_size - ((y_w * scale) + y_offset))
    return (px, py)

def init_visualization(waypoints, padding=0.2, canvas_size=500):
    """
    Create a static background image showing:
      • the mirrored waypoints as a yellow polyline
      • a blue circle at the first waypoint (start)
    and store scale/offset so we can overlay the current robot pose on each step.

    After calling this, the window "DiSProD Viz" will appear at once.
    """
    global viz_background, viz_scale, viz_x_offset, viz_y_offset, viz_canvas_size

    # Compute bounding box of all waypoints, then expand by 'padding' meters:
    xs = waypoints[:, 0]
    ys = waypoints[:, 1]
    x_min, x_max = xs.min() - padding, xs.max() + padding
    y_min, y_max = ys.min() - padding, ys.max() + padding

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Decide pixel‐margin and scale:
    margin = int(0.05 * canvas_size)  # 5% of image width as margin
    usable = canvas_size - 2 * margin
    scale = usable / max_range

    # Compute offsets so that (x_min,y_min) maps to roughly (margin, margin from bottom):
    x_offset = margin - int(x_min * scale)
    y_offset = margin - int(y_min * scale)

    # Create a blank BGR image:
    background = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Draw the Dubins curve (yellow polyline) segment by segment:
    for i in range(len(waypoints) - 1):
        p_w0 = waypoints[i]
        p_w1 = waypoints[i + 1]
        p_px = world_to_pixel(p_w0, scale, x_offset, y_offset, canvas_size)
        p_py = world_to_pixel(p_w1, scale, x_offset, y_offset, canvas_size)
        cv2.line(background, p_px, p_py, (0, 235, 255), thickness=2)  # BGR = (0,235,255) is yellow-ish

    # Draw the start point (first waypoint) as a filled blue circle:
    start_px = world_to_pixel(waypoints[0], scale, x_offset, y_offset, canvas_size)
    cv2.circle(background, start_px, radius=6, color=(255, 0, 0), thickness=-1)  # Blue dot

    # Store globals and show the window once:
    viz_background = background
    viz_scale = scale
    viz_x_offset = x_offset
    viz_y_offset = y_offset
    viz_canvas_size = canvas_size

    cv2.namedWindow('DiSProD Viz', cv2.WINDOW_NORMAL)
    cv2.imshow('DiSProD Viz', viz_background)
    cv2.waitKey(1)  # Just 1 ms to force a draw


def update_visualization(current_xy, yaw=None):
    """
    Overlay the current gripper position (red square) and yaw arrow (green) on top of viz_background.
    Call this once per policy step to refresh the window.
    """
    global viz_background, viz_scale, viz_x_offset, viz_y_offset, viz_canvas_size

    if viz_background is None:
        return  # Nothing to do if visualization was never initialized

    frame = viz_background.copy()

    # Draw a filled red square at current_xy:
    px, py = world_to_pixel(current_xy, viz_scale, viz_x_offset, viz_y_offset, viz_canvas_size)
    cv2.rectangle(frame, (px - 5, py - 5), (px + 5, py + 5), (0, 0, 255), thickness=-1)

    # If yaw is given, draw a small green arrow of length ~15 px:
    if yaw is not None:
        length = 15  # pixels
        dx = int(length * np.cos(yaw))
        dy = int(length * np.sin(yaw))
        tip = (px + dx, py - dy)  # subtract dy because screen Y is inverted
        cv2.arrowedLine(frame, (px, py), tip, (0, 255, 0), thickness=2, tipLength=0.2)

    # Show the updated frame:
    cv2.imshow('DiSProD Viz', frame)
    cv2.waitKey(1)  # 1 ms delay is enough to refresh


# def generate_arc_waypoints_from_gripper(start_pos, radius=0.5, start_angle=-np.pi/4, end_angle=np.pi/4, num_points=20):
#     """
#     Generate waypoints that start from the gripper's current position.
#     """
#     waypoints = []
#     for i in range(num_points):
#         t = i / (num_points - 1)  # 0 to 1
        
#         # Interpolate angle
#         angle = start_angle + t * (end_angle - start_angle)
        
#         # Generate points along arc
#         if i == 0:
#             # First point is exactly at gripper position
#             x = start_pos[0]
#             y = start_pos[1]
#         else:
#             # Arc points
#             arc_progress = t * radius
#             x = start_pos[0] + arc_progress * np.cos(angle)
#             y = start_pos[1] + arc_progress * np.sin(angle)
        
#         waypoints.append(np.array([x, y]))
    
#     return np.array(waypoints)

def find_closest_point_on_path(current_pos, waypoints):
    """
    Find the closest waypoint and index on the path.
    
    Args:
        current_pos: Current position [x, y] or [x, y, z]
        waypoints: Array of waypoints
        
    Returns:
        Tuple of (closest_point, closest_index, distance)
    """
    pos_2d = current_pos[:2]
    distances = np.linalg.norm(waypoints - pos_2d, axis=1)
    closest_idx = np.argmin(distances)
    return waypoints[closest_idx], closest_idx, distances[closest_idx]

def calculate_path_tangent(waypoints, index):
    """
    Calculate the tangent direction at a given point on the path.
    
    Args:
        waypoints: Array of waypoints
        index: Current index on path
        
    Returns:
        Tangent angle in radians
    """
    next_idx = min(index + 1, len(waypoints) - 1)
    if index == next_idx:  # At the end
        prev_idx = max(index - 1, 0)
        tangent = waypoints[index] - waypoints[prev_idx]
    else:
        tangent = waypoints[next_idx] - waypoints[index]
    
    return np.arctan2(tangent[1], tangent[0])

def discretize_orientation_error(orientation_error, bin_size_degrees=10.0):
    """
    Discretize orientation error into bins.
    
    Args:
        orientation_error: Orientation error in radians
        bin_size_degrees: Size of each bin in degrees
        
    Returns:
        Discretized orientation error in radians
    """
    num_bins = int(360 / bin_size_degrees)
    bin_index = int(((orientation_error + np.pi) * 180/np.pi) / bin_size_degrees) % num_bins
    return (bin_index * bin_size_degrees * np.pi/180) - np.pi

def calculate_path_following_state(current_position, current_orientation, waypoints):
    """
    Calculate the state for path-following policy.
    
    Args:
        current_position: Current gripper position [x,y,z]
        current_orientation: Current gripper orientation (quaternion)
        waypoints: Array of path waypoints
        
    Returns:
        State array [position_error_x, position_error_y, orientation_error, progress, deviation]
    """
    # Find closest point on path
    pos_2d = current_position[:2]
    closest_point, closest_idx, deviation = find_closest_point_on_path(pos_2d, waypoints)
    
    # Calculate position error
    position_error = pos_2d - closest_point
    
    # Calculate progress along path
    progress = closest_idx / (len(waypoints) - 1)
    
    # Get current orientation (yaw from quaternion)
    quat = [current_orientation.w, current_orientation.x, current_orientation.y, current_orientation.z]
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = rotation.as_euler('xyz')
    current_yaw = euler[2]
    
    # Calculate desired orientation (tangent at closest point)
    desired_yaw = calculate_path_tangent(waypoints, closest_idx)
    
    # Calculate orientation error
    orientation_error = np.arctan2(
        np.sin(current_yaw - desired_yaw),
        np.cos(current_yaw - desired_yaw)
    )
    
    # Discretize orientation error
    discretized_orientation_error = discretize_orientation_error(orientation_error)
    
    # Return state vector
    state = np.array([
        position_error[0],
        position_error[1],
        discretized_orientation_error,
        progress,
        deviation
    ], dtype=np.float32)
    
    return state

def calculate_state_space(robot_state_client, box_dimensions, handle_position, handle_length):
    """Calculate the state space for policy input"""
    # Get current gripper position and orientation
    current_position, current_orientation = get_gripper_position(robot_state_client)
    
    # Get initial position (could be stored as a class variable in a real implementation)
    # For this example, we'll just use the current position
    initial_position = current_position
    
    # Calculate state
    state = calculate_state(current_position, current_orientation, initial_position)
    
    return state

def get_gripper_position(robot_state_client):
    """
    Get the current position of the gripper.
    
    Args:
        robot_state_client: Robot state client
        
    Returns:
        Tuple of (position, orientation) where position is a numpy array [x,y,z]
        and orientation is a quaternion
    """
    # Get robot state
    robot_state = robot_state_client.get_robot_state()
    transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
    
    # Try to get hand transform
    try:
        vision_tform_hand = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, "hand")
        if vision_tform_hand:
            position = np.array([vision_tform_hand.x, vision_tform_hand.y, vision_tform_hand.z])
            orientation = vision_tform_hand.rot
            return position, orientation
    except Exception as e:
        print(f"Could not get hand transform: {e}")
        
    # Try wrist transform as fallback
    try:
        vision_tform_wrist = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, "arm_link_wr1")
        if vision_tform_wrist:
            print("Using wrist link as proxy for hand")
            position = np.array([vision_tform_wrist.x, vision_tform_wrist.y, vision_tform_wrist.z])
            orientation = vision_tform_wrist.rot
            return position, orientation
    except Exception as e:
        print(f"Could not get wrist transform: {e}")
    
    # Use body as final fallback
    vision_tform_body = get_vision_tform_body(transforms_snapshot)
    position = np.array([vision_tform_body.x, vision_tform_body.y, vision_tform_body.z + 0.5])  # Add offset for approximate arm height
    orientation = vision_tform_body.rot
    print("WARNING: Using body position as fallback for gripper position")
    
    return position, orientation

def calculate_state(current_position, current_orientation, initial_position):
    """
    Calculate the state for the policy.
    
    Args:
        current_position: Current gripper position [x,y,z]
        current_orientation: Current gripper orientation (quaternion)
        initial_position: Initial gripper position [x,y,z]
        
    Returns:
        State array [joint_axis_x, joint_axis_y, displacement_x, displacement_y]
    """
    # Calculate displacement (difference between current and initial position)
    displacement = current_position - initial_position
    displacement_2d = displacement[:2]  # Just x,y components
    
    # Calculate joint axis from orientation
    # Use the x-axis of the gripper as the joint axis
    forward_x = 1.0
    forward_y = 0.0
    forward_z = 0.0
    
    # Transform this vector using the quaternion
    try:
        # Convert bosdyn quaternion to scipy rotation
        quat = [current_orientation.w, current_orientation.x, current_orientation.y, current_orientation.z]
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Note scipy uses x,y,z,w order
        
        # Rotate the forward vector
        forward_vector = rotation.apply([forward_x, forward_y, forward_z])
        joint_axis = forward_vector[:2]  # Just x,y components
        
        # Normalize
        norm = np.linalg.norm(joint_axis)
        if norm > 0:
            joint_axis = joint_axis / norm
        else:
            joint_axis = np.array([1.0, 0.0])  # Default if calculation fails
    except Exception as e:
        print(f"Error calculating joint axis: {e}")
        joint_axis = np.array([1.0, 0.0])  # Default in case of error
    
    # Combine into state
    state = np.concatenate([joint_axis, displacement_2d])
    return state.astype(np.float32)

def cv_mouse_callback(event, x, y, flags, param):
    """Callback for mouse events in the OpenCV window"""
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def add_grasp_constraint(config, grasp, robot_state_client):
    """Add constraints to the grasp command"""
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.
        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

def load_policy(policy_dir, policy_name, state_dim=5, action_dim=3):
    """Load the TD3 policy model."""
    # Create a device for model execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create a policy with 4 state dimensions and 3 action dimensions
    model = TD3(0.001, state_dim, action_dim, 1.0)
    
    try:
        # Load just the actor part
        model.load_actor(policy_dir, policy_name)
        print(f"Successfully loaded policy from {policy_dir}/{policy_name}_actor.pth")
        
        # Wrap the select_action method to ensure it returns a standard numpy array
        original_select_action = model.select_action
        
        def wrapped_select_action(state):
            action = original_select_action(state)
            # Convert to standard numpy array of Python floats
            if isinstance(action, torch.Tensor):
                action = action.cpu().detach().numpy()
            return action.astype(float).flatten()
        
        # Replace the select_action method
        model.select_action = wrapped_select_action
        
        return model
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None

def force_action_to_command(action, gripper_position, gripper_orientation,
                             robot_state_client, movement_scale=0.15, use_whole_body=True,
                               apply_orientation=False, desired_yaw=None):
    """
    Convert a force-based action to a robot command with better whole-body coordination.
    
    Args:
        action: Policy action [force_dir_x, force_dir_y, force_scale]
        gripper_position: Current gripper position [x,y,z]
        gripper_orientation: Current gripper orientation (quaternion)
        robot_state_client: Robot state client to get current position
        movement_scale: Scale factor for movement (meters)
        use_whole_body: If True, uses whole-body synchro command, otherwise just arm command
        apply_orientation: If True, adjust gripper orientation to match path
        desired_yaw: Desired yaw angle for orientation control
    Returns:
        Robot command
    """
    # Parse the action
    force_dir = np.array(action[:2])
    force_scale = action[2]
    
    # Normalize force direction if not zero
    norm = np.linalg.norm(force_dir)
    if norm > 0:
        force_dir = force_dir / norm
    else:
        force_dir = np.array([1.0, 0.0])  # Default if zero
    
    # Calculate movement vector - INVERTED for pulling behavior
    delta_x = force_dir[0] * force_scale * movement_scale
    delta_y = force_dir[1] * force_scale * movement_scale

    # print(f"  DEBUG: Force dir: [{force_dir[0]:.3f}, {force_dir[1]:.3f}], Delta: [{delta_x:.3f}, {delta_y:.3f}]")
    
    # Calculate target position for arm
    target_arm_x = gripper_position[0] + delta_x
    target_arm_y = gripper_position[1] + delta_y
    target_arm_z = gripper_position[2]  # Keep same height

    if apply_orientation and desired_yaw is not None:
        # convert desired yaw to quaternion
        target_quaternion = Rotation.from_euler('z', desired_yaw).as_quat()
        qw = target_quaternion[3]
        qx = target_quaternion[0]
        qy = target_quaternion[1]
        qz = target_quaternion[2]
    else:
        # Extract quaternion components
        qw = gripper_orientation.w
        qx = gripper_orientation.x
        qy = gripper_orientation.y
        qz = gripper_orientation.z
    
    if use_whole_body:
        # Create arm pose command
        arm_command = RobotCommandBuilder.arm_pose_command(
            target_arm_x, target_arm_y, target_arm_z,
            qw, qx, qy, qz, VISION_FRAME_NAME, 0.5  # 0.5 seconds duration
        )
        
        # Let's try a simpler approach: create a trajectory point command for the body
        # Get current robot body position in vision frame
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        
        # Calculate a target position for the body that's closer to the arm
        # but doesn't try to match it exactly
        target_body_x = vision_tform_body.x + (delta_x * 0.6)  # Body moves at 60% of arm movement
        target_body_y = vision_tform_body.y + (delta_y * 0.6)
        
        # Create the body trajectory command
        body_command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=target_body_x,
            goal_y=target_body_y,
            goal_heading=vision_tform_body.rot.to_yaw(),  # Maintain current heading
            frame_name=VISION_FRAME_NAME
        )
        
        # Combine into a synchronized command
        command = RobotCommandBuilder.build_synchro_command(body_command, arm_command)
    else:
        # Use only the arm (original behavior)
        command = RobotCommandBuilder.arm_pose_command(
            target_arm_x, target_arm_y, target_arm_z,
            qw, qx, qy, qz, VISION_FRAME_NAME, 0.5
        )
    
    return command

def arm_object_grasp(config, robot, robot_state_client, image_client, command_client, manipulation_api_client):
    """
    Function to grasp an object with the robot arm.
    
    Returns:
        bool: True if grasp was successful, False otherwise
    """
    global g_image_click, g_image_display

    # Take a picture with a camera
    robot.logger.info('Getting an image from: %s', config.image_source)
    image_responses = image_client.get_image_from_sources([config.image_source])

    if len(image_responses) != 1:
        print(f'Got invalid number of images: {len(image_responses)}')
        print(image_responses)
        return False

    image = image_responses[0]
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    img = np.fromstring(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Show the image to the user and wait for them to click on a pixel
    robot.logger.info('Click on an object to start grasping...')
    image_title = 'Click to grasp'
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    g_image_click = None
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # Quit
            print('"q" pressed, exiting.')
            return False

    robot.logger.info(
        f'Picking object at image location ({g_image_click[0]}, {g_image_click[1]})')
    robot.logger.info('Picking object at image location (%s, %s)', g_image_click[0],
                      g_image_click[1])

    pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole)

    # Optionally add a grasp constraint
    add_grasp_constraint(config, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot
    grasp_successful = False
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print(
            f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
        )

        # For policy integration, track current state
        current_state = calculate_state_space(robot_state_client, [20.0, 20.0, 20.0], 9.0, 5.2)
        print(f'State space: joint_axis=[{current_state[0]:.3f}, {current_state[1]:.3f}], '
              f'displacement=[{current_state[2]:.3f}, {current_state[3]:.3f}]')

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            robot.logger.info('Grasp succeeded.')
            robot.logger.info('Robot is ready for policy execution')
            grasp_successful = True
            break
        
        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            robot.logger.info('Grasp failed.')
            break

        time.sleep(0.25)

    robot.logger.info('Finished grasp operation.')
    # cv2.destroyWindow(image_title)
    return grasp_successful

def execute_path_following_policy(robot, policy, robot_state_client, command_client, 
                                 waypoints=None, max_steps=200, use_whole_body=True, 
                                 movement_scale=0.05, apply_orientation=True):
    """
    Execute the path-following policy.
    
    Args:
        robot: Robot instance
        policy: Loaded TD3 policy
        robot_state_client: Robot state client
        command_client: Robot command client
        waypoints: Array of waypoints defining the path (if None, generates arc)
        max_steps: Maximum policy steps to execute
        use_whole_body: If True, use whole-body synchro commands
        movement_scale: Scale factor for movement (meters)
        apply_orientation: If True, align gripper with path tangent
    """
    # Get initial gripper position/orientation
    initial_position, initial_orientation = get_gripper_position(robot_state_client)
    print(f"Initial gripper position: {initial_position}")
    
    # Generate waypoints if not provided
    # if waypoints is None:
    #     print("Generating arc waypoints...")
    #     waypoints = generate_arc_waypoints_from_gripper(initial_position[:2], radius=1.5, start_angle=0, end_angle=np.pi, num_points=20)
    # print(f"Following path with {len(waypoints)} waypoints")

    waypoints = []
    # using disprod to generate waypoints
    if os.path.exists(DUBINS_CAR_CONFIG_PATH):
        waypoint_counter = 0
        try:
            goal_x = initial_position[0] + 2.0
            goal_y = initial_position[1] + 0
            for waypoint in generate_waypoints_dubins(
                start_x = initial_position[0],
                start_y = initial_position[1],
                goal_x = goal_x,
                goal_y = goal_y,
                n_episodes=1,
                depth=30, 
                alg='disprod',
                map_name='no-ob-1'
            ):
                waypoint_counter += 1
                waypoints.append(waypoint[:2])  # Use only x,y for path following
                # print(f"Generated waypoint {waypoint_counter}: {waypoint}")
            # print(f"Generated {waypoint_counter} waypoints using disprod.")
        except Exception as e:
            print(f"Error generating waypoints with disprod: {e}")
    # convert the waypoints to a numpy array
    waypoints = np.array(waypoints, dtype=np.float32)

    # Mirror the Dubins-generated X coordinates about the starting X (X0)
    # this makes sure that "forward is negative X" on SPOT
    X0= initial_position[0]
    waypoints[:, 0] = 2*X0 - waypoints[:, 0]    

    # visulaization of the waypoints
    init_visualization(waypoints, padding=0.2, canvas_size=500)
    print (f"Waypoints after mirroring: {waypoints}")

    # Run policy execution loop
    step_delay = 0.5  # seconds
    goal_threshold = 0.2  # 20cm threshold for completion
    progress_threshold = 0.95  # 95% progress to consider done
    
    # print("Waypoints (first two):", waypoints[0], waypoints[1])
    print("Starting path-following policy execution...")

    for step in range(max_steps):
        # Get current gripper position and orientation
        current_position, current_orientation = get_gripper_position(robot_state_client)
        
        # recompute yaw from quaternion for visualization
        quat = [current_orientation.w, current_orientation.x,
                current_orientation.y, current_orientation.z]
        yaw = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')[2]

        # b) Update the visualization with the new gripper position + yaw
        update_visualization(current_position[:2], yaw)
        cv2.waitKey(1)  # to refresh DiSProD Viz
        
        # Calculate state for policy
        state = calculate_path_following_state(current_position, current_orientation, waypoints)
      
        # Get action from policy
        action = policy.select_action(state)

        # Extract state components for logging
        position_error = state[:2]
        orientation_error = state[2]
        progress = state[3]
        deviation = state[4]
        # Log state and action
        print(f"Step {step}:")
        print(f"  Position: {current_position[:2]}")
        print(f"  Position Error: [{position_error[0]:.3f}, {position_error[1]:.3f}]")
        print(f"  Orientation Error: {orientation_error:.3f} rad ({orientation_error*180/np.pi:.1f} deg)")
        print(f"  Progress: {progress:.3f}, Deviation: {deviation:.3f}")
        print(f"  Action: dir=[{action[0]:.3f}, {action[1]:.3f}], scale={action[2]:.3f}")
        
        # Get desired orientation for current path position
        _, closest_idx, _ = find_closest_point_on_path(current_position[:2], waypoints)
        # Calculate desired yaw based on path tangent
        desired_yaw = calculate_path_tangent(waypoints, closest_idx) if apply_orientation else None
        
        # Convert action to robot command
        cmd = force_action_to_command(
            action, 
            current_position, 
            current_orientation,
            robot_state_client,
            movement_scale=movement_scale,
            use_whole_body=use_whole_body,
            apply_orientation=apply_orientation,
            desired_yaw=desired_yaw
        )
        
        # Execute command
        end_time_secs = time.time() + 0.5
        command_client.robot_command(cmd, end_time_secs=end_time_secs)
        
        # Check if goal reached
        progress = state[3]
        deviation = state[4]
        if progress > progress_threshold and deviation < goal_threshold:
            print(f"Path following completed! Progress={progress:.3f}, Deviation={deviation:.3f}")
            break
        
        # Wait before next step
        time.sleep(step_delay)
    
    print("Path-following policy execution completed.")
    # cv2.destroyAllWindows('DiSProD Viz')  # Close visualization window

def run_integrated_demo(config):
    """
    Run the integrated demo that combines arm grasp and policy execution.
    """
    # Set up logging
    bosdyn.client.util.setup_logging(config.verbose)

    # Create robot connection
    sdk = bosdyn.client.create_standard_sdk('ArmGraspPolicyClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    # Verify the robot has an arm
    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped
    verify_estop(robot)

    # Create clients
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # Load the policy
    if config.run_policy:
        policy = load_policy(config.policy_dir, config.policy_name)
        if policy is None and config.run_policy:
            robot.logger.error("Failed to load policy model. Continuing with grasp only.")
            config.run_policy = False

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on the robot
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Stand up
        robot.logger.info('Commanding robot to stand...')
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Step 1: Execute grasp
        if config.run_grasp:
            grasp_success = arm_object_grasp(
                config, robot, robot_state_client, image_client, command_client, manipulation_api_client
            )
            # cv2.destroyAllWindows('Click to grasp')  # Close OpenCV window after grasp
            # If grasp was successful and policy execution is enabled, run the policy
            if grasp_success and config.run_policy:
                robot.logger.info('Proceeding to policy execution...')
                # Execute policy
                execute_path_following_policy(
                    robot, 
                    policy, 
                    robot_state_client, 
                    command_client,
                    waypoints=config.waypoints,  # You can pass custom waypoints if needed
                    max_steps=config.max_steps,
                    use_whole_body=config.use_whole_body,
                    movement_scale=config.movement_scale,
                    apply_orientation=config.force_top_down_grasp or config.force_horizontal_grasp
                )
            else:
                if not grasp_success:
                    robot.logger.info('Grasp was not successful. Skipping policy execution.')
                            
                # If only policy execution is requested (no grasp)
                elif config.run_policy:
                    robot.logger.info('Running policy execution (without grasp)...')
                    execute_path_following_policy(
                        robot, 
                        policy, 
                        robot_state_client, 
                        command_client,
                        waypoints=config.waypoints,  # You can pass custom waypoints if needed
                        max_steps=config.max_steps,
                        use_whole_body=config.use_whole_body,
                        movement_scale=config.movement_scale,
                        apply_orientation=config.force_top_down_grasp or config.force_horizontal_grasp
                    )

                    # Power off the robot if requested
                    if config.power_off:
                        robot.logger.info('Sitting down and turning off.')
                        robot.power_off(cut_immediately=False, timeout_sec=20)
                        assert not robot.is_powered_on(), 'Robot power off failed.'
                        robot.logger.info('Robot safely powered off.')
                    else:
                        robot.logger.info('Demo completed. Robot remains powered on.')

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    
    # Grasp-related arguments
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    
    # Policy-related arguments
    parser.add_argument('--policy-dir', help='Directory containing policy models', 
                        default='/home/shivam/spot/spot_flex_novelty/code/robot/models/revolute_model')
    parser.add_argument('--policy-name', help='Name of the policy model', default='best_model')
    parser.add_argument('--goal-distance', type=float, help='Distance to goal in meters', default=2.0)
    parser.add_argument('--max-steps', type=int, help='Maximum policy steps', default=100)
    parser.add_argument('--movement-scale', type=float, help='Scale factor for movement (meters)', default=0.05)
    parser.add_argument('--use-whole-body', action='store_true', 
                        help='Use whole-body control for policy execution', default=True)
    parser.add_argument('--arm-only', dest='use_whole_body', action='store_false',
                        help='Use arm-only control for policy execution')
    
    # Control flow arguments
    parser.add_argument('--run-grasp', action='store_true', help='Run the grasp portion of the demo', default=True)
    parser.add_argument('--run-policy', action='store_true', help='Run the policy portion of the demo', default=True)
    parser.add_argument('--no-grasp', dest='run_grasp', action='store_false', help='Skip the grasp portion')
    parser.add_argument('--no-policy', dest='run_policy', action='store_false', help='Skip the policy portion')
    parser.add_argument('--power-off', action='store_true', help='Power off the robot after completion', default=True)
    parser.add_argument('--no-power-off', dest='power_off', action='store_false', help='Leave the robot powered on after completion')
    
    # waypoints arguments
    parser.add_argument('--waypoints-file', help='Path to waypoints file (numpy array)')
    parser.add_argument('--arc-radius', type=float, help='Radius for arc path generation', default=1.5)
    parser.add_argument('--num-waypoints', type=int, help='Number of waypoints to generate', default=20)
    parser.add_argument('--apply-orientation', action='store_true', 
                        help='Align gripper with path tangent', default=True)
    parser.add_argument('--no-orientation', dest='apply_orientation', action='store_false',
                        help='Keep gripper orientation fixed')

    options = parser.parse_args()

    # After parsing, add waypoint handling:
    if options.waypoints_file:
        options.waypoints = np.load(options.waypoints_file)
    else:
        options.waypoints = None  # Will generate arc in execute function

    # Validate grasp options
    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print('Error: cannot force more than one type of grasp. Choose only one.')
        sys.exit(1)
        
    # Ensure at least one task is selected
    if not options.run_grasp and not options.run_policy:
        print('Error: must select at least one task (grasp or policy execution)')
        sys.exit(1)

    try:
        run_integrated_demo(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)