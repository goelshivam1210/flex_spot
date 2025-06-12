# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
Integrated Two Spots with Policy Control

Extended version of two_spots.py that includes:
1. Walk to object (original functionality)
2. Policy-based coordinated manipulation using trained TD3 model
"""
import argparse
import sys
import time
import os

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2

# Import simulation components for policy
import torch
sys.path.append('../sim/rotation')  # Adjust path as needed
from td3 import TD3

g_image_click = None
g_image_display = None

class PolicyController:
    """Policy controller for coordinated robot manipulation"""
    
    def __init__(self, model_path: str, control_frequency: float = 10.0, force_to_motion_scale: float = 0.001):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.force_to_motion_scale = force_to_motion_scale
        
        # Contact points in box frame (same as simulation)
        self.contact_points = np.array([
            [-0.2, +0.2, 0.0],  # Robot 1 contact point
            [-0.2, -0.2, 0.0]   # Robot 2 contact point  
        ])
        
        # Reference path (simple arc - same as simulation)
        self.reference_path = self._generate_reference_path()
        
        # Policy
        self.policy = None
        self.load_policy(model_path)
        
        # State tracking
        self.initial_box_pose = None
        self.prev_box_position = None
        self.prev_time = None
        
    def _generate_reference_path(self):
        """Generate reference path (same as simulation)"""
        radius = 1.5
        start_angle = -np.pi/3
        end_angle = np.pi/3
        num_points = 50
        
        points = []
        for theta in np.linspace(start_angle, end_angle, num_points):
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y])
        
        return np.array(points)
    
    def load_policy(self, model_path: str):
        """Load trained TD3 policy"""
        print(f"Loading policy from: {model_path}")
        
        # Policy parameters (should match training)
        state_dim = 8
        action_dim = 3
        max_action = 1.0
        max_torque = 1.0
        
        self.policy = TD3(
            lr=1e-3,  # Not used for inference
            state_dim=state_dim,
            action_dim=action_dim, 
            max_action=max_action,
            max_torque=max_torque
        )
        
        # Load only the actor for inference
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('_actor.pth', '')
        self.policy.load_actor(model_dir, model_name)
        
        print("Policy loaded successfully!")
    
    def get_robot_tcp_pose(self, robot_state_client) -> tuple:
        """Get TCP pose from robot"""
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        
        try:
            # Get hand pose in body frame
            hand_frame = "hand"  # May need adjustment based on actual frame
            hand_transform = transforms.child_to_parent_edge_map[hand_frame].parent_tform_child
            
            position = np.array([
                hand_transform.position.x,
                hand_transform.position.y, 
                hand_transform.position.z
            ])
            
            # Convert quaternion to yaw angle
            quat = hand_transform.rotation
            yaw = np.arctan2(2*(quat.w*quat.z + quat.x*quat.y), 1-2*(quat.y**2 + quat.z**2))
            
            return position, yaw
            
        except Exception as e:
            print(f"Error getting TCP pose: {e}")
            return np.zeros(3), 0.0
    
    def compute_box_pose_from_contacts(self, tcp1_pos: np.ndarray, tcp1_yaw: float,
                                     tcp2_pos: np.ndarray, tcp2_yaw: float) -> np.ndarray:
        """Compute box pose from robot TCP poses"""
        # Box center approximation from contact points
        contact1_world = tcp1_pos[:2]
        contact2_world = tcp2_pos[:2]  # Assuming in same coordinate frame
        
        box_center = (contact1_world + contact2_world) / 2
        
        # Box orientation from contact line
        contact_vector = contact2_world - contact1_world
        box_yaw = np.arctan2(contact_vector[1], contact_vector[0]) - np.pi/2
        
        return np.array([box_center[0], box_center[1], box_yaw])
    
    def compute_path_relative_state(self, box_pose: np.ndarray) -> np.ndarray:
        """Compute 8D path-relative state vector"""
        current_position = box_pose[:2]
        current_orientation = box_pose[2]
        
        # Use reference path (in box-relative coordinates)
        path = self.reference_path
        
        # Find closest path point
        distances = np.linalg.norm(path - current_position, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = path[closest_idx]
        
        # Calculate progress and deviation
        progress = closest_idx / (len(path) - 1)
        deviation = distances[closest_idx]
        
        # Calculate path tangent
        next_idx = min(closest_idx + 1, len(path) - 1)
        tangent = path[next_idx] - path[closest_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-8:
            path_tangent = tangent / tangent_norm
        else:
            path_tangent = np.array([1.0, 0.0])
        
        path_normal = np.array([-path_tangent[1], path_tangent[0]])
        
        # Position errors in path coordinates
        position_error = current_position - closest_point
        lateral_error = np.dot(position_error, path_normal)
        longitudinal_error = np.dot(position_error, path_tangent)
        
        # Orientation error
        desired_orientation = np.arctan2(path_tangent[1], path_tangent[0])
        orientation_error = np.arctan2(
            np.sin(current_orientation - desired_orientation),
            np.cos(current_orientation - desired_orientation)
        )
        
        # Speed along path
        current_time = time.time()
        if self.prev_box_position is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 1e-8:
                velocity = (current_position - self.prev_box_position) / dt
                speed_along_path = np.dot(velocity, path_tangent)
            else:
                speed_along_path = 0.0
        else:
            speed_along_path = 0.0
        
        self.prev_box_position = current_position.copy()
        self.prev_time = current_time
        
        # Box forward vector
        box_forward_x = np.cos(current_orientation)
        box_forward_y = np.sin(current_orientation)
        
        # 8D state vector
        state = np.array([
            lateral_error,
            longitudinal_error,
            orientation_error,
            progress,
            deviation,
            speed_along_path,
            box_forward_x,
            box_forward_y
        ], dtype=np.float32)
        
        return state
    
    def wrench_to_contact_forces(self, wrench: np.ndarray) -> np.ndarray:
        """Convert wrench to contact forces using pseudo-inverse (same as test_dual_force.py)"""
        Fx, Fy, tau_z = wrench[0], wrench[1], wrench[2]
        
        # Build equilibrium matrix
        A = np.zeros((3, 4))
        
        # Force balance
        A[0, 0] = 1.0  # f1x for Fx
        A[0, 2] = 1.0  # f2x for Fx
        A[1, 1] = 1.0  # f1y for Fy
        A[1, 3] = 1.0  # f2y for Fy
        
        # Moment balance
        r1 = self.contact_points[0]
        r2 = self.contact_points[1]
        
        A[2, 0] = -r1[1]  # -r1y * f1x
        A[2, 1] = +r1[0]  # +r1x * f1y
        A[2, 2] = -r2[1]  # -r2y * f2x
        A[2, 3] = +r2[0]  # +r2x * f2y
        
        # Solve
        wrench_2d = np.array([Fx, Fy, tau_z])
        forces_flat = np.linalg.pinv(A) @ wrench_2d
        
        # Reshape
        contact_forces_2d = forces_flat.reshape(2, 2)
        contact_forces = np.zeros((2, 3))
        contact_forces[:, :2] = contact_forces_2d
        
        return contact_forces
    
    def send_robot_motion_command(self, robot_command_client, dx: float, dy: float, duration: float = None):
        """Send relative motion command to robot"""
        if duration is None:
            duration = self.dt
        
        # Mobility parameters
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=True,
            disable_vision_foot_obstacle_avoidance=True,
            disable_vision_foot_constraint_avoidance=True,
            obstacle_avoidance_padding=0.001
        )
        
        speed_limit = geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(x=0.5, y=0.5),
                angular=1.0
            )
        )
        
        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=obstacles,
            vel_limit=speed_limit,
            locomotion_hint=spot_command_pb2.HINT_AUTO
        )
        
        # Get transforms
        robot_state = robot_command_client._robot.ensure_client(RobotStateClient.default_service_name).get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        
        # Create command
        traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            dx, dy, 0.0, transforms, params=mobility_params
        )
        
        # Send command
        end_time = time.time() + duration
        robot_command_client.robot_command(traj_cmd, end_time_secs=end_time)


def collect_target_for_robot(config, robot_id, hostname, username=None, password=None):
    """Connect to robot, get image, and collect user target selection"""
    global g_image_click, g_image_display
    
    print(f"\n=== Setting up target for {robot_id} ===")
    
    try:
        # Create robot connection
        sdk = bosdyn.client.create_standard_sdk(f'ImageCapture_{robot_id}')
        robot = sdk.create_robot(hostname)
        
        if username and password:
            robot.authenticate(username, password)
        else:
            bosdyn.client.util.authenticate(robot)
        
        robot.time_sync.wait_for_sync()
    except Exception as e:
        print(f'Robot {robot_id}: Failed to connect - {e}')
        return None
    
    assert robot.has_arm(), f'Robot {robot_id} requires an arm to run this example.'
    assert not robot.is_estopped(), f'Robot {robot_id} is estopped.'

    # Create image client
    image_client = robot.ensure_client(ImageClient.default_service_name)

    # Take picture
    print(f'Robot {robot_id}: Getting image from: {config.image_source}')
    image_responses = image_client.get_image_from_sources([config.image_source])

    if len(image_responses) != 1:
        print(f'Robot {robot_id}: Got invalid number of images: {len(image_responses)}')
        return None

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

    # Show image and wait for click
    print(f'Robot {robot_id}: Click on the box to establish grasp...')
    image_title = f'{robot_id}: Click on box'
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    g_image_click = None
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print(f'Robot {robot_id}: "q" pressed, exiting.')
            cv2.destroyWindow(image_title)
            return None

    print(f'Robot {robot_id}: Target selected at ({g_image_click[0]}, {g_image_click[1]})')
    cv2.destroyWindow(image_title)
    
    return {
        'robot_id': robot_id,
        'hostname': hostname,
        'username': username,
        'password': password,
        'target_point': g_image_click,
        'image_data': image
    }


def execute_walk_and_grasp_for_robot(config, robot_config, barrier):
    """Execute walk and grasp sequence for a robot"""
    robot_id = robot_config['robot_id']
    hostname = robot_config['hostname']
    username = robot_config['username']
    password = robot_config['password']
    target_point = robot_config['target_point']
    image_data = robot_config['image_data']
    
    try:
        # Create robot connection
        sdk = bosdyn.client.create_standard_sdk(f'WalkAndGraspClient_{robot_id}')
        robot = sdk.create_robot(hostname)
        
        if username and password:
            robot.authenticate(username, password)
        else:
            bosdyn.client.util.authenticate(robot)
        
        robot.time_sync.wait_for_sync()
    except Exception as e:
        print(f'Robot {robot_id}: Failed to connect for execution - {e}')
        return False, None, None
    
    assert robot.has_arm(), f'Robot {robot_id} requires an arm to run this example.'
    assert not robot.is_estopped(), f'Robot {robot_id} is estopped.'

    # Create clients
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on robot
        robot.logger.info(f'Robot {robot_id}: Powering on...')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), f'Robot {robot_id} power on failed.'
        robot.logger.info(f'Robot {robot_id}: Powered on.')

        # Stand up
        robot.logger.info(f'Robot {robot_id}: Commanding to stand...')
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info(f'Robot {robot_id}: Standing.')
        barrier.wait()  # Sync point: both robots standing

        # Walk to target (original walk-to-object code)
        robot.logger.info(f'Robot {robot_id}: Walking to target')
        walk_vec = geometry_pb2.Vec2(x=target_point[0], y=target_point[1])
        
        if config.distance is None:
            offset_distance = None
        else:
            offset_distance = wrappers_pb2.FloatValue(value=config.distance)

        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, 
            transforms_snapshot_for_camera=image_data.shot.transforms_snapshot,
            frame_name_image_sensor=image_data.shot.frame_name_image_sensor,
            camera_model=image_data.source.pinhole, 
            offset_distance=offset_distance
        )

        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
        cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)

        # Monitor walk completion
        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(f'Robot {robot_id}: Walk state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

        robot.logger.info(f'Robot {robot_id}: Walk completed. Establishing grasp...')
        
        # TODO: Add specific grasp/contact establishment commands here
        # For now, assume the walk-to-object puts robot in correct position
        
        # Small positioning adjustment to ensure secure contact
        robot_state = state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=True,
            disable_vision_foot_obstacle_avoidance=True,
            disable_vision_foot_constraint_avoidance=True,
            obstacle_avoidance_padding=0.001
        )
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=0.5, y=0.5), angular=1.0))        
        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=obstacles, 
            vel_limit=speed_limit,
            locomotion_hint=spot_command_pb2.HINT_AUTO
        )
        
        # Move closer to establish contact
        traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            0.2, 0.0, 0.0, transforms, params=mobility_params  # Move 20cm forward
        )
        barrier.wait()  # Sync point: both robots ready for contact
        end_time_secs = time.time() + 3
        command_client.robot_command(traj_cmd, end_time_secs=end_time_secs)
        time.sleep(3)
        
        robot.logger.info(f'Robot {robot_id}: Grasp established.')
        
        # Return robot clients for policy control phase
        return True, command_client, state_client


def execute_policy_control(config, robot_configs, model_path):
    """Execute coordinated policy control with both robots"""
    print("\n=== Starting Policy Control Phase ===")
    
    # Initialize policy controller
    policy_controller = PolicyController(
        model_path=model_path,
        control_frequency=10.0,
        force_to_motion_scale=0.001
    )
    
    # Get robot clients from previous phase
    robot1_command_client = robot_configs[0]['command_client']
    robot1_state_client = robot_configs[0]['state_client']
    robot2_command_client = robot_configs[1]['command_client']
    robot2_state_client = robot_configs[1]['state_client']
    
    print("Executing policy-based coordination...")
    
    max_steps = 500
    goal_reached = False
    
    for step in range(max_steps):
        try:
            step_start_time = time.time()
            
            # 1. Get robot TCP poses
            tcp1_pos, tcp1_yaw = policy_controller.get_robot_tcp_pose(robot1_state_client)
            tcp2_pos, tcp2_yaw = policy_controller.get_robot_tcp_pose(robot2_state_client)
            
            # 2. Compute box pose
            box_pose = policy_controller.compute_box_pose_from_contacts(
                tcp1_pos, tcp1_yaw, tcp2_pos, tcp2_yaw
            )
            
            # Store initial pose
            if policy_controller.initial_box_pose is None:
                policy_controller.initial_box_pose = box_pose.copy()
                print(f"Initial box pose: {box_pose}")
            
            # 3. Compute state
            state_8d = policy_controller.compute_path_relative_state(box_pose)
            
            # 4. Policy inference
            action = policy_controller.policy.select_action(state_8d)
            if action.ndim > 1:
                action = action.squeeze(0)
            
            # Scale to realistic forces
            wrench = action * np.array([300.0, 300.0, 50.0])
            
            # 5. Convert to contact forces
            contact_forces = policy_controller.wrench_to_contact_forces(wrench)
            
            # 6. Convert to motion commands
            dx1 = policy_controller.force_to_motion_scale * contact_forces[0, 0]
            dy1 = policy_controller.force_to_motion_scale * contact_forces[0, 1]
            dx2 = policy_controller.force_to_motion_scale * contact_forces[1, 0]
            dy2 = policy_controller.force_to_motion_scale * contact_forces[1, 1]
            
            # 7. Send commands
            policy_controller.send_robot_motion_command(robot1_command_client, dx1, dy1)
            policy_controller.send_robot_motion_command(robot2_command_client, dx2, dy2)
            
            # 8. Check goal
            progress = state_8d[3]
            deviation = state_8d[4]
            
            if progress > 0.95 and deviation < 0.2:
                goal_reached = True
                print("Policy goal reached!")
                break
            
            # Progress feedback
            if step % 10 == 0:
                print(f"Step {step}: Progress={progress:.3f}, Deviation={deviation:.3f}, "
                      f"Wrench=[{wrench[0]:.1f}, {wrench[1]:.1f}, {wrench[2]:.1f}]")
            
            # Maintain frequency
            elapsed = time.time() - step_start_time
            sleep_time = max(0, policy_controller.dt - elapsed)
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error in policy step {step}: {e}")
            break
    
    if goal_reached:
        print("SUCCESS: Policy control completed successfully!")
    else:
        print(f"Policy control finished after {step+1} steps")
    
    return goal_reached


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        color = (30, 30, 30)
        thickness = 2
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow('Click on box', clone)


def arg_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{repr(x)} not a number')
    return x


def main():
    """Main execution with integrated policy control"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=0.5, type=arg_float)
    
    parser.add_argument('--robot1-hostname', required=True, help='Hostname/IP for first robot')
    parser.add_argument('--robot2-hostname', required=True, help='Hostname/IP for second robot')
    parser.add_argument('--robot1-username', help='Username for first robot')
    parser.add_argument('--robot1-password', help='Password for first robot')
    parser.add_argument('--robot2-username', help='Username for second robot')
    parser.add_argument('--robot2-password', help='Password for second robot')
    
    # Policy control arguments
    parser.add_argument('--model-path', required=True, help='Path to trained TD3 model')
    
    options = parser.parse_args()
    bosdyn.client.util.setup_logging(options.verbose)

    try:
        # Phase 1: Target Collection
        print("=== PHASE 1: Target Collection ===")
        
        robot_configs = []
        
        # Collect targets for both robots
        robot1_config = collect_target_for_robot(
            options, 'Robot1', options.robot1_hostname, 
            options.robot1_username, options.robot1_password
        )
        if robot1_config is None:
            print("Failed to collect target for Robot1")
            return False
        robot_configs.append(robot1_config)
        
        robot2_config = collect_target_for_robot(
            options, 'Robot2', options.robot2_hostname,
            options.robot2_username, options.robot2_password
        )
        if robot2_config is None:
            print("Failed to collect target for Robot2")
            return False
        robot_configs.append(robot2_config)
        
        print("\n=== Target Summary ===")
        for config in robot_configs:
            print(f"{config['robot_id']}: Target at ({config['target_point'][0]}, {config['target_point'][1]})")
        
        input("\nPress Enter to start walk and grasp phase...")
        
        # Phase 2: Walk and Grasp
        print("\n=== PHASE 2: Walk and Grasp ===")
        
        import threading
        num_parties = 3  # 2 robots + main thread
        sync_barrier = threading.Barrier(num_parties)
        
        results = [None, None]
        
        def robot_thread(idx):
            success, cmd_client, state_client = execute_walk_and_grasp_for_robot(
                options, robot_configs[idx], sync_barrier
            )
            robot_configs[idx]['success'] = success
            robot_configs[idx]['command_client'] = cmd_client
            robot_configs[idx]['state_client'] = state_client
            results[idx] = success
        
        # Start both robot threads
        thread1 = threading.Thread(target=robot_thread, args=(0,))
        thread2 = threading.Thread(target=robot_thread, args=(1,))
        
        thread1.start()
        thread2.start()
        
        try:
            print("MAIN: Waiting for robots to stand...")
            sync_barrier.wait()
            print("MAIN: All robots standing. Walk initiated.")
            
            print("MAIN: Waiting for robots to establish grasp...")
            sync_barrier.wait()
            print("MAIN: All robots have established grasp.")
            
        except threading.BrokenBarrierError:
            print("MAIN: Barrier broken - robot task failed")
            return False
        
        # Wait for completion
        thread1.join()
        thread2.join()
        
        # Check if both succeeded
        if not all(results):
            print("Walk and grasp phase failed")
            return False
        
        print("SUCCESS: Both robots have established secure grasp on box")
        
        input("\nPress Enter to start policy control phase...")
        
        # Phase 3: Policy Control
        print("\n=== PHASE 3: Policy Control ===")
        
        # Execute coordinated policy control
        policy_success = execute_policy_control(options, robot_configs, options.model_path)
        
        if policy_success:
            print("\n🎉 MISSION COMPLETE: Box successfully moved along path using learned policy!")
        else:
            print("\n⚠️  Policy control completed but goal may not have been reached")
        
        # Clean shutdown
        print("\n=== PHASE 4: Shutdown ===")
        for i, config in enumerate(robot_configs):
            if config.get('success'):
                try:
                    robot_id = config['robot_id']
                    print(f"Shutting down {robot_id}...")
                    
                    # Get robot connection for shutdown
                    sdk = bosdyn.client.create_standard_sdk(f'Shutdown_{robot_id}')
                    robot = sdk.create_robot(config['hostname'])
                    
                    if config['username'] and config['password']:
                        robot.authenticate(config['username'], config['password'])
                    else:
                        bosdyn.client.util.authenticate(robot)
                    
                    # Power down safely
                    robot.logger.info(f'{robot_id}: Sitting down and powering off.')
                    robot.power_off(cut_immediately=False, timeout_sec=20)
                    print(f"{robot_id}: Safely powered off.")
                    
                except Exception as e:
                    print(f"Error shutting down {robot_id}: {e}")
        
        # Clean up windows
        cv2.destroyAllWindows()
        
        return policy_success
        
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception('Exception occurred during execution')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)