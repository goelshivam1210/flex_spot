# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to walk the robot to an object, usually in preparation for manipulation.
"""
import argparse
import sys
import time
import math

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
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.frame_helpers import get_se2_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, VISION_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME, HAND_FRAME_NAME, get_vision_tform_body, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus


from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2


g_image_click = None
g_image_display = None

def generate_base_arc_deltas(radius: float,
                             start_angle: float,
                             end_angle: float,
                             num_points: int):
    """
    Returns a list of (dx, dy, dyaw) tuples, each representing a small
    body-frame move along a circular arc of given radius.

    - radius: circle radius in meters
    - start_angle, end_angle: in radians, measured from the robot’s forward axis
    - num_points: total samples along that arc (>=2)
    """
    # Compute absolute (x, y, yaw) in body frame for each sample:
    pts = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        # we flip sin and cos beacuse of the weird transofrom and origin
        x = radius * math.sin(angle)      # body-frame x
        y = radius * math.cos(angle)      # body-frame y
        yaw = angle                       # desired heading relative to start
        pts.append((x, y, yaw))

    # Turn those into relative deltas between successive pts:
    deltas = []
    for i in range(1, len(pts)):
        x_prev, y_prev, yaw_prev = pts[i - 1]
        x_cur,  y_cur,  yaw_cur  = pts[i]
        dx = x_cur  - x_prev
        dy = y_cur  - y_prev
        dyaw = yaw_cur - yaw_prev
        deltas.append((dx, dy, dyaw))
    return deltas

def collect_target_for_robot(config, robot_id, hostname, username=None, password=None):
    """Connect to robot, get image, and collect user target selection"""
    # global robot_image_clicks, robot_image_displays
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

    # Create image client only (we don't need full robot operation yet)
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
    print(f'Robot {robot_id}: Click on an object to walk up to...')
    image_title = 'Click to walk up to something'
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
    print(f'Robot {robot_id}: Press any key to continue to next robot...')
    
    # Return the target data
    return {
        'robot_id': robot_id,
        'hostname': hostname,
        'username': username,
        'password': password,
        'target_point': g_image_click,
        'image_data': image
    }

def execute_walk_for_robot(config, robot_config, barrier, velocity, distance, duration, radius):
    """Execute the walk command for a robot with preset target"""
    robot_id = robot_config['robot_id']
    hostname = robot_config['hostname']
    username = robot_config['username']
    password = robot_config['password']
    target_point = robot_config['target_point']
    image_data = robot_config['image_data']
    
    try:
        # Create robot connection
        sdk = bosdyn.client.create_standard_sdk(f'WalkToObjectClient_{robot_id}')
        robot = sdk.create_robot(hostname)
        
        if username and password:
            robot.authenticate(username, password)
        else:
            bosdyn.client.util.authenticate(robot)
        
        robot.time_sync.wait_for_sync()
    except Exception as e:
        print(f'Robot {robot_id}: Failed to connect for execution - {e}')
        return False
    
    assert robot.has_arm(), f'Robot {robot_id} requires an arm to run this example.'
    assert not robot.is_estopped(), f'Robot {robot_id} is estopped.'

    # Create clients
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on robot
        robot.logger.info(f'Robot {robot_id}: Powering on...')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), f'Robot {robot_id} power on failed.'
        robot.logger.info(f'Robot {robot_id}: Powered on.')

        # Stand up
        robot.logger.info(f'Robot {robot_id}: Commanding to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info(f'Robot {robot_id}: Standing.')
        barrier.wait()

        # Use the preset target point
        robot.logger.info(f'Robot {robot_id}: Walking to preset target at ({target_point[0]}, {target_point[1]})')

        walk_vec = geometry_pb2.Vec2(x=target_point[0], y=target_point[1])

        # Build offset distance
        if config.distance is None:
            offset_distance = None
        else:
            offset_distance = wrappers_pb2.FloatValue(value=config.distance)

        # Build the proto using the stored image data
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, 
            transforms_snapshot_for_camera=image_data.shot.transforms_snapshot,
            frame_name_image_sensor=image_data.shot.frame_name_image_sensor,
            camera_model=image_data.source.pinhole, 
            offset_distance=offset_distance
        )

        # Send walk command
        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
        cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)

        # Monitor feedback
        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(f'Robot {robot_id}: Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

        robot.logger.info(f'Robot {robot_id}: Finished.')
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                    disable_vision_foot_obstacle_avoidance=True,
                                                    disable_vision_foot_constraint_avoidance=True,
                                                    obstacle_avoidance_padding=.001)

        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=0.5, y=0.5), angular=1))        
        mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        
        # for making sure the robots touch the surface
        # traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #     0.6,
        #     0.0,
        #     0.0,
        #     transforms,
        #     params = mobility_params
        # )

        # A hand pose front of and to the side of the robot.
        flat_body_T_hand = get_hand_pose_for_example()

        # Request the end effector to move to a pose expressed in ODOM.  After arriving at the pose request
        # Spot to walk forward. Since the arm command is expressed in ODOM, the hand will remain fixed in space
        # as the robot walks forwards.
        odom_T_flat_body = get_a_tform_b(
            robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME, BODY_FRAME_NAME)
        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)
        arm_command = RobotCommandBuilder.arm_pose_command_from_pose(odom_T_hand.to_proto(),
                                                                     ODOM_FRAME_NAME, seconds=2)

        # Construct a RobotCommand from the arm command
        command = RobotCommandBuilder.build_synchro_command(arm_command)

        barrier.wait()
        # Send the request to the robot.
        cmd_id = command_client.robot_command(command)
        robot.logger.info(
            'Moving end-effector to a location in front of and to the side of the robot.')

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)

        # end_time_secs = time.time() + 5  # Half second for command execution
        # command_client.robot_command(traj_cmd, end_time_secs=end_time_secs)
        robot.logger.info(f'Robot {robot_id}: Finished 0.5 m bump')

        start_angle = 0.0          # start facing “forward”
        end_angle   = math.pi/2  # sweep 90° to the left
        num_points  = 20
        # deltas = generate_base_arc_deltas(radius, start_angle, end_angle, num_points)

        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=velocity[0], y=velocity[1]), angular=velocity[2]))        
        mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        command_arm = RobotCommandBuilder.arm_joint_freeze_command()

        # robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        barrier.wait()
        # for dx, dy, dyaw in deltas:
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        #     dy = -dy
        #     print (f'x =  {dx}, y = {dy}, dyaw = {dyaw}')
        #     dt = duration / (num_points-1)

            # relative_move(
            #     dx=dx, dy=dy, dyaw=dyaw, dt=dt,
            #     frame_name=ODOM_FRAME_NAME, 
            #     robot_command_client=robot_command_client,
            #     robot_state_client=robot_state_client,
            #     mobility_params=mobility_params   
            # )

        num_steps = 5
        dx = distance[0]/num_steps
        dy = 0
        dyaw = distance[2]/num_steps
        # dt = duration / (num_steps-1)
        dt = (distance[0]/velocity[0]) / (num_steps-1)
        for i in range(num_steps):
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=velocity[0], y=velocity[1]), angular=velocity[2]))        
            mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            command_arm = RobotCommandBuilder.arm_joint_freeze_command()
            robot_state = robot_state_client.get_robot_state()
            transforms = robot_state.kinematic_state.transforms_snapshot
            traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                        dx,
                        dy,
                        0,
                        transforms,
                        params = mobility_params,
                        build_on_command=command_arm)
            end_t = time.time() + dt
            cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_t)
            # block_until_arm_arrives(command_client, cmd_id, timeout_sec=5.0) # Or use a mobility feedback loop
            print(f'step = {i} dx = {dx}, dyaw = {dyaw}')
            # time.sleep()
            time.sleep(dt+1)
            barrier.wait()

            command_arm = RobotCommandBuilder.arm_joint_freeze_command()
            robot_state = robot_state_client.get_robot_state()
            transforms = robot_state.kinematic_state.transforms_snapshot
            traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                        0,
                        dy,
                        dyaw,
                        transforms,
                        params = mobility_params,
                        build_on_command=command_arm)
            end_t = time.time() + dt
            cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_t)
            # block_until_arm_arrives(command_client, cmd_id, timeout_sec=5.0) # Or use a mobility feedback loop
            print(f'step = {i} dx = {dx}, dyaw = {dyaw}')
            # time.sleep()
            time.sleep(dt+1)
            barrier.wait()

        
        # traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #     distance[0],
        #     distance[1],
        #     distance[2],
        #     transforms,
        #     params = mobility_params
        # )
        

        barrier.wait()
        # end_time_secs = time.time() + duration  # Half second for command execution
        # command_client.robot_command(traj_cmd, end_time_secs=end_time_secs)
        time.sleep(5)
        robot.logger.info(f'Robot {robot_id}: Finished {distance} m bump')
        # Power off
        robot.logger.info(f'Robot {robot_id}: Sitting down and turning off.')
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), f'Robot {robot_id} power off failed.'
        robot.logger.info(f'Robot {robot_id}: Safely powered off.')
        
        return True

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to walk up to something'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def relative_move(dx, dy, dyaw, dt, frame_name, robot_command_client, robot_state_client, mobility_params):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    command_arm = RobotCommandBuilder.arm_joint_freeze_command()

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=mobility_params, build_on_command=command_arm)
    end_time = dt
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the goal.')
            return True
        time.sleep(1)


def arg_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{repr(x)} not a number')
    return x


def get_hand_pose_for_example():
    """
    A hand pose in front of and to the side of the robot.  This pose of the end-effector
    was chosen such that the robot can walk forwards/backwards without having the body of
    the robot collide with the hand.
    """

    # Position of hand frame is in front of and to the right of the body.
    x = 0.9
    y = 0.0
    z = 0.0
    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    # Orientation of the hand is a 90-degree rotation about body z, e.g. x_hand points along -y_body
    qw = 1
    qx = 0
    qy = 0
    qz = 0
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

    return geometry_pb2.SE3Pose(position=hand_ewrt_flat_body, rotation=flat_body_Q_hand)



def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-d', '--distance', help='Distance from object to walk to (meters).',
                        default=None, type=arg_float)
    
    parser.add_argument('--robot1-hostname', required=True, help='Hostname/IP for first robot')
    parser.add_argument('--robot2-hostname', required=True, help='Hostname/IP for second robot')
    parser.add_argument('--robot1-username', help='Username for first robot')
    parser.add_argument('--robot1-password', help='Password for first robot')
    parser.add_argument('--robot2-username', help='Username for second robot')
    parser.add_argument('--robot2-password', help='Password for second robot')
    
    options = parser.parse_args()
    bosdyn.client.util.setup_logging(options.verbose)

    global robot_image_clicks, robot_image_displays
    # robot_image_clicks = {}
    # robot_image_displays = {}

    try:
        # Phase 1: Sequential target collection
        print("=== Collecting targets for both robots ===")
        
        robot_configs = []
        
        # Collect target for Robot1
        robot1_config = collect_target_for_robot(
            options, 'Robot1', options.robot1_hostname, 
            options.robot1_username, options.robot1_password
        )
        if robot1_config is None:
            print("Failed to collect target for Robot1")
            return False
        robot_configs.append(robot1_config)
        
        # Collect target for Robot2
        robot2_config = collect_target_for_robot(
            options, 'Robot2', options.robot2_hostname,
            options.robot2_username, options.robot2_password
        )
        if robot2_config is None:
            print("Failed to collect target for Robot2")
            return False
        robot_configs.append(robot2_config)
        
        # Show summary and get user confirmation
        print("\n=== Target Summary ===")
        for config in robot_configs:
            print(f"{config['robot_id']}: Target at ({config['target_point'][0]}, {config['target_point'][1]})")
        
        input("Press Enter to start synchronized robot execution...")
        
        # Synchronized robot execution
        print("\n=== Synchronized robot execution ===")
        
        import threading

        num_parties = 3
        sync_barrier = threading.Barrier(num_parties)
        
        ########## 
        # Phi
        ##########
        radius = 3
        v_x = 0.5
        v_ang = v_x/radius
        velocity = [v_x, 0, v_ang]
        distance = [(2*np.pi*radius)/4, 0, np.pi/2]
        duration = 6.5
        thread1 = threading.Thread(
            target=execute_walk_for_robot,
            args=(options, robot_configs[0], sync_barrier, velocity, distance, duration, radius)
        )

        ########## 
        # Psi
        ##########
        radius += 1
        v_ang = v_x/radius
        velocity = [v_x, 0, v_ang]
        distance = [(2*np.pi*radius)/4, 0.0, np.pi/2]
        duration = 6.5
        thread2 = threading.Thread(
            target=execute_walk_for_robot,
            args=(options, robot_configs[1], sync_barrier, velocity, distance, duration, radius)
        )
        
        # Start both threads simultaneously
        print("Starting both robots simultaneously...")
        thread1.start()
        thread2.start()

        try:
            # === MAIN THREAD AT CHECKPOINT 2 ===
            print("MAIN: Waiting for all robots to stand...")
            sync_barrier.wait()
            print("MAIN: All robots are standing. Walk command initiated.")

            # === MAIN THREAD AT CHECKPOINT 3 ===
            print("MAIN: Waiting for all robots to reach the object")
            sync_barrier.wait()
            print("MAIN: All robots have reached the object.")

            # === MAIN THREAD AT CHECKPOINT 3 ===
            print("MAIN: Waiting for all robots to reach the object")
            sync_barrier.wait()
            print("MAIN: All robots have reached the object.")

            for i in range(10):
                print (f"Syncing waypoint {i}")
                sync_barrier.wait()
                print (f"Synced waypoint {i}")

            # === MAIN THREAD AT CHECKPOINT 3 ===
            print("MAIN: Waiting for all robots to complete their task...")
            sync_barrier.wait()
            print("MAIN: All robots have completed their task.")

        except threading.BrokenBarrierError:
            print("MAIN: A barrier was broken! One of the robots likely failed its task.")
        
        # Wait for both to complete
        thread1.join()
        thread2.join()
        
        print("Both robots completed their walk-to-object tasks")
        
        # Clean up any remaining windows
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)
