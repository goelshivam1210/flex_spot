import math
from datetime import time

import cv2
from bosdyn.api import geometry_pb2, manipulation_api_pb2, robot_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2, SE3Pose
from bosdyn.client.frame_helpers import get_a_tform_b, get_se2_a_tform_b, VISION_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.math_helpers import Quat
from bosdyn.client.robot_command import RobotCommandBuilder

from spot.spot_camera import SpotCamera
from spot_wrapper.wrapper import SpotWrapper


def get_current_yaw(spot: SpotWrapper) -> float:
    """
    Gets the robot's current pose and returns only its yaw angle.

    Args:
        SpotWrapper: The robot wrapper object to get the pose from.

    Returns:
        A float representing the robot's current yaw angle in radians.
    """

    # This gets the transform from vision frame to body frame (SE2: x, y, angle)
    vision_tform_body = get_se2_a_tform_b(
        spot.robot_state.kinematic_state.transforms_snapshot, "vision", "body"
    )
    yaw = vision_tform_body.angle
    print(f"Yaw: {yaw:.2f} radians")
    return yaw

def grasp_object(spot: SpotWrapper, grasp_pt):
    """
    Uses the Spot Manipulation API to grasp at the vertical edge detected in the color image.
    """
    cx, cy = grasp_pt  # decompose to x and y

    # get image TODO: is this right?
    image_response = spot.spot_images.get_hand_rgb_image()

    # Make and send grasp command
    grasp_vec = geometry_pb2.Vec2(x=cx, y=cy)
    pick = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=grasp_vec,
        transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
        frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
        camera_model=image_response.source.pinhole,
    )
    request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=pick
    )

    print(f"Requesting grasp at pixel ({cx}, {cy})")
    response = spot.manipulation_command(request)
    cmd_id = response.manipulation_cmd_id

    # Monitor for completion with a timeout
    # TODO: is this needed?
    max_wait = 15  # seconds
    poll_interval = 0.25
    start_time = time.time()
    while True:
        time.sleep(poll_interval)
        fb_resp = spot.get_manipulation_command_feedback(cmd_id)
        state = fb_resp.current_state
        print(f"Manipulation feedback: {state} ({manipulation_api_pb2.ManipulationFeedbackState.Name(state)})")
        if state == manipulation_api_pb2.MANIP_STATE_DONE or \
        state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            print(f"Robot {spot.id}: Grasp complete.")
            break
        elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            print(f"Robot {spot.id}: Grasp failed.")
            break
        elif time.time() - start_time > max_wait:
            print(f"Robot {spot.id}: Grasp feedback timed out after {max_wait} seconds.")
            break
    return True

def get_current_pose(spot: SpotWrapper):
    snapshot = spot.robot_state.kinematic_state.transforms_snapshot
    vision_tform_body = get_se2_a_tform_b(snapshot, "vision", "body")
    x, y, yaw = vision_tform_body.x, vision_tform_body.y, vision_tform_body.angle
    print(f"{spot.id}: Current pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
    return x, y, yaw

def return_to_saved_yaw(spot: SpotWrapper, saved_yaw: float, tolerance=0.02, max_time=5):
    """Rotate Spot in place to the saved yaw (in the vision frame)."""

    def wrap_to_pi(angle):
        """Wrap an angle in radians to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    print(f"{spot.id}: Returning to saved yaw: {saved_yaw:.2f} radians...")

    current_x, current_y, current_yaw = get_current_pose()
    diff = current_yaw - saved_yaw
    print(f"{spot.id}: Yaw diff: {diff:.2f} radians")

    # 1. Take fresh snapshot
    snapshot = spot.robot_state.kinematic_state.transforms_snapshot

    # 2. Get current hand pose in vision (world) frame
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    if vision_T_hand is None:
        raise RuntimeError("Hand transform not found")

    # 3. Build and send freeze command using that pose
    hand_pose_proto = vision_T_hand.to_proto()
    arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
        hand_pose_proto, VISION_FRAME_NAME, seconds=2.0)

    sync_cmd = RobotCommandBuilder.build_synchro_command(arm_cmd)
    cmd_id = spot.robot_command(sync_cmd)
    print("Hand now frozen in world (vision) frame.")

    obstacles = robot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                disable_vision_foot_obstacle_avoidance=True,
                                                disable_vision_foot_constraint_avoidance=True,
                                                obstacle_avoidance_padding=.001)

    speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
        linear=Vec2(x=0.5, y=0.5), angular=1))
    mobility_params = robot_command_pb2.MobilityParams(
        obstacle_params=obstacles, vel_limit=speed_limit,
        locomotion_hint=robot_command_pb2.HINT_AUTO)

    # Send one-shot trajectory command to saved yaw
    cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=current_x,
        goal_y=current_y,
        goal_heading=saved_yaw,
        params=mobility_params,
        frame_name='vision'
    )
    spot.robot_command(cmd, duration=max_time)
    print(f"{spot.id}: Positional rotation command sent.")
    time.sleep(max_time)
    print(f"{spot.id}: Done rotating.")


def push_object(spot: SpotWrapper, dx=0, dy=0, d_yaw=0, dt=10):
    """
    Push the grasped object by walking Spot's base in a given direction.
    Args:
        direction_x, direction_y: Unit vector for direction in Spot's BODY frame.
        distance: Distance to push in meters.
        speed: Speed in m/s.
    """
    # 1. Tool frame: identity (no offset)
    wr1_T_tool = SE3Pose(0, 0, 0, Quat())

    # 2. Hold the current hand pose in body frame
    snapshot = spot.robot_state.kinematic_state.transforms_snapshot
    hand_in_body_proto = snapshot.child_to_parent_edge_map['hand'].parent_tform_child
    hold_pose = SE3Pose.from_proto(hand_in_body_proto)

    # robot_cmd = robot_command_pb2.RobotCommand()
    # impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
    # impedance_cmd.root_frame_name = GRAV_ALIGNED_BODY_FRAME_NAME
    # impedance_cmd.root_tform_task.CopyFrom(hold_pose.to_proto())
    # impedance_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())

    # arm_cmd = arm_command_pb2.ArmCommand()
    # arm_cmd.arm_impedance_command.CopyFrom(impedance_cmd)

    obstacles = robot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                                disable_vision_foot_obstacle_avoidance=True,
                                                disable_vision_foot_constraint_avoidance=True,
                                                obstacle_avoidance_padding=.001)

    # vx = dx/dt
    # vy = dy/dt
    # v_yaw = d_yaw/dt

    vx = 0.5
    vy = 0.5
    v_yaw = 0.5
    speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=vx, y=vy), angular=v_yaw))
    mobility_params = robot_command_pb2.MobilityParams(
                obstacle_params=obstacles, vel_limit=speed_limit,
                locomotion_hint=robot_command_pb2.HINT_AUTO)

    command_arm = RobotCommandBuilder.arm_joint_freeze_command()
    transforms = spot.robot_state.kinematic_state.transforms_snapshot
    traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                dx,
                dy,
                d_yaw,
                transforms,
                params = mobility_params,
                build_on_command=command_arm
    )
    cmd_id = spot.robot_command(traj_cmd, duration=dt)

    # # 3. Build mobility command to walk in the desired direction
    # move_x = direction_x * distance
    # move_y = direction_y * distance
    # move_rot = 0.0

    # mobility_cmd = RobotCommandBuilder.synchro_velocity_command(
    #     v_x=move_x, v_y=move_y, v_rot=move_rot
    # )

    # # 4. Build synchronized robot command
    # command = RobotCommandBuilder.build_synchro_command(mobility_cmd, arm_cmd)

    # cmd_id = self._command_client.robot_command(command)
    # print(f"Robot {self.spot_id}: Pushing object {distance} meters in \
    #       direction ({direction_x}, {direction_y})")

    # # 5. Feedback blocking loop (unchanged)
    # timeout = distance / speed + 5
    # start = time.time()
    # while True:
    #     feedback = self._command_client.robot_command_feedback(cmd_id)
    #     mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
    #     status = mobility_feedback.status
    #     print(f"Mobility command status: {status}")
    #     if status == 2:  # STATUS_SUCCESS per API docs
    #         print(f"Push complete.")
    #         break
    #     if time.time() - start > timeout:
    #         print("Push command timed out.")
    #         break
    #     time.sleep(0.25)


def convert_to_cv_image(image_response) -> cv2.Mat:
    SpotCamera.decode_image(image_response.shot.image)