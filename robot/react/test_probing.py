import argparse
import time
import math
import numpy as np
import cv2

from bosdyn.api import(
    arm_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2
)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import (
  RobotCommandBuilder,
  blocking_stand,
  block_until_arm_arrives
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.client import math_helpers
from bosdyn.client.math_helpers import SE2Pose, SE3Pose
from bosdyn.client.frame_helpers import get_se2_a_tform_b, get_a_tform_b, ODOM_FRAME_NAME, BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.docking import DockingClient, blocking_dock_robot

from google.protobuf import wrappers_pb2

from spot.spot import Spot
from spot.spot_client import SpotClient
from spot.spot_camera import SpotCamera
from spot.spot_perception import SpotPerception
from react.force_prober import ForceProber

parser = argparse.ArgumentParser()
parser.add_argument("--hostname", required=True, help="Spot robot hostname or IP")
parser.add_argument(
    "--image-source", 
    default="hand_color_image",
    # default = "hand_image",
    help="Camera source to capture from. Possible choices: back_fisheye_image, \
        frontleft_fisheye_image, frontright_fisheye_image, hand_color_image, \
        hand_color_in_hand_depth_frame, hand_image, left_fisheye_image, \
        right_fisheye_image"
)
parser.add_argument(
    "--depth-image-source",
    default="hand_depth_in_hand_color_frame",
    help="Depth image source to capture from. Possible choices: back_depth, \
        back_depth_in_visual_frame, frontleft_depth, frontleft_depth_in_visual_frame, \
        frontright_depth, frontright_depth_in_visual_frame, hand_depth, \
        hand_depth_in_hand_color_frame, left_depth, left_depth_in_visual_frame, \
        right_depth, right_depth_in_visual_frame"
)
args = parser.parse_args()

# Minimal config stub
class Config: pass
config = Config()
config.image_source = args.image_source
config.depth_image_source = args.depth_image_source

hand_img_src = "hand_color_image"
hand_depth_src = "hand_depth_in_hand_color_frame"

spot = Spot(id="Spot", hostname=args.hostname, config=config)
robot_side = "left"

spot.start()

with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
    spot.power_on()
    spot.stand_up()

    # # 1. Save initial yaw
    saved_yaw = spot.save_initial_yaw()

    command_client = spot._client._command_client
    state_client = spot._client._state_client
    robot_state = state_client.get_robot_state()
    transforms = robot_state.kinematic_state.transforms_snapshot
    force_probe = ForceProber()

    # 1. Take picture of box and get grasp point
    spot.open_gripper()

    color_img, depth_img = spot.take_picture(
        color_src=hand_img_src,
        depth_src=hand_depth_src,
        save_images=True
    )
    # grasp_pt = SpotPerception.get_vertical_edge_grasp_point(
    #     color_img, depth_img, spot.id, save_img=True
    # )
    target_pixel = SpotPerception.find_grasp_sam(
                    color_img, depth_img,
                    left=(robot_side == "left"),
                    conf=0.15,
                    max_distance_m=3.0,
                )
    spot.grasp_edge(target_pixel, img_src=args.image_source)
    # spot.open_gripper()


    # 2. Grasp edge
    # spot.grasp_edge(grasp_pt)
    force_probe.probe(spot)


    spot.open_gripper()

    robot_state = state_client.get_robot_state()
    transforms = robot_state.kinematic_state.transforms_snapshot
    duration = 5.0
    end_time = time.time() + duration


    # Walk backwards
    traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        -1, 0.0, 0.0, transforms
    )
    cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_time)
    time.sleep(duration)

    spot.return_to_saved_yaw(saved_yaw)


    # Test Probing


    # spot.open_gripper()  # Keep gripper open to prevent losing grip

    # 3. Push box
    # spot.push_object()