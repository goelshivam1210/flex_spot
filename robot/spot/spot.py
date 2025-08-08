"""
spot.py

Manages connectiion, state, and actions for a single spot robot.

Author: Tim
Date: June 2025
"""

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

from .spot_client import SpotClient
from .spot_camera import SpotCamera
from .spot_perception import SpotPerception

class Spot:
    """Manages connection, state, and actions for a single robot."""

    def __init__(self, id, hostname, config=None):  #TODO Delete config
        self.id = id
        self.config = config

        self._client = SpotClient(id, hostname)
        self._camera = SpotCamera(id, self._client)

        self.target_point = None
        self.image_data = None

    @property
    def lease_client(self):
        return self._client._lease_client
    
    def start(self):
        return self._client.start()

    def power_on(self):
        return self._client.power_on()
    
    def take_picture(self, color_src:str=None, depth_src:str=None,
                     save_images:bool=False):
        return self._camera.take_picture(color_src, depth_src, save_images)

    def stand_up(self, timeout_sec: float = 10):
        """
        Power on the robot and command it to stand.
        """
        blocking_stand(self._client._command_client, timeout_sec=timeout_sec)
        print(f"{self.id}: Standing complete.")

    def open_gripper(self, timeout_sec: float = 5.0):
        """
        Open Spot's gripper using a claw command.
        """
        # Build and send gripper open command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = self._client._command_client.robot_command(gripper_cmd)
        
        # Wait until the gripper command completes
        block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec=timeout_sec)
        print(f"{self.id}: Gripper open complete.")

    def close_gripper(self, timeout_sec: float = 5.0):
        """
        Close Spot's gripper using a claw command.
        """
        # Build and send gripper open command
        gripper_cmd = RobotCommandBuilder.claw_gripper_close_command()
        cmd_id = self._client._command_client.robot_command(gripper_cmd)
        
        # Wait until the gripper command completes
        block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec=timeout_sec)
        print(f"{self.id}: Gripper close complete.")

    def unstow_arm(self, timeout_sec=3.0):
        """Unstow arm to ready position."""
        unstow_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = self._client._command_client.robot_command(unstow_cmd)
        block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec)
        print(f"{self.id}: Arm unstowed")

    def stow_arm(self, timeout_sec=3.0):
        """Stow arm back to resting position."""
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        cmd_id = self._client._command_client.robot_command(stow_cmd)
        block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec)
        print(f"{self.id}: Arm stowed")

    def walk_to_pixel(self, pixel_xy, img_src="hand_color_image", offset_distance=0.2, timeout = 15):
        """Walk to a pixel location without grasping."""
        cx, cy = pixel_xy
        img_client = self._client._image_client
        image_response = img_client.get_image_from_sources([img_src])[0]
        
        walk_vec = geometry_pb2.Vec2(x=cx, y=cy)
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec,
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
            camera_model=image_response.source.pinhole,
            offset_distance=wrappers_pb2.FloatValue(value=offset_distance)
        )
        
        request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
        response = self._client._manip_client.manipulation_api_command(request)
        
        # Wait for completion with timeout
        print(f"{self.id}: Walking to pixel ({cx}, {cy})...")
        start_time = time.time()
        timeout_duration = timeout
        
        while True:
            time.sleep(0.25)
            if time.time() - start_time > timeout_duration:
                print(f"{self.id}: Walk timed out after {timeout_duration}s")
                break
                
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=response.manipulation_cmd_id
            )
            fb_resp = self._client._manip_client.manipulation_api_feedback_command(fb_req)
            
            if fb_resp.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                print(f"{self.id}: Walk completed successfully")
                break
        
        return True
    
    def move_arm_to_position(self, target_position, timeout=3.0):
        """Move arm to target position in vision frame."""
        try:
            # Get current robot state for frame info
            robot_state = self._client._state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            
            if vision_T_hand is None:
                raise Exception("Could not get current hand pose")
            
            # Create target pose (keep current orientation)
            target_pose = SE3Pose(
                x=target_position[0],
                y=target_position[1], 
                z=target_position[2],
                rot=vision_T_hand.rot
            )
            
            # Send arm command
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
            )
            cmd_id = self._client._command_client.robot_command(arm_cmd)
            block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec=timeout)
            
            return True
            
        except Exception as e:
            print(f"{self.id}: Error moving arm to position: {e}")
            return False
    def perform_wiggle_movements(self, interactive_perception, start_position=None):
        """Perform wiggle movements and return trajectory for joint analysis."""
        try:
            # Get current hand position if not provided
            if start_position is None:
                robot_state = self._client._state_client.get_robot_state()
                snapshot = robot_state.kinematic_state.transforms_snapshot
                vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
                
                if vision_T_hand is None:
                    raise Exception("Could not get hand pose")
                    
                start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
            
            print(f"{self.id}: Starting wiggle movements from position: {start_position}")
            
            # Generate wiggle positions
            wiggle_positions = interactive_perception.generate_wiggle_positions(start_position)
            print(f"{self.id}: Generated {len(wiggle_positions)} wiggle positions")
            
            # Collect actual trajectory data
            trajectory = []
            
            for i, target_pos in enumerate(wiggle_positions):
                print(f"{self.id}: Moving to position {i+1}/{len(wiggle_positions)}")
                
                # Move to target position
                if self.move_arm_to_position(target_pos):
                    # Get actual achieved position
                    time.sleep(0.5)  # Brief pause to settle
                    robot_state = self._client._state_client.get_robot_state()
                    snapshot = robot_state.kinematic_state.transforms_snapshot
                    actual_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
                    
                    if actual_hand_pose:
                        actual_position = np.array([actual_hand_pose.x, actual_hand_pose.y, actual_hand_pose.z])
                        trajectory.append(actual_position)
                    else:
                        trajectory.append(target_pos)  # Fallback
                else:
                    trajectory.append(target_pos)  # Fallback
            
            trajectory = np.array(trajectory)
            print(f"{self.id}: Collected trajectory with {len(trajectory)} points")
            
            return trajectory
            
        except Exception as e:
            print(f"{self.id}: Error during wiggle movements: {e}")
            return None   
    def dock(self, dock_id=521, timeout_sec=10.0):
        """Dock the robot at the specified docking station."""
        try:
            print(f"{self.id}: Starting docking sequence...")
            
            # Stand up before docking
            blocking_stand(self._client._command_client, timeout_sec=timeout_sec)
            
            # Initialize docking client if not already done
            docking_client = self._client._spot.ensure_client(DockingClient.default_service_name)
            
            # Execute docking
            blocking_dock_robot(self._client._spot, dock_id=dock_id)
            
            print(f"{self.id}: Robot docked successfully at station {dock_id}")
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"{self.id}: Error during docking: {e}")
            return False

    def align_to_box_with_pointcloud(self, region=None, angle_threshold_deg=5):
        """
        Rotates Spot in place using the body camera point cloud until aligned normal to the box.
        region: (x1, y1, x2, y2) in image pixels to crop (optional)
        angle_threshold_deg: allowed deviation from normal (in degrees)
        """
        max_attempts = 10
        for _ in range(max_attempts):
            # Take picture from body camera
            img_result = self.take_picture()
            if img_result is None or not isinstance(img_result, tuple):
                print("Failed to capture color and depth images.")
                return False
            color_img, depth_img = img_result
            # Get camera intrinsics for the body camera
            image_client = self._spot.ensure_client(ImageClient.default_service_name)
            sources = image_client.list_image_sources()
            camera_model = None
            for src in sources:
                if src.name == self.config.image_source:
                    camera_model = src.pinhole
                    break
            if camera_model is None:
                print("Could not retrieve camera intrinsics for image source.")
                return False
            # Convert depth region to point cloud
            points = depth_to_point_cloud(depth_img, camera_model, region)
            normal, _ = fit_plane(points)
            if normal is None:
                print("Could not fit plane to box face.")
                return False
            # In camera frame, forward is [0,0,1]
            angle = np.arccos(np.clip(np.dot(normal, [0,0,1]), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            print(f"Box normal angle from forward: {angle_deg:.1f} deg")
            if abs(angle_deg) < angle_threshold_deg:
                print("Spot is aligned normal to the box.")
                return True
            # Rotate left or right based on sign of normal's y component (camera frame)
            if normal[0] < 0:
                yaw = 0.08
            else:
                yaw = -0.08
            print(f"Rotating {'left' if yaw>0 else 'right'} by small step to improve alignment.")
            cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=yaw)
            self._command_client.robot_command(cmd)
            time.sleep(1.0)
        print("Failed to align after several attempts.")
        return False
       
    def grasp_edge(self, grasp_pt, img_src="hand_color_image"):
        """
        Uses the Spot Manipulation API to grasp at the vertical edge detected in the color image.
        """
        cx, cy = grasp_pt  # decompose to x and y

        # Get fresh image
        img_client = self._client._image_client
        image_response = img_client.get_image_from_sources([img_src])[0]

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
        response = self._client._manip_client.manipulation_api_command(request)
        cmd_id = response.manipulation_cmd_id

        # Monitor for completion with a timeout
        max_wait = 15  # seconds
        poll_interval = 0.25
        start_time = time.time()
        while True:
            time.sleep(poll_interval)
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_id)
            fb_resp = self._client._manip_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=fb_req)
            state = fb_resp.current_state
            print(f"Manipulation feedback: {state} ({manipulation_api_pb2.ManipulationFeedbackState.Name(state)})")
            if state == manipulation_api_pb2.MANIP_STATE_DONE or \
            state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print(f"Robot {self.id}: Grasp complete.")
                break
            elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                print(f"Robot {self.id}: Grasp failed.")
                break
            elif time.time() - start_time > max_wait:
                print(f"Robot {self.id}: Grasp feedback timed out after {max_wait} seconds.")
                break
        return True

    def push_object(self, dx=0, dy=0, d_yaw=0, dt=10):
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
        robot_state_client = self._client._state_client
        robot_state = robot_state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        hand_in_body_proto = snapshot.child_to_parent_edge_map['hand'].parent_tform_child
        hold_pose = SE3Pose.from_proto(hand_in_body_proto)

        # robot_cmd = robot_command_pb2.RobotCommand()
        # impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
        # impedance_cmd.root_frame_name = GRAV_ALIGNED_BODY_FRAME_NAME
        # impedance_cmd.root_tform_task.CopyFrom(hold_pose.to_proto())
        # impedance_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())

        # arm_cmd = arm_command_pb2.ArmCommand()
        # arm_cmd.arm_impedance_command.CopyFrom(impedance_cmd)

        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
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
        mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)

        command_client = self._client._command_client

        command_arm = RobotCommandBuilder.arm_joint_freeze_command()
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    dx,
                    dy,
                    d_yaw,
                    transforms,
                    params = mobility_params,
                    build_on_command=command_arm
        )
        end_t = time.time() + dt
        cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_t)
        time.sleep(dt+1)


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

    def save_initial_yaw(self) -> float:
        """
        Gets the robot's current pose and returns only its yaw angle.

        Args:
            robot_state_client: The robot's state client.

        Returns:
            A float representing the robot's current yaw angle in radians.
        """
        state = self._client._state_client.get_robot_state()
        # This gets the transform from vision frame to body frame (SE2: x, y, angle)
        vision_tform_body = get_se2_a_tform_b(
            state.kinematic_state.transforms_snapshot, "vision", "body"
        )
        yaw = vision_tform_body.angle
        print(f"{self.id}: Yaw saved: {yaw:.2f} radians")
        return yaw
    
    # def return_to_saved_yaw(
    #     self,
    #     saved_yaw: float,
    # ):
    #     """Rotate Spot in place to the saved yaw (in the vision frame)."""
    #     print(f"{self.id}: Returning to saved yaw: {saved_yaw:.2f} radians...")
    #     state = self._client._state_client.get_robot_state()
    #     vision_tform_body = get_se2_a_tform_b(
    #         state.kinematic_state.transforms_snapshot, "vision", "body"
    #     )
    #     current_x = vision_tform_body.x
    #     current_y = vision_tform_body.y

    #     # target_pose = SE2Pose(x=current_x, y=current_y, angle=saved_yaw)
    #     cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
    #         goal_x=current_x,
    #         goal_y=current_y,
    #         goal_heading=saved_yaw,
    #         frame_name="vision"
    #     )
    #     self._client._command_client.robot_command(cmd)
    #     print(f"{self.id}: Command sent to rotate. Waiting...")
    #     import time
    #     time.sleep(3)

    def get_current_pose(self):
        state = self._client._state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        vision_tform_body = get_se2_a_tform_b(snapshot, "vision", "body")
        x, y, yaw = vision_tform_body.x, vision_tform_body.y, vision_tform_body.angle
        print(f"{self.id}: Current pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
        return x, y, yaw

    def return_to_saved_yaw(self, saved_yaw: float, tolerance=0.02, max_time=5):
        """Rotate Spot in place to the saved yaw (in the vision frame)."""
        def wrap_to_pi(angle):
            """Wrap an angle in radians to [-pi, pi]."""
            return (angle + math.pi) % (2 * math.pi) - math.pi
        
        print(f"{self.id}: Returning to saved yaw: {saved_yaw:.2f} radians...")

        current_x, current_y, current_yaw = self.get_current_pose()
        diff = current_yaw - saved_yaw
        print(f"{self.id}: Yaw diff: {diff:.2f} radians")

        # 1. Take fresh snapshot
        snapshot = self._client._state_client.get_robot_state().kinematic_state.transforms_snapshot

        # 2. Get current hand pose in vision (world) frame
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        if vision_T_hand is None:
            raise RuntimeError("Hand transform not found")

        # 3. Build and send freeze command using that pose
        hand_pose_proto = vision_T_hand.to_proto()
        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
            hand_pose_proto, VISION_FRAME_NAME, seconds=2.0)

        sync_cmd = RobotCommandBuilder.build_synchro_command(arm_cmd)
        cmd_id = self._client._command_client.robot_command(sync_cmd)
        block_until_arm_arrives(self._client._command_client, cmd_id, timeout_sec=3.0)
        print("Hand now frozen in world (vision) frame.")
        
        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=True,
                                            disable_vision_foot_obstacle_avoidance=True,
                                            disable_vision_foot_constraint_avoidance=True,
                                            obstacle_avoidance_padding=.001)

        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=0.5, y=0.5), angular=1))        
        mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)

        # Send one-shot trajectory command to saved yaw
        end_time = time.time() + max_time
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=current_x,
            goal_y=current_y,
            goal_heading=saved_yaw,
            params = mobility_params,
            frame_name='vision'
        )
        self._client._command_client.robot_command(cmd, end_time_secs=end_time)
        print(f"{self.id}: Positional rotation command sent.")
        time.sleep(max_time)
        print(f"{self.id}: Done rotating.")


if __name__ == "__main__":
    import argparse

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

    spot.start()

    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        spot.power_on()
        spot.stand_up()

        # 1. Save initial yaw
        saved_yaw = spot.save_initial_yaw()

        # 2. Walk forward by 1 meter (no rotation)
        # walk_distance = 1.5  # meters
        # command_client = spot._client._command_client
        # state_client = spot._client._state_client
        # robot_state = state_client.get_robot_state()
        # transforms = robot_state.kinematic_state.transforms_snapshot

        # duration = 4.0
        # end_time = time.time() + duration

        # # Walk forward
        # traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #     walk_distance, 0.0, 0.0, transforms
        # )
        # cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_time)
        # time.sleep(duration)

        # print("finished walking forwards")

        # radians = -math.pi / 2  # -90 degrees

        # end_time = time.time() + duration
        # x, y, _ = spot.get_current_pose()
        # # Rotate 90° CW
        # traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #     x, y, radians, transforms
        # )
        # cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_time)
        
        # print(f"rotating {radians} radians.")
        # time.sleep(duration)
        # print("finished rotating in place")


        # # 4. Rotate back to original heading using your return_to_saved_yaw function
        # spot.return_to_saved_yaw(saved_yaw)

        # 1. Take picture of box and get grasp point
        # spot.open_gripper()
        # spot.close_gripper()

        # color_img, depth_img = spot.take_picture(
        #     color_src=hand_img_src,
        #     depth_src=hand_depth_src,
        #     save_images=True
        # )
        # grasp_pt = SpotPerception.get_vertical_edge_grasp_point(
        #     color_img, depth_img, spot.id, save_img=True
        # )

        # # 2. Grasp edge
        # spot.grasp_edge(grasp_pt)
        # spot.open_gripper()  # Keep gripper open to prevent losing grip

        # # 3. Push box
        # spot.push_object()

##############
# Old Code
##############

    # def walk_to_target(self, pixel_xy: tuple, image_data, offset_distance: float = None):
    #     """
    #     Walk to a target specified in pixel coordinates from an image.
    #     Args:
    #         pixel_xy: (x, y) tuple in pixel coordinates.
    #         image_data: ImageResponse returned by take_picture.
    #         offset_distance: optional float distance in meters to stop before target.
    #     """
    #     # Ensure manipulation client is initialized
    #     if not hasattr(self, 'manip_client'):
    #         self.setup_clients()
    #     # Build Vec2 and optional offset
    #     walk_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
    #     od = None if offset_distance is None else wrappers_pb2.FloatValue(value=offset_distance)
    #     walk_to = manipulation_api_pb2.WalkToObjectInImage(
    #         pixel_xy=walk_vec,
    #         transforms_snapshot_for_camera=image_data.shot.transforms_snapshot,
    #         frame_name_image_sensor=image_data.shot.frame_name_image_sensor,
    #         camera_model=image_data.source.pinhole,
    #         offset_distance=od
    #     )
    #     request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
    #     response = self._manip_client.manipulation_api_command(manipulation_api_request=request)
    #     # Wait for completion
    #     while True:
    #         time.sleep(0.25)
    #         fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
    #             manipulation_cmd_id=response.manipulation_cmd_id)
    #         fb_resp = self._manip_client.manipulation_api_feedback_command(
    #             manipulation_api_feedback_request=fb_req)
    #         if fb_resp.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
    #             print(f"Robot {self.spot_id}: Reached target.")
    #             break