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
    robot_command_pb2,
    trajectory_pb2,
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

    def stand_up(self, timeout_sec: float = 20):
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

    def push_object(self, dx=0, dy=0, d_yaw=0, vx=0.5, vy= 0.5, v_yaw=0.5, dt=10):
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

    def push_object_impedance(
        self,
        dx=0, dy=0, d_yaw=0,
        stiffness=250.0,
        damping=30.0,
        dt=10,
    ):
        """
        Push grasped object by offsetting the arm equilibrium in body frame.

        Args:
            dx, dy: Position offset in body frame (m).
            d_yaw: Yaw rotation of the hand (rad).
            stiffness: Translational stiffness (N/m).
            damping: Translational damping (Ns/m).
            dt: Duration to hold the command (s).
        """
        command_client = self._client._command_client
        snapshot = self._client._state_client.get_robot_state().kinematic_state.transforms_snapshot
        body_T_hand = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        if body_T_hand is None:
            raise RuntimeError("Cannot get hand pose in body frame.")

        hand_T_body = body_T_hand.inverse()
        (hand_dx, hand_dy, hand_dz) = hand_T_body.rot.transform_point(dx, dy, 0)
        displacement = SE3Pose(hand_dx, hand_dy, hand_dz, Quat())
        cmd = self._build_impedance_cmd(
            GRAV_ALIGNED_BODY_FRAME_NAME, body_T_hand, displacement, stiffness, damping
        )
        command_client.robot_command(cmd)
        time.sleep(dt)

    def _build_impedance_cmd(self, root_frame, root_tform_task, desired_tool, stiffness, damping):
        """Build arm impedance command.

        Task frame is placed at root_tform_task relative to root_frame.
        Translational stiffness/damping use the caller's values for x/y and
        a fixed 500 N/m for z (keep height). Rotational gains are kept low
        (20 Nm/rad stiffness, 1 Nms/rad damping) to avoid the arm fighting
        wrist orientation changes — matching the gains used by ForceProber.

        Args:
            root_frame: Reference frame name (e.g. VISION_FRAME_NAME or
                        GRAV_ALIGNED_BODY_FRAME_NAME).
            root_tform_task: SE3Pose locating the task frame in root_frame.
            desired_tool: SE3Pose target for the tool in the task frame.
                          (0,0,0,identity) = hold current pose.
            stiffness, damping: Translational impedance gains (N/m, Ns/m).
        """
        cmd = robot_command_pb2.RobotCommand()
        imp = cmd.synchronized_command.arm_command.arm_impedance_command
        imp.root_frame_name = root_frame
        imp.root_tform_task.CopyFrom(root_tform_task.to_proto())
        imp.wrist_tform_tool.CopyFrom(SE3Pose(0, 0, 0, Quat()).to_proto())
        imp.diagonal_stiffness_matrix.CopyFrom(
            geometry_pb2.Vector(values=[stiffness, stiffness, 500.0, 20.0, 20.0, 20.0])
        )
        imp.diagonal_damping_matrix.CopyFrom(
            geometry_pb2.Vector(values=[damping, damping, damping, 1.0, 1.0, 1.0])
        )
        pt = trajectory_pb2.SE3TrajectoryPoint()
        pt.pose.CopyFrom(desired_tool.to_proto())
        traj = trajectory_pb2.SE3Trajectory()
        traj.points.append(pt)
        imp.task_tform_desired_tool.CopyFrom(traj)
        return cmd

    def _build_walk_with_arm_cmd(self, arm_cmd, snapshot, dx, dy, d_yaw, vx, vy, v_yaw):
        """Build body walk in vision frame, combined with an arm command."""
        vision_tform_body = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=True,
            disable_vision_foot_obstacle_avoidance=True,
            disable_vision_foot_constraint_avoidance=True,
            obstacle_avoidance_padding=0.001,
        )
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=vx, y=vy), angular=v_yaw
        ))
        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=obstacles,
            vel_limit=speed_limit,
            locomotion_hint=spot_command_pb2.HINT_AUTO,
        )
        return RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=vision_tform_body.x + dx,
            goal_y=vision_tform_body.y + dy,
            goal_heading=vision_tform_body.angle + d_yaw,
            frame_name=VISION_FRAME_NAME,
            params=mobility_params,
            build_on_command=arm_cmd,
        )

    def push_object_impedance_vf(
        self,
        dx=0, dy=0, d_yaw=0,
        vx=0.5, vy=0.5, v_yaw=0.5,
        dt=10,
        stiffness=600.0,
        damping=45.0,
        two_phase=False,
    ):
        """
        Push with arm impedance + body mobility. Drop-in replacement for push_object_vf.

        All displacements (dx, dy, d_yaw) are in the VISION frame.
        Vision-frame displacements are converted to the hand frame so the arm
        moves in the vision XY plane regardless of hand orientation.

        Single-phase (default):
            Impedance root = VISION frame. Task frame at hand's current vision
            pose. Arm target = hand + (dx,dy) in hand frame, which is a fixed
            goal in vision space. Body walks (dx, dy, d_yaw) simultaneously.
            Arm pushes toward goal; body catches up; arm relaxes as error drops.

        Two-phase (two_phase=True):
            1. Arm extends by (dx, dy) via impedance. Body stays.
            2. 0-offset impedance settle at new hand pose (1 s).
            3. Hand frozen at its current vision-frame pose
               (arm_pose_command_from_pose in VISION frame).
            4. Body walks (dx, dy, d_yaw). Hand stays fixed in world space.

        Args:
            dx, dy, d_yaw: Displacement in VISION frame (m, m, rad).
            vx, vy, v_yaw: Body velocity limits.
            dt: Total duration (s). In two-phase mode, split between phases.
            stiffness, damping: Translational impedance gains.
            two_phase: If True, arm pushes first then body catches up.
        """
        state_client = self._client._state_client
        command_client = self._client._command_client
        snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot

        hold = SE3Pose(0, 0, 0, Quat())

        # Vision (dx, dy, 0) → hand frame so arm moves in vision XY plane.
        hand_T_vision = get_a_tform_b(snapshot, "hand", VISION_FRAME_NAME)
        (hx, hy, hz) = hand_T_vision.rot.transform_point(dx, dy, 0)
        displacement = SE3Pose(hx, hy, hz, Quat())

        if two_phase:
            body_T_hand = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
            if body_T_hand is None:
                raise RuntimeError("Cannot get hand pose in body frame.")

            # 1. Arm extends via impedance; body stays.
            arm_cmd = self._build_impedance_cmd(
                GRAV_ALIGNED_BODY_FRAME_NAME, body_T_hand, displacement, stiffness, damping
            )
            command_client.robot_command(arm_cmd)
            phase1_dt = dt * 0.6
            time.sleep(phase1_dt)

            # 2. 0-offset impedance settle at new hand pose.
            snap2 = state_client.get_robot_state().kinematic_state.transforms_snapshot
            body_T_hand2 = get_a_tform_b(snap2, GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
            settle_cmd = self._build_impedance_cmd(
                GRAV_ALIGNED_BODY_FRAME_NAME, body_T_hand2, hold, stiffness, damping
            )
            command_client.robot_command(settle_cmd)
            time.sleep(1.0)

            # 3. Freeze hand in world (vision) frame — same pattern as
            #    return_to_saved_yaw: arm_pose_command_from_pose, send
            #    standalone, block until arm arrives.
            snap3 = state_client.get_robot_state().kinematic_state.transforms_snapshot
            vision_T_hand3 = get_a_tform_b(snap3, VISION_FRAME_NAME, "hand")
            if vision_T_hand3 is None:
                raise RuntimeError("Cannot get hand pose in vision frame.")
            freeze_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                vision_T_hand3.to_proto(), VISION_FRAME_NAME, seconds=2.0
            )
            sync_freeze = RobotCommandBuilder.build_synchro_command(freeze_cmd)
            freeze_id = command_client.robot_command(sync_freeze)
            block_until_arm_arrives(command_client, freeze_id, timeout_sec=3.0)

            # 4. Body catches up to the hand. Hand stays frozen because
            #    the separate body-only walk doesn't override the arm
            #    subsystem (same as return_to_saved_yaw).
            vision_tform_body = get_se2_a_tform_b(snap3, VISION_FRAME_NAME, BODY_FRAME_NAME)
            body_x, body_y = vision_tform_body.x, vision_tform_body.y
            hand_x, hand_y = float(vision_T_hand3.x), float(vision_T_hand3.y)
            gap = math.sqrt((hand_x - body_x)**2 + (hand_y - body_y)**2)
            target_x = body_x + 0.5 * (hand_x - body_x)
            target_y = body_y + 0.5 * (hand_y - body_y)
            print(f"{self.id}: Phase2 catchup — body=({body_x:.3f},{body_y:.3f}) "
                  f"hand=({hand_x:.3f},{hand_y:.3f}) gap={gap:.3f}m "
                  f"target=({target_x:.3f},{target_y:.3f})")

            obstacles = spot_command_pb2.ObstacleParams(
                disable_vision_body_obstacle_avoidance=True,
                disable_vision_foot_obstacle_avoidance=True,
                disable_vision_foot_constraint_avoidance=True,
                obstacle_avoidance_padding=0.001,
            )
            catchup_speed = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=0.5, y=0.5), angular=0.5
            ))
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles,
                vel_limit=catchup_speed,
                locomotion_hint=spot_command_pb2.HINT_AUTO,
            )
            walk_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=target_x,
                goal_y=target_y,
                goal_heading=vision_tform_body.angle + d_yaw,
                frame_name=VISION_FRAME_NAME,
                params=mobility_params,
            )
            phase2_dt = max(dt, 3.0)
            end_t = time.time() + phase2_dt
            command_client.robot_command(walk_cmd, end_time_secs=end_t)
            time.sleep(phase2_dt)

            # 5. Return hand to where it was before the body moved,
            #    using arm_pose_command_from_pose (the saved vision_T_hand3).
            return_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                vision_T_hand3.to_proto(), VISION_FRAME_NAME, seconds=2.0
            )
            sync_return = RobotCommandBuilder.build_synchro_command(return_cmd)
            return_id = command_client.robot_command(sync_return)
            block_until_arm_arrives(command_client, return_id, timeout_sec=3.0)

            # 6. 0-offset impedance settle at restored hand pose.
            snap4 = state_client.get_robot_state().kinematic_state.transforms_snapshot
            body_T_hand4 = get_a_tform_b(snap4, GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
            settle_cmd2 = self._build_impedance_cmd(
                GRAV_ALIGNED_BODY_FRAME_NAME, body_T_hand4, hold, stiffness, damping
            )
            command_client.robot_command(settle_cmd2)
            time.sleep(1.0)
        else:
            # Single-phase: arm target is a fixed point in VISION frame.
            # Root = vision frame so the goal doesn't drift as the body walks.
            vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            if vision_T_hand is None:
                raise RuntimeError("Cannot get hand pose in vision frame.")

            arm_cmd = self._build_impedance_cmd(
                VISION_FRAME_NAME, vision_T_hand, displacement, stiffness, damping
            )
            walk_cmd = self._build_walk_with_arm_cmd(
                arm_cmd, snapshot, dx, dy, d_yaw, vx, vy, v_yaw
            )
            end_t = time.time() + dt
            command_client.robot_command(walk_cmd, end_time_secs=end_t)
            time.sleep(dt + 1)

    def transform_sim_to_vision_frame(self, sim_dx, sim_dy, sim_d_yaw, initial_pose=None):
        """
        Transform simulation commands to vision frame coordinates.
        
        Args:
            sim_dx, sim_dy, sim_d_yaw: Commands from simulation (assuming sim starts at 0,0,0)
            initial_pose: (x, y, yaw) tuple of robot's initial pose in vision frame.
                          If None, uses current pose as reference.
        
        Returns:
            (dx, dy, d_yaw): Commands transformed to vision frame
        """
        if initial_pose is None:
            # Use current pose as reference
            current_x, current_y, current_yaw = self.get_current_pose()
            initial_pose = (current_x, current_y, current_yaw)
        
        init_x, init_y, init_yaw = initial_pose
        
        # Create rotation matrix for initial yaw
        cos_yaw = math.cos(init_yaw)
        sin_yaw = math.sin(init_yaw)
        
        # Rotate the simulation deltas by the initial yaw
        # This accounts for the robot's initial orientation
        vision_dx = cos_yaw * sim_dx - sin_yaw * sim_dy
        vision_dy = sin_yaw * sim_dx + cos_yaw * sim_dy
        vision_d_yaw = sim_d_yaw  # Yaw rotation is the same in any frame
        
        print(f"{self.id}: Sim commands: dx={sim_dx:.3f}, dy={sim_dy:.3f}, d_yaw={sim_d_yaw:.3f}")
        print(f"{self.id}: Vision commands: dx={vision_dx:.3f}, dy={vision_dy:.3f}, d_yaw={vision_d_yaw:.3f}")
        
        return vision_dx, vision_dy, vision_d_yaw

    def push_object_vf(self, dx=0, dy=0, d_yaw=0, vx=0.5, vy=0.5, v_yaw=0.5, dt=10):
        """
        Push the grasped object by walking Spot's base in a given direction in the VISION frame.
        Args:
            dx, dy, d_yaw: Delta movements in the VISION frame (meters and radians).
            vx, vy, v_yaw: Velocity limits for the movement.
            dt: Duration of the movement in seconds.
        """
        # Get current robot state and transforms
        robot_state_client = self._client._state_client
        robot_state = robot_state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        
        # Get current pose in vision frame
        vision_tform_body = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
        current_x = vision_tform_body.x
        current_y = vision_tform_body.y
        current_yaw = vision_tform_body.angle
        
        # Calculate target pose in vision frame
        target_x = current_x + dx
        target_y = current_y + dy
        target_yaw = current_yaw + d_yaw
        
        print(f"{self.id}: Pushing in vision frame - Current: ({current_x:.2f}, {current_y:.2f}, {current_yaw:.2f})")
        print(f"{self.id}: Target: ({target_x:.2f}, {target_y:.2f}, {target_yaw:.2f})")
        
        # Set up obstacle avoidance parameters
        obstacles = spot_command_pb2.ObstacleParams(
            disable_vision_body_obstacle_avoidance=True,
            disable_vision_foot_obstacle_avoidance=True,
            disable_vision_foot_constraint_avoidance=True,
            obstacle_avoidance_padding=.001
        )
        
        # Set up velocity limits
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=vx, y=vy), 
            angular=v_yaw
        ))
        
        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=obstacles, 
            vel_limit=speed_limit,
            locomotion_hint=spot_command_pb2.HINT_AUTO
        )
        
        # Freeze arm to maintain grasp
        command_arm = RobotCommandBuilder.arm_joint_freeze_command()
        
        # Create trajectory command in vision frame
        traj_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=target_x,
            goal_y=target_y,
            goal_heading=target_yaw,
            frame_name=VISION_FRAME_NAME,
            params=mobility_params,
            build_on_command=command_arm
        )
        
        # Execute command
        command_client = self._client._command_client
        end_t = time.time() + dt
        cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_t)
        
        print(f"{self.id}: Push command sent, waiting {dt} seconds...")
        time.sleep(dt + 1)
        print(f"{self.id}: Push complete.")

    def push_object_from_sim(
        self,
        dx=0, dy=0, d_yaw=0,
        vx=0.5, vy=0.5, v_yaw=0.5,
        dt=10,
        initial_pose=None,
        use_impedance=False,
        stiffness=600.0,
        damping=45.0,
        two_phase=False,
    ):
        """
        Push object using simulation commands transformed to vision frame.

        Args:
            dx, dy, d_yaw: Commands from simulation (assuming sim starts at 0,0,0).
            vx, vy, v_yaw: Body velocity limits.
            dt: Duration (s).
            initial_pose: (x, y, yaw) of robot's initial vision-frame pose.
            use_impedance: Use arm impedance control instead of base mobility.
            stiffness, damping: Impedance parameters (only when use_impedance=True).
            two_phase: When use_impedance=True, arm pushes first then body catches up.
        """
        vision_dx, vision_dy, vision_d_yaw = self.transform_sim_to_vision_frame(
            dx, dy, d_yaw, initial_pose
        )

        if use_impedance:
            self.push_object_impedance_vf(
                vision_dx, vision_dy, vision_d_yaw,
                vx, vy, v_yaw, dt,
                stiffness=stiffness, damping=damping,
                two_phase=two_phase,
            )
        else:
            self.push_object_vf(vision_dx, vision_dy, vision_d_yaw, vx, vy, v_yaw, dt)


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

    def check_grip(self):
        """
        Check if the robot is currently holding something.
        Returns:
            bool: True if holding something, False otherwise
        """
        robot_state = self._client._state_client.get_robot_state()
        is_holding = robot_state.manipulator_state.is_gripper_holding_item
        print(f"{self.id}: Gripper holding something? {is_holding}")
        return is_holding

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
        spot.open_gripper()
        spot.close_gripper()
        spot.open_gripper()
        # spot.stand_up()

        # 1. Save initial yaw
        # saved_yaw = spot.save_initial_yaw()
    

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