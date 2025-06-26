"""Manages connectiion, state, and actions for a single spot robot."""
import time
import numpy as np

import cv2

from bosdyn.api import(
    arm_command_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2
)

import bosdyn.client
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import (
  RobotCommandBuilder,
  RobotCommandClient,
  blocking_stand,
  block_until_arm_arrives
)
from bosdyn.client.robot_state import RobotStateClient

from google.protobuf import wrappers_pb2

def depth_to_point_cloud(depth_img, camera_model, region=None):
    # Assume depth in millimeters, shape [H,W]
    if region:
        x1, y1, x2, y2 = region
        depth_crop = depth_img[y1:y2, x1:x2]
        xs, ys = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    else:
        h, w = depth_img.shape
        depth_crop = depth_img
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    depths = depth_crop.flatten().astype(np.float32) / 1000.0  # mm to meters
    xs = xs.flatten()
    ys = ys.flatten()
    # Camera intrinsics
    fx = camera_model.intrinsics.focal_length.x
    fy = camera_model.intrinsics.focal_length.y
    cx = camera_model.intrinsics.principal_point.x
    cy = camera_model.intrinsics.principal_point.y
    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points = np.stack([X, Y, Z], axis=1)
    return points

def fit_plane(points):
    """Fit a plane to points using least squares. Returns (normal, offset)."""
    valid = ~np.isnan(points).any(axis=1) & (points[:,2] > 0.2) & (points[:,2] < 3.0)  # filter valid range
    pts = points[valid]
    if pts.shape[0] < 10:
        return None, None
    # Fit plane to points: Ax + By + Cz + D = 0
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    U, S, Vt = np.linalg.svd(pts_centered)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    d = -centroid.dot(normal)
    return normal, d

class Spot:
    """Manages connectiion, state, and actions for a single robot."""

    def __init__(self, spot_id, hostname, config):
        self.spot_id = spot_id
        self.hostname = hostname
        self.config = config

        self._sdk = None
        self._spot = None        
        self.lease_client = None
        self._manip_client = None
        self._command_client = None

        self.target_point = None
        self.image_data = None

    # Helper function to setup clients
    def setup_clients(self):
        """
        Initialize lease and manipulation API clients on self.
        """
        self.lease_client = self._spot.ensure_client(LeaseClient.default_service_name)
        self._manip_client = self._spot.ensure_client(ManipulationApiClient.default_service_name)
        self._command_client = self._spot.ensure_client(RobotCommandClient.default_service_name)

    def connect(self):
        """Connects to the robot and authenticates."""
        try:
            self._sdk = bosdyn.client.create_standard_sdk(
                f'Controller_{self.spot_id}'
            )
            self._spot = self._sdk.create_robot(self.hostname)
            bosdyn.client.util.authenticate(self._spot)
            self._spot.time_sync.wait_for_sync()
            print(f"{self.spot_id}: Connection successful.")
            return True
        except Exception as e:
            print(f"{self.spot_id}: Failed to connect - {e}")
            return False            


    def get_target_from_user(self):
        """
        Displays an image from the robot's camera and waits for user to
        click on a target.
        """
        cv_image = self.take_picture()
        # Create a dictionary to hold state for the callback, avoiding globals.
        callback_data = {'clicked_point': None, 'display_image': cv_image}

        # Show image and wait for click
        print(f'Robot {self.spot_id}: Click on an object to walk up to...')
        image_title = 'Click to walk up to target'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(
            image_title, self._get_target_point, param=(self.spot_id, img_data)
        )


    def take_picture(self, return_proto=False):
        """
        Takes a picture with the robot's camera.
        Returns:
            If depth image source is configured, returns (cv_image, depth_image).
            Otherwise, returns just the color image.
        """
        # Ensure we have a client
        if not self._spot:
            print(f"{self.spot_id}: Not connected. Cannot get image.")
            return None

        image_client = self._spot.ensure_client(
            ImageClient.default_service_name
        )
        image_source = self.config.image_source
        depth_image_source = getattr(self.config, "depth_image_source", None)

        # sources = [image_source]
        # if depth_image_source:
        #     sources.append(depth_image_source)

        sources = [depth_image_source, image_source]

        print(f'Robot {self.spot_id}: Getting image(s) from: {sources}')
        image_responses = image_client.get_image_from_sources(sources)

        if len(image_responses) != len(sources):
            print(f'Robot {self.spot_id}: Got {len(image_responses)} images, expected {len(sources)}.')
            return None
        
        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
            cv_visual, cv2.COLOR_GRAY2RGB)

        # Map depth ranges to color

        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        # Add the two images together.
        out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)
        cv2.imwrite("color_and_depth.jpg", out)

        cv_image = self.decode_image(image_responses[1].shot.image)
        cv2.imwrite("original_image.png", cv_image)
        print("Saved depth image to original_image.png")
        if depth_image_source:
            depth_image = self.decode_image(image_responses[0].shot.image)
            cv2.imwrite("original_depth.png", depth_image)
            print("Saved depth image to original_depth.png")
            if return_proto:
                return (cv_image, depth_image, image_responses[0])
            return (cv_image, depth_image)
        else:
            if return_proto:
                return (cv_image, image_responses[0])
            return cv_image


    def decode_image(self, image, source_name=None):
        """
        Decodes the image data from the robot.
        Args:
            image (image_pb2.Image): The image data from the robot.
        Returns:
            cv_image: The decoded image in OpenCV format.
        """
        if image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8

        img_data = np.frombuffer(image.data, dtype=dtype)
        if image.format == image_pb2.Image.FORMAT_RAW:
            cv_image = img_data.reshape((image.rows, image.cols))
        else:
            cv_image = cv2.imdecode(img_data, -1)

        # if source_name in ["frontleft_fisheye_image", "frontright_fisheye_image"]:
        #     # Most Spot fisheye images are rotated 90 deg counterclockwise
        #     cv_image = np.rot90(cv_image, k=1)

        return cv_image

    def stand_up(self, timeout_sec: float = 10):
        """
        Power on the robot and command it to stand.
        """
        # Power on
        self._spot.power_on(timeout_sec=20)
        assert self._spot.is_powered_on(), "Power on failed."
        # Stand
        command_client = self._spot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=timeout_sec)
        print(f"Robot {self.spot_id}: Standing complete.")

    def walk_to_target(self, pixel_xy: tuple, image_data, offset_distance: float = None):
        """
        Walk to a target specified in pixel coordinates from an image.
        Args:
            pixel_xy: (x, y) tuple in pixel coordinates.
            image_data: ImageResponse returned by take_picture.
            offset_distance: optional float distance in meters to stop before target.
        """
        # Ensure manipulation client is initialized
        if not hasattr(self, 'manip_client'):
            self.setup_clients()
        # Build Vec2 and optional offset
        walk_vec = geometry_pb2.Vec2(x=pixel_xy[0], y=pixel_xy[1])
        od = None if offset_distance is None else wrappers_pb2.FloatValue(value=offset_distance)
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec,
            transforms_snapshot_for_camera=image_data.shot.transforms_snapshot,
            frame_name_image_sensor=image_data.shot.frame_name_image_sensor,
            camera_model=image_data.source.pinhole,
            offset_distance=od
        )
        request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
        response = self._manip_client.manipulation_api_command(manipulation_api_request=request)
        # Wait for completion
        while True:
            time.sleep(0.25)
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=response.manipulation_cmd_id)
            fb_resp = self._manip_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=fb_req)
            if fb_resp.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                print(f"Robot {self.spot_id}: Reached target.")
                break

    def open_gripper(self, timeout_sec: float = 5.0):
        """
        Open Spot's gripper using a claw command.
        """
        # Ensure command client is initialized
        if not self._command_client:
            self.setup_clients()
        # Build and send gripper open command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = self._command_client.robot_command(gripper_cmd)
        # Wait until the gripper command completes
        block_until_arm_arrives(self._command_client, cmd_id, timeout_sec=timeout_sec)
        print(f"Robot {self.spot_id}: Gripper open complete.")

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


    def print_behavior_faults(self):
        """
        Retrieve and print current behavior faults from Spot.
        """
        # Ensure state client is available
        state_client = self._spot.ensure_client(RobotStateClient.default_service_name)
        state = state_client.get_robot_state()
        faults = state.behavior_fault_state.faults
        if not faults:
            print(f"Robot {self.spot_id}: No behavior faults.")
        else:
            print(f"Robot {self.spot_id}: Current behavior faults:")
            for fault in faults:
                # Print full protobuf representation for each fault
                print(f"  - {fault}")


    def find_strongest_vertical_edge(self, cv_image):
        # Preprocess (convert to grayscale, blur, etc.)
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image.copy()        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,minLineLength=50, maxLineGap=10)
        best_line = None
        best_score = -np.inf
        h = cv_image.shape[0]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = np.hypot(dx, dy)
                # Check if vertical enough
                if dx < dy and length > 30:
                    # Score: longer, and closer to bottom of image (foreground)
                    avg_y = (y1 + y2) / 2
                    score = avg_y + 2 * length  # Heavily favor lines lower in image
                    if score > best_score:
                        best_score = score
                        best_line = (x1, y1, x2, y2)
        return best_line

    def grasp_vertical_edge(self):
        img = self.take_picture()
        if img is None:
            print("Failed to capture image.")
            return
        line = self.find_strongest_vertical_edge(img)
        if not line:
            print("No strong vertical edge found.")
            return
        # Pick the midpoint of the line as the grasp point
        x1, y1, x2, y2 = line
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        print(f"Attempting grasp at pixel ({cx}, {cy})")

        # Optionally, use depth here to get z
        # If you want to walk to the object:
        # self.walk_to_target((cx, cy), img_data=image_data_from_take_picture)

        # Move the arm to the edge (using manipulation API, or a simple arm pose)
        # Here, for demo, just open the gripper first:
        self.open_gripper()
        #TODO: Add code to move arm to the pose, then close gripper
        # This is where you'd use the Manipulation API to send a grasp command at (cx, cy)

    def get_depth_at_pixel(self, depth_image, cx, cy, search_radius=5):
        """
        Returns a valid depth value and pixel position near (cx, cy) in the depth image.
        Args:
            depth_image (np.ndarray): Depth image (uint16 or float32).
            cx, cy (int): Pixel coordinates.
            search_radius (int): Radius in pixels to search for a valid depth.
        Returns:
            (depth_value, px, py): depth in meters (float), pixel x, y. If not found, returns (None, None, None).
        """
        h, w = depth_image.shape[:2]
        # Try the center pixel first
        if 0 <= cx < w and 0 <= cy < h:
            d = depth_image[cy, cx]
            if np.isfinite(d) and d > 0:
                return (float(d) / 1000.0 if depth_image.dtype == np.uint16 else float(d), cx, cy)
        # Search in a small window for a valid depth
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    px = cx + dx
                    py = cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        d = depth_image[py, px]
                        if np.isfinite(d) and d > 0:
                            return (float(d) / 1000.0 if depth_image.dtype == np.uint16 else float(d), px, py)
        return (None, None, None)

    def pixel_to_camera_frame(self, u, v, depth, camera_model):
        """
        Convert image pixel (u, v) and depth to camera frame 3D point using pinhole intrinsics.
        Args:
            u, v: Pixel coordinates.
            depth: Depth in meters.
            camera_model: image_pb2.ImageSource.pinhole (PinholeCameraModel).
        Returns:
            (x, y, z): 3D point in camera frame.
        """
        # camera_model contains focal_length, center_x, center_y
        fx = camera_model.intrinsics.focal_length.x
        fy = camera_model.intrinsics.focal_length.y
        cx = camera_model.intrinsics.principal_point.x
        cy = camera_model.intrinsics.principal_point.y
        # Unproject
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return (x, y, z)

    def get_vertical_edge_grasp_point(self):
        """
        Detects the strongest vertical edge, finds a valid depth at/near its midpoint,
        draws the edge and target pixel on the color image, saves to disk, and returns the 3D grasp point in camera frame
        as (x, y, z) in meters if depth is available, else (cx, cy, None) where cx,cy are the target pixel coordinates.
        Always saves the grasp visualization image.
        Returns:
            (x, y, z): 3D grasp point in camera frame if depth available, else (cx, cy, None).
        """
        # 1. Capture color and depth image
        img_result = self.take_picture()
        if img_result is None or not isinstance(img_result, tuple):
            print("Failed to capture color and depth images.")
            return None
        color_img, depth_img = img_result
        
        # 2. Find strongest vertical edge
        line = self.find_strongest_vertical_edge(color_img)
        if not line:
            print("No strong vertical edge found.")
            return None
        x1, y1, x2, y2 = line
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # 3. Look up valid depth at or near (cx, cy)
        depth, px, py = self.get_depth_at_pixel(depth_img, cx, cy, search_radius=5)
        # 4. Draw detected edge and pixel on color image and save
        color_img_mark = color_img.copy()
        # Draw the detected vertical edge as a green line
        cv2.line(color_img_mark, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw the grasp pixel (red dot). If valid depth found, use (px,py), else use (cx,cy)
        if depth is not None:
            cv2.circle(color_img_mark, (px, py), 6, (0, 0, 255), -1)
        else:
            cv2.circle(color_img_mark, (cx, cy), 6, (0, 0, 255), -1)
        cv2.imwrite("edge_and_depth_pixel.png", color_img_mark)
        print(f"Saved edge and grasp pixel visualization to edge_and_depth_pixel.png")
        # 5. Retrieve camera model from Spot's API
        image_client = self._spot.ensure_client(ImageClient.default_service_name)
        image_source = self.config.image_source
        # Get image source info to get camera intrinsics
        sources = image_client.list_image_sources()
        camera_model = None
        for src in sources:
            if src.name == image_source:
                camera_model = src.pinhole
                break
        if camera_model is None:
            print("Could not retrieve camera intrinsics for image source.")
            return None
        # 6. Compute (x, y, z) grasp point in camera frame if depth available, else return pixel coords and None
        if depth is not None:
            grasp_point = self.pixel_to_camera_frame(px, py, depth, camera_model)
            print(f"Detected grasp point in camera frame: {grasp_point}")
            return grasp_point
        else:
            print("No valid depth at or near edge midpoint. Returning pixel coordinates and None for depth.")
            return (cx, cy, None)

    def grasp_at_edge(self):
        """
        Uses the Spot Manipulation API to grasp at the vertical edge detected in the color image.
        """
        img_result = self.take_picture()
        if img_result is None or not isinstance(img_result, tuple):
            print("Failed to capture color and depth images.")
            return False
        color_img, depth_img = img_result

        # Get grasp point
        result = self.get_vertical_edge_grasp_point()
        if result is None:
            print("No grasp point detected.")
            return False

        # Parse result
        if result[2] is None:
            # Only (cx, cy) available (no depth)
            cx, cy = int(result[0]), int(result[1])
            depth = None
        else:
            # (x, y, z) in camera frame; get pixel for grasp
            cx, cy = None, None
            # To improve: keep track of which pixel we used in get_vertical_edge_grasp_point and return it too!
            # For now, just detect again for demonstration.
            line = self.find_strongest_vertical_edge(color_img)
            if not line:
                print("No strong vertical edge found.")
                return False
            x1, y1, x2, y2 = line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Get latest image response from the robot, for the Manipulation API
        image_client = self._spot.ensure_client(ImageClient.default_service_name)
        image_source = self.config.image_source
        image_response = image_client.get_image_from_sources([image_source])[0]

        # Prepare Manipulation API request
        self.setup_clients()
        # WalkToObjectInImage not needed, just grasp
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
        response = self._manip_client.manipulation_api_command(request)
        cmd_id = response.manipulation_cmd_id

        # Monitor for completion
        # Monitor for completion with a timeout and more state checks
        max_wait = 30  # seconds
        poll_interval = 0.25
        start_time = time.time()
        while True:
            time.sleep(poll_interval)
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_id)
            fb_resp = self._manip_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=fb_req)
            state = fb_resp.current_state
            print(f"Manipulation feedback: {state} ({manipulation_api_pb2.ManipulationFeedbackState.Name(state)})")
            if state == manipulation_api_pb2.MANIP_STATE_DONE or \
            state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print(f"Robot {self.spot_id}: Grasp complete.")
                break
            elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                print(f"Robot {self.spot_id}: Grasp failed.")
                break
            elif time.time() - start_time > max_wait:
                print(f"Robot {self.spot_id}: Grasp feedback timed out after {max_wait} seconds.")
                break
        return True


    def push_object(self, dx=1, dy=0.5, distance=0.3, speed=0.2):
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
        robot_state_client = self._spot.ensure_client(RobotStateClient.default_service_name)
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

        dt = 2 
        command_client = self._spot.ensure_client(RobotCommandClient.default_service_name)

        command_arm = RobotCommandBuilder.arm_joint_freeze_command()
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
        traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    dx,
                    dy,
                    0,
                    transforms,
                    # params = mobility_params,
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

    # Connect to Spot and capture image
    spot = Spot("Spot", args.hostname, config)
    if not spot.connect():
        print("Failed to connect to Spot.")
        exit(1)

    # image_client = controller._spot.ensure_client(ImageClient.default_service_name)
    # for src in image_client.list_image_sources():
    #     print(src.name)
    
    spot.setup_clients()
    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        spot.stand_up()
        # controller.print_behavior_faults()
        spot.open_gripper()

        # 1. Approach the box (using body camera, walk to target)
        # color_img, depth_img, image_data = controller.take_picture(return_proto=True)
        # print("Depth min/max:", np.min(depth_img), np.max(depth_img))
        pt = spot.get_vertical_edge_grasp_point()
        # controller.walk_to_target((pt[0], pt[1]), image_data=image_data, offset_distance=1)

        # 2. Align normal using body camera and point cloud
        # controller.align_to_box_with_pointcloud(region=None, angle_threshold_deg=5)

        # 3. Switch to hand camera, grip
        spot.grasp_at_edge()

        spot.open_gripper()

        # # 4. Push
        spot.push_object()

        # img_result = controller.take_picture()
        # if img_result is None:
        #     print("Failed to capture image.")
        #     exit(1)

        # pt = controller.get_vertical_edge_grasp_point()
        # if pt is not None:
        #     print("Grasp point (camera frame):", pt)

        # controller.grasp_at_edge()

        # controller.push_object()

    # Save original capture(s) for debugging
    # if isinstance(img_result, tuple):
    #     color_img, depth_img = img_result
    #     cv2.imwrite("original_image.png", color_img)
    #     cv2.imwrite("original_depth.png", depth_img)
    #     print("Saved original image to original_image.png")
    #     print("Saved depth image to original_depth.png")
    # else:
    #     cv2.imwrite("original_image.png", img_result)
    #     print("Saved original image to original_image.png")

    # detect_bounding_box(img)