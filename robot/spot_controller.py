"""Manages connectiion, state, and actions for a single robot."""
import numpy as np

import cv2

import bosdyn.client
from bosdyn.api import image_pb2

from bosdyn.api import geometry_pb2, manipulation_api_pb2
from google.protobuf import wrappers_pb2
import time
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
# Add gripper command imports
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives

# Import RobotStateClient for behavior fault inspection
from bosdyn.client.robot_state import RobotStateClient

# === OpenAI and helper imports ===
import os
import base64
import json
import re

from openai import OpenAI

from functools import wraps
from bosdyn.client.lease import LeaseKeepAlive


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SpotController:
    """Manages connectiion, state, and actions for a single robot."""

    def __init__(self, spot_id, hostname, config):
        self.spot_id = spot_id
        self.hostname = hostname
        self.config = config

        self._sdk = None
        self._spot = None        
        self._lease_client = None
        self._manip_client = None
        self._command_client = None

        self.target_point = None
        self.image_data = None

    # Helper function to setup clients
    def setup_clients(self):
        """
        Initialize lease and manipulation API clients on self.
        """
        self._lease_client = self._spot.ensure_client(LeaseClient.default_service_name)
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
        
    def main_movement(self):
        """ The main movement script of the spot
        """
        self.setup_clients()
        with LeaseKeepAlive(self._lease_client, must_acquire=True, return_at_exit=True):
            self.stand_up()
            # controller.print_behavior_faults()
            self.open_gripper()



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


    def take_picture(self):
        """
        Takes a picture with the robot's camera.
        Returns:
            cv_image: The image captured by the robot.
        """
        # Ensure we have a client
        if not self._spot:
            print(f"{self.spot_id}: Not connected. Cannot get image.")
            return None

        image_client = self._spot.ensure_client(
            ImageClient.default_service_name
        )
        image_source = self.config.image_source

        # Take picture with Spot
        print(f'Robot {self.spot_id}: Getting image from: {image_source}')
        image_responses = image_client.get_image_from_sources([image_source])

        if len(image_responses) != 1:
            print(f'Robot {self.spot_id}: Got {len(image_responses)} images, \
                    expected 1.')
            return None

        image = image_responses[0].shot.image
        return self.decode_image(image)


    def decode_image(self, image):
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
        response = self.manip_client.manipulation_api_command(manipulation_api_request=request)
        # Wait for completion
        while True:
            time.sleep(0.25)
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=response.manipulation_cmd_id)
            fb_resp = self.manip_client.manipulation_api_feedback_command(
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


# === GPT-4o bounding box helper and script ===
def detect_bounding_box(image):
    # Downscale image to reduce payload size
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image
    image_path = "original_image.png"

    # Getting the Base64 string
    base64_image = encode_image(image_path)

    # Load image to determine dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Image dimensions: width={w}, height={h}")


    prompt_text = (
        f"The image is {w} pixels wide and {h} pixels tall. "
        "Return the bounding box coordinates [x1, y1, x2, y2] of the vertical \
        edge on the right side of the wooden partition or panel in the foreground of the image, \
        in pixel coordinate space."
    )

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system",
            "content": (
                "You are an assistant that only returns JSON. "
                "Given an image embed, you MUST output exactly "
                "one JSON array [x1,y1,x2,y2], nothing else."
            )},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    print(f"OpenAI API response {response.output_text}")

    # Parse JSON bounding box coordinates
    content = response.output_text.strip()
    # Extract JSON array from descriptive text
    match = re.search(r'\[.*?\]', content)
    if not match:
        raise RuntimeError(f"Could not find bounding box JSON in response: {content}")
    coords = json.loads(match.group(0))
    x1, y1, x2, y2 = coords

    # Load original image, draw box, and save annotated image
    img = cv2.imread(image_path)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    annotated_path = "annotated_image.png"
    cv2.imwrite(annotated_path, img)
    print(f"Saved annotated image to {annotated_path}")

if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", required=True, help="Spot robot hostname or IP")
    parser.add_argument("--image-source", default="hand_color_image", help="Camera source to capture from")
    args = parser.parse_args()

    # Minimal config stub
    class Config: pass
    config = Config()
    config.image_source = args.image_source

    # Connect to Spot and capture image
    controller = SpotController("Spot", args.hostname, config)
    if not controller.connect():
        print("Failed to connect to Spot.")
        exit(1)

    controller.main_movement()
    
    img = controller.take_picture()
    if img is None:
        print("Failed to capture image.")
        exit(1)

    # Save original capture for debugging
    cv2.imwrite("original_image.png", img)
    print("Saved original image to original_image.png")

    # Run detection
    detect_bounding_box(img)
    # Draw bbox and display
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # # Save annotated image
    # cv2.imwrite("annotated_image.png", img)
    # print("Saved annotated image to annotated_image.png")
    # # cv2.imshow("Detected Object", img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
