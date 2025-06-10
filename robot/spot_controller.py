"""Manages connectiion, state, and actions for a single robot."""
import numpy as np

import cv2

import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient


class SpotController:
    """Manages connectiion, state, and actions for a single robot."""

    def __init__(self, spot_id, hostname, config):
        self.spot_id = spot_id
        self.hostname = hostname
        self.config = config

        self._sdk = None
        self._spot = None
        self._command_client = None
        self._manipulation_api_client = None
        self._lease_client = None

        self.target_point = None
        self.image_data = None


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

        image_responses = image_client.get_image_from_source(
            image_source=image_source
        )

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
