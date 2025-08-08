"""
spot_camera.py

Handles image acquisition and decoding for Spot robot.

Author: Tim
Date: June 26, 2025
"""

import numpy as np
import cv2 

from bosdyn.api import image_pb2

class SpotCamera:
    """
    Encapsulates image capture and decoding for a Spot robot.
    """
    def __init__(self, id, spot_client):
        self.id = id
        self._spot_client = spot_client

    def take_picture(self, color_src: str = None, depth_src: str = None,
                     save_images: bool = False):
        """
        Takes a picture with the robot's camera.

        Parameters:
            color_source (str, optional): The rgb camera to use
            depth_source (str, optional): The depth camera to use
            save_images (bool): Whether or not to save the images

        Returns:
            The images
        """
        image_client = self._spot_client._image_client

        sources = []
        order = []
        if color_src:
            sources.append(color_src)
            order.append("color")
        if depth_src:
            sources.append(depth_src)
            order.append("depth")

        if not sources:
            print("No image sources provided")
            return None
        
        print(f'{self.id}: Getting image(s) from: {sources}')
        image_responses = image_client.get_image_from_sources(sources)
        if len(image_responses) != len(sources):
            print(f'{self.id}: Got {len(image_responses)} \
                  images, expected {len(sources)}.')
            return None
        
        imgs = {}
        for idx, typ in enumerate(order):
            img_proto = image_responses[idx].shot.image
            imgs[typ] = self.decode_image(img_proto)

        if "color" in imgs and "depth" in imgs:
            if save_images:
                SpotCamera.save_image(imgs["color"], f"images/{self.id}_color.jpg")
                SpotCamera.save_image(imgs["depth"], f"images/{self.id}_depth.jpg")
                SpotCamera.save_color_depth_overlay(
                    imgs["color"], imgs["depth"], 
                    out_path=f"images/{self.id}_color_and_depth.jpg"
                )
            return imgs["color"], imgs["depth"]
        elif "color" in imgs:
            if save_images:
                SpotCamera.save_image(imgs["color"], f"images/{self.id}_color.jpg")
            return imgs["color"]
        elif "depth" in imgs:
            if save_images:
                SpotCamera.save_image(imgs["depth"], f"images/{self.id}_depth.jpg")
            return imgs["depth"]
        else: 
            return None

    @staticmethod
    def decode_image(image):
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

    @staticmethod
    def save_image(image, out_path) -> None:
        """
        Saves an image to disk.
        Args:
            image (np.ndarray): The image to save.
            out_path (str): The file path to save the image.
        """
        cv2.imwrite(out_path, image)
        print(f"Saved image to {out_path}")

    @staticmethod
    def save_color_depth_overlay(color_img, depth_img, 
                                 out_path="images/color_and_depth.jpg") -> None:
        """
        Creates and saves an RGB+depth overlay.
        Args:
            color_img (np.ndarray): RGB image.
            depth_img (np.ndarray): Depth image (uint16).
            out_path (str): File path to save overlay.
        """
        # Convert color to RGB if needed
        if len(color_img.shape) == 3:
            visual_rgb = color_img  
        else: 
            visual_rgb = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)

        # Convert 16-bit depth to 8-bit for colormap
        min_val = np.min(depth_img)
        max_val = np.max(depth_img)
        depth_range = max_val - min_val if max_val > min_val else 1
        depth8 = ((depth_img - min_val) * 255.0 / depth_range).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

        # Combine the images
        overlay = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)
        cv2.imwrite(out_path, overlay)
        print(f"Saved overlay image to {out_path}")
    
    def list_sources(self):
        """
        Return all the available image sources
        """
        image_client = self._spot_client._image_client
        return image_client.list_image_sources()