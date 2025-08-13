"""
spot_perception.py

Perception and geometry utilities for Spot vision and manipulation.

Author: Tim
Date: June 27, 2025
"""

import numpy as np
import cv2

g_image_click = None
g_image_display = None
class SpotPerception:
    """
    Contains computer vision and geometric helper functions for Spot.
    All methods are static, as they don't depend on class state.
    """

    @staticmethod
    def find_strongest_vertical_edge(cv_img, depth_img):
        """
        Finds the strongest vertical edge in the input image.

        Args:
            cv_image (np.ndarray): Input color or grayscale image.
        Returns:
            edge_x (int): The x-coordinate of the strongest vertical edge.
        
        #TODO Update to help the robots know if they are left or right edge
        #TODO Use depth to help figure out what edges are the box
        """
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img.copy()        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=50,minLineLength=50, maxLineGap=10
        )
        best_line = None
        best_score = -np.inf
        h = cv_img.shape[0]
        if lines is not None:
            max_distance_m = 3.0  # or whatever you want
            good_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                depth, px, py = SpotPerception.get_depth_at_pixel(depth_img, mid_x, mid_y, search_radius=20)
                if depth is not None and depth < max_distance_m:
                    # Optionally, you can use (px, py) as your grasp point instead of the exact midpoint
                    good_lines.append((line, px, py, depth))
            
            for line, px, py, depth in good_lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = np.hypot(dx, dy)
                # Check if vertical enough
                if dx < 0.5*dy and length > 30:
                    # Score: longer, and closer to bottom of image (foreground)
                    avg_y = (y1 + y2) / 2
                    score = avg_y + 2 * length  # Heavily favor lines lower in image
                    if score > best_score:
                        best_score = score
                        best_line = (x1, y1, x2, y2, px, py)

        if not best_line:
            print("best_line is empty")
        return best_line

    @staticmethod
    def get_vertical_edge_grasp_point(visual_img, depth_img, id="spot", 
                                      save_img:bool=False):
        """
        Calculates a grasp point on a detected vertical edge.

        Args:
            visual_img (np.ndarray): Visual image (rgb or gray)
            depth_img (np.ndarray): Depth image
        Returns:
            grasp_point (tuple): (x, y) pixel coordinates for grasp.
        """
        # Find the strongest vertical line
        line = SpotPerception.find_strongest_vertical_edge(visual_img, depth_img)
        if not line:
            print("No strong vertical line found.")
            return None
        x1, y1, x2, y2, px, py = line
        
        if save_img:
            SpotPerception.save_markup_img(visual_img, line, id)
        
        return px, py
    @staticmethod
    def save_markup_img(img, line, id, depth=None):
        """
        Save image with edge detection markup.
        """
        img_mark = img.copy()
        x1, y1, x2, y2, px, py = line
        # Calculate line midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        # Draw the detected vertical edge as a green line
        cv2.line(img_mark, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw the grasp pixel (red dot). If valid depth found, use (px,py), else use (cx,cy)
        cv2.circle(img_mark, (px, py), 6, (0, 0, 255), -1)
        
        path = f"images/{id}_edge_and_depth_pixel.png"
        cv2.imwrite(path, img_mark)
        print(f"{id}: Saved edge and grasp visualization to {path}")

    @staticmethod
    def depth_to_point_cloud(depth_img, camera_model, region=None):
        """
        Converts a depth image to a point cloud in camera frame.

        Args:
            depth_img (np.ndarray): Depth image (uint16).
            camera_model: Camera intrinsics/model (from Spot SDK).
            region (tuple, optional): (xmin, xmax, ymin, ymax) crop region.

        Returns:
            points (np.ndarray): Nx3 array of XYZ points.
        """
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

    @staticmethod
    def fit_plane(points):
        """
        Fits a plane to a set of 3D points.

        Args:
            points (np.ndarray): Nx3 array of 3D points.

        Returns:
            plane_normal (np.ndarray): 3-vector for normal.
            plane_point (np.ndarray): A point on the plane.
        """
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

    @staticmethod
    def get_depth_at_pixel(depth_img, x, y, search_radius=5):
        """
        Gets the depth value at a pixel.

        Args:
            depth_img (np.ndarray): Depth image.
            x (int): Pixel x-coordinate.
            y (int): Pixel y-coordinate.
            search_radius (int): Radius in pixels to search for a valid depth.

        Returns:
            depth (float): Depth value at (x, y).
        """
        h, w = depth_img.shape[:2]
        # Try the center pixel first
        if 0 <= x < w and 0 <= y < h:
            d = depth_img[y, x]
            if np.isfinite(d) and d > 0:
                return (float(d) / 1000.0 if depth_img.dtype == np.uint16 else float(d), x, y)
        # Search in a small window for a valid depth
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < w and 0 <= py < h:
                        d = depth_img[py, px]
                        if np.isfinite(d) and d > 0:
                            return (float(d) / 1000.0 if depth_img.dtype == np.uint16 else float(d), px, py)
        return (None, None, None)

    @staticmethod
    def pixel_to_camera_frame(x, y, depth, camera_model):
        """
        Converts pixel coordinates and depth to camera frame XYZ.

        Args:
            x (int): Pixel x-coordinate.
            y (int): Pixel y-coordinate.
            depth (float): Depth at (x, y).
            camera_model: Camera intrinsics/model (from Spot SDK).

        Returns:
            xyz (np.ndarray): 3D point in camera frame.
        """
        # camera_model contains focal_length, center_x, center_y
        fx = camera_model.intrinsics.focal_length.x
        fy = camera_model.intrinsics.focal_length.y
        cx = camera_model.intrinsics.principal_point.x
        cy = camera_model.intrinsics.principal_point.y
        # Unproject
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        z = depth
        return (x, y, z)

    @staticmethod
    def get_target_from_user(img):
        """
        Displays an image from the robot's camera and waits for user to
        click on a target.
        """
        # Show the image to the user and wait for them to click on a pixel
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, SpotPerception.cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_click = None
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        print(f"g_image_click {g_image_click}")
        return g_image_click

    @staticmethod
    def cv_mouse_callback(event, x, y, flags, param):
        global g_image_click, g_image_display
        clone = g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            # print('mouse', x, y)
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)

    @staticmethod
    def display_pixel_selection(img, x, y, wait):
        img_copy = img.copy()
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Target pixel'
        height = img_copy.shape[0]
        width = img_copy.shape[1]
        cv2.line(img_copy, (0, y), (width, y), color, thickness)
        cv2.line(img_copy, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, img_copy)
        if wait:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q'):
                return False

        return True
