"""
path_generator.py

Path generation for real-robot path-following policy execution.

Generates all path types used in simulation (arc, straight, S-curve, meander,
triple-S) in a local coordinate frame, then transforms them into the robot's
vision frame using the saved initial pose.

All paths are returned as np.ndarray of shape [N x 3] (x, y, z) in the
vision frame, with z held constant at the EEF height at episode start.

Date: February 2026
"""

import math
import numpy as np


class PathGenerator:
    """
    Generates 2D paths in local frame and transforms them to the vision frame.

    Usage:
        gen = PathGenerator(saved_yaw=yaw, origin_xy=np.array([x, y]), z_height=z)
        path = gen.arc(radius=1.5, start_angle=0, end_angle=math.pi/3)
        path = gen.s_curve(length=3.0, amplitude=0.5)
        # etc.

    All returned paths are [N x 3] float32 arrays in the vision frame.
    """

    def __init__(
        self,
        saved_yaw: float,
        origin_xy: np.ndarray,
        z_height: float,
    ):
        """
        Args:
            saved_yaw:  Robot yaw at episode start (radians, vision frame).
            origin_xy:  EEF [x, y] position at episode start (vision frame).
            z_height:   EEF z at episode start — held constant across all path points.
        """
        self.saved_yaw = float(saved_yaw)
        self.origin_xy = np.array(origin_xy[:2], dtype=float)
        self.z_height  = float(z_height)

        # Precompute rotation matrix (local → vision)
        c = math.cos(saved_yaw)
        s = math.sin(saved_yaw)
        self._R = np.array([[c, -s],
                             [s,  c]])

    # ------------------------------------------------------------------
    # Public path generators
    # ------------------------------------------------------------------

    def arc(
        self,
        radius: float,
        start_angle: float,
        end_angle: float,
        num_points: int = 50,
    ) -> np.ndarray:
        """
        Circular arc path. Matches simulation's _generate_arc_path() exactly.

        In simulation, the arc is generated as:
            x = radius * cos(theta),  y = radius * sin(theta)
        for theta in [start_angle, end_angle].

        After zero-origining in _to_vision(), the first point is subtracted
        so the path starts at the EEF position. The curvature direction is
        determined by the sign of (end_angle - start_angle):
            positive → curves left (counter-clockwise)
            negative → curves right (clockwise)

        Args:
            radius:      Arc radius (m). Larger = gentler curve.
            start_angle: Start angle of the arc (radians).
            end_angle:   End angle of the arc (radians).
            num_points:  Number of waypoints.

        Returns:
            [num_points x 3] path in vision frame.
        """
        pts_local = np.array([
            [radius * math.cos(t), radius * math.sin(t)]
            for t in np.linspace(start_angle, end_angle, num_points)
        ])
        # Note: _to_vision() subtracts pts_local[0] first, so the arc starts
        # at origin in local frame regardless of start_angle. The shape
        # (curvature) is preserved because all points shift equally.
        return self._to_vision(pts_local)

    def straight(
        self,
        length: float,
        num_points: int = 50,
    ) -> np.ndarray:
        """
        Straight-line path along the robot's forward direction.

        Args:
            length:     Total path length (m).
            num_points: Number of waypoints.

        Returns:
            [num_points x 3] path in vision frame.
        """
        pts_local = np.column_stack([
            np.linspace(0, length, num_points),
            np.zeros(num_points),
        ])
        return self._to_vision(pts_local)

    def s_curve(
        self,
        length: float = 3.0,
        amplitude: float = 0.5,
        num_points: int = 100,
    ) -> np.ndarray:
        """
        S-shaped path: one full sine cycle over `length` metres forward.
        Matches test_generalization.generate_s_path() exactly.

        Args:
            length:     Forward extent of the S (m).
            amplitude:  Lateral amplitude (m).
            num_points: Number of waypoints.

        Returns:
            [num_points x 3] path in vision frame.
        """
        t = np.linspace(0, 2 * math.pi, num_points)
        x = (t / (2 * math.pi)) * length
        y = np.sin(t) * amplitude + np.sin(1.5 * t) * amplitude * 0.2
        pts_local = np.column_stack([x, y])
        return self._to_vision(pts_local)

    def meander(
        self,
        length: float = 4.0,
        amplitude: float = 0.4,
        num_points: int = 150,
    ) -> np.ndarray:
        """
        Meandering path: superposition of two sine frequencies.
        Matches test_generalization.generate_meander_path() exactly.

        Args:
            length:     Forward extent (m).
            amplitude:  Primary lateral amplitude (m).
            num_points: Number of waypoints.

        Returns:
            [num_points x 3] path in vision frame.
        """
        t = np.linspace(0, 4 * math.pi, num_points)
        x = (t / (4 * math.pi)) * length
        y = np.sin(t) * amplitude + np.sin(2.3 * t) * amplitude * 0.4
        pts_local = np.column_stack([x, y])
        return self._to_vision(pts_local)

    def triple_s(
        self,
        length: float = 4.5,
        amplitude: float = 0.4,
        num_points: int = 150,
    ) -> np.ndarray:
        """
        Three consecutive S-curves.
        Matches test_generalization.generate_triple_s_path() exactly.

        Args:
            length:     Forward extent (m).
            amplitude:  Lateral amplitude (m).
            num_points: Number of waypoints.

        Returns:
            [num_points x 3] path in vision frame.
        """
        t = np.linspace(0, 6 * math.pi, num_points)
        x = (t / (6 * math.pi)) * length
        y = np.sin(t) * amplitude
        pts_local = np.column_stack([x, y])
        return self._to_vision(pts_local)

    # ------------------------------------------------------------------
    # Private transform
    # ------------------------------------------------------------------

    def _to_vision(self, pts_local: np.ndarray) -> np.ndarray:
        """
        Transform [N x 2] local-frame points to [N x 3] vision-frame points.

        Steps:
          1. Subtract first point so the path origin is at (0, 0).
          2. Rotate by saved_yaw to align with robot orientation.
          3. Translate to EEF start position in vision frame.
          4. Append constant z_height.
        """
        # Step 1: zero-origin in local frame
        pts = pts_local - pts_local[0]

        # Step 2 & 3: rotate + translate
        pts_vision = (self._R @ pts.T).T + self.origin_xy

        # Step 4: append z
        z_col = np.full((len(pts_vision), 1), self.z_height)
        return np.column_stack([pts_vision, z_col]).astype(np.float32)