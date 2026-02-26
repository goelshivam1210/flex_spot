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
import matplotlib.pyplot as plt


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
        start_angle: float = math.pi,
        end_angle: float = 4*math.pi/3,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone tester for PathGenerator (vision-frame paths).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pose / frame (simple defaults for quick testing)
    parser.add_argument(
        "--origin-x", type=float, default=0.0,
        help="EEF x position at path start in vision frame (m).",
    )
    parser.add_argument(
        "--origin-y", type=float, default=0.0,
        help="EEF y position at path start in vision frame (m).",
    )
    parser.add_argument(
        "--z-height", type=float, default=0.5,
        help="EEF z height used for all waypoints (m).",
    )
    parser.add_argument(
        "--yaw-deg", type=float, default=0.0,
        help="Robot yaw at path start, in degrees (vision frame).",
    )

    # Path selection
    parser.add_argument(
        "--path-type",
        choices=["arc", "straight", "s_curve", "meander", "triple_s"],
        default="arc",
        help="Which path primitive to generate.",
    )

    # Arc-specific arguments (more user-friendly)
    parser.add_argument(
        "--arc-radius", type=float, default=1.5,
        help="Radius of the arc in metres.",
    )
    parser.add_argument(
        "--arc-angle-deg", type=float, default=60.0,
        help=(
            "Total sweep of the arc in degrees. "
            "Positive = curve left (CCW), negative = curve right (CW)."
        ),
    )
    parser.add_argument(
        "--arc-num-points", type=int, default=50,
        help="Number of waypoints to sample along the arc.",
    )

    # Shared parameters for the other paths
    parser.add_argument(
        "--length", type=float, default=3.0,
        help="Forward length for straight / S / meander / triple-S paths (m).",
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.5,
        help="Lateral amplitude for S / meander / triple-S paths (m).",
    )
    parser.add_argument(
        "--num-points", type=int, default=100,
        help="Number of waypoints for non-arc paths.",
    )

    parser.add_argument(
        "--save-csv", type=str, default="path_preview.csv",
        help="Optional CSV file to save the generated path (x,y,z,idx).",
    )

    args = parser.parse_args()

    origin_xy = np.array([args.origin_x, args.origin_y], dtype=float)
    yaw_rad = math.radians(args.yaw_deg)

    gen = PathGenerator(saved_yaw=yaw_rad, origin_xy=origin_xy, z_height=args.z_height)

    if args.path_type == "arc":
        # Choose arc parameterisation so that, for yaw_deg ≈ 0,
        # the net motion is primarily in +X. We start behind the origin
        # (π radians) and sweep by arc_angle_deg.
        start_angle = math.pi
        end_angle = math.pi + math.radians(args.arc_angle_deg)
        path = gen.arc(
            radius=args.arc_radius,
            start_angle=start_angle,
            end_angle=end_angle,
            num_points=args.arc_num_points,
        )
    elif args.path_type == "straight":
        path = gen.straight(length=args.length, num_points=args.num_points)
    elif args.path_type == "s_curve":
        path = gen.s_curve(length=args.length, amplitude=args.amplitude, num_points=args.num_points)
    elif args.path_type == "meander":
        path = gen.meander(length=args.length, amplitude=args.amplitude, num_points=args.num_points)
    elif args.path_type == "triple_s":
        path = gen.triple_s(length=args.length, amplitude=args.amplitude, num_points=args.num_points)
    else:
        raise ValueError(f"Unknown path type: {args.path_type}")

    # Basic summary
    pts_2d = path[:, :2]
    deltas = np.diff(pts_2d, axis=0)
    total_len = float(np.sum(np.linalg.norm(deltas, axis=1))) if len(deltas) > 0 else 0.0

    print(f"[path_generator] type={args.path_type}")
    print(f"[path_generator] origin=({args.origin_x:.3f}, {args.origin_y:.3f}), yaw={args.yaw_deg:.1f}deg, z={args.z_height:.3f}m")
    print(f"[path_generator] waypoints={len(path)} total_length≈{total_len:.3f}m")
    print(f"[path_generator] first=({path[0,0]:.3f}, {path[0,1]:.3f}, {path[0,2]:.3f})")
    print(f"[path_generator] last =({path[-1,0]:.3f}, {path[-1,1]:.3f}, {path[-1,2]:.3f})")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "x", "y", "z"])
            for i, pt in enumerate(path):
                w.writerow([i, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
        print(f"[path_generator] Saved path to {args.save_csv}")

    # Quick 2D plot for visual inspection
    x, y = path[:, 0], path[:, 1]
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, "b-", lw=2, label="Path")
    plt.scatter([x[0]], [y[0]], color="green", s=60, label="Start")
    plt.scatter([x[-1]], [y[-1]], color="red", s=60, label="End")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Path preview — {args.path_type}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()