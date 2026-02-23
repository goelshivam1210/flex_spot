"""
state_estimator.py

Stateful 6D state estimator for real-robot path-following policy execution.

Mirrors the state representation from the simulation environment exactly:
    s = [lateral_error, orientation_error, speed_fwd, speed_lat, angular_vel, norm_reactive_force]

All quantities are derived from the end-effector (EEF) pose, which is the
only sensor input available after contact is established.

Date: February 2026
"""

import math
import numpy as np


class StateEstimator:
    """
    Stateful 6D state estimator for path-following policy.

    Initialized once per episode with the grasp pose and a pre-estimated
    reactive force feature. Called every control step with the current
    EEF pose to produce the policy input state.

    State vector (matches simulation exactly):
        [0] lateral_error       — signed perpendicular distance from EEF to path (m)
        [1] orientation_error   — continuous angular error between EEF heading and path tangent (rad)
        [2] speed_fwd           — EEF speed projected onto path tangent (m/s)
        [3] speed_lat           — EEF speed projected onto path normal (m/s)
        [4] angular_vel         — yaw rate of EEF (rad/s)
        [5] norm_reactive_force — μmg / F_max, constant for episode (dimensionless)
    """

    def __init__(
        self,
        init_hand_pos: np.ndarray,
        init_yaw: float,
        init_time: float,
        norm_reactive_force: float,
        max_force: float = 400.0,
    ):
        """
        Args:
            init_hand_pos:       EEF position at episode start [x, y, z] in vision frame.
            init_yaw:            EEF yaw angle at episode start (radians).
            init_time:           Wall-clock timestamp at episode start (seconds).
            norm_reactive_force: Pre-estimated μmg/F_max for this episode.
                                 Use ForceProber result or a calibrated default.
            max_force:           Maximum robot force (N). Used only for reference / logging.
        """
        self.norm_reactive_force = float(np.clip(norm_reactive_force, 0.0, 1.0))
        self.max_force = max_force

        # Previous-step state for finite-difference velocity estimation
        self._prev_pos  = np.array(init_hand_pos[:2], dtype=float)  # 2D only
        self._prev_yaw  = float(init_yaw)
        self._prev_time = float(init_time)

        # Closest path index — updated every step, exposed for external use
        self.closest_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        init_hand_pos: np.ndarray,
        init_yaw: float,
        init_time: float,
        norm_reactive_force: float,
    ):
        """
        Re-initialise for a new episode without constructing a new object.
        Useful when running multiple episodes in sequence.
        """
        self.norm_reactive_force = float(np.clip(norm_reactive_force, 0.0, 1.0))
        self._prev_pos  = np.array(init_hand_pos[:2], dtype=float)
        self._prev_yaw  = float(init_yaw)
        self._prev_time = float(init_time)
        self.closest_idx = 0

    def compute(
        self,
        current_hand_pos: np.ndarray,
        current_yaw: float,
        current_time: float,
        path_points: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the 6D state vector from current EEF observations.

        Args:
            current_hand_pos: Current EEF position [x, y, z] or [x, y] in vision frame.
            current_yaw:      Current EEF yaw angle (radians).
            current_time:     Current wall-clock timestamp (seconds).
            path_points:      Array of 2D or 3D path waypoints [N x 2] or [N x 3].
                              Only the first two columns (x, y) are used.

        Returns:
            np.ndarray of shape (6,) with dtype float32.
        """
        pos_2d = np.array(current_hand_pos[:2], dtype=float)
        pts_2d = np.array(path_points)[:, :2]  # drop z if present

        # --- 1. Closest path point (full search, same as sim) ---
        dists = np.linalg.norm(pts_2d - pos_2d, axis=1)
        self.closest_idx = int(np.argmin(dists))

        # --- 2. Path tangent and normal at closest point ---
        tangent, normal = self._path_tangent_normal(pts_2d, self.closest_idx)

        # --- 3. Geometric errors ---
        lateral_error     = self._lateral_error(pos_2d, pts_2d[self.closest_idx], normal)
        orientation_error = self._orientation_error(current_yaw, tangent)

        # --- 4. Velocities via finite difference over control interval ---
        dt = current_time - self._prev_time
        if dt > 1e-6:
            vel_2d    = (pos_2d - self._prev_pos) / dt
            speed_fwd = float(np.dot(vel_2d, tangent))
            speed_lat = float(np.dot(vel_2d, normal))
            angular_vel = _wrap_to_pi(current_yaw - self._prev_yaw) / dt
        else:
            # First step or clock stall — zero velocities, same as sim reset
            speed_fwd   = 0.0
            speed_lat   = 0.0
            angular_vel = 0.0

        # --- 5. Update history ---
        self._prev_pos  = pos_2d.copy()
        self._prev_yaw  = float(current_yaw)
        self._prev_time = float(current_time)

        state = np.array(
            [
                lateral_error,
                orientation_error,
                speed_fwd,
                speed_lat,
                angular_vel,
                self.norm_reactive_force,
            ],
            dtype=np.float32,
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _path_tangent_normal(pts_2d: np.ndarray, idx: int):
        """
        Compute unit tangent and left-pointing normal at path index idx.
        Mirrors the tangent singularity fix in env._get_state().
        """
        n = len(pts_2d)
        if idx == n - 1 and n > 1:
            # At last point: backward difference
            raw = pts_2d[idx] - pts_2d[idx - 1]
        else:
            next_idx = min(idx + 1, n - 1)
            raw = pts_2d[next_idx] - pts_2d[idx]

        norm = np.linalg.norm(raw)
        tangent = raw / norm if norm > 1e-8 else np.array([1.0, 0.0])
        normal  = np.array([-tangent[1], tangent[0]])  # left-perpendicular
        return tangent, normal

    @staticmethod
    def _lateral_error(pos_2d, closest_pt, normal):
        """Signed lateral error: positive = left of path."""
        return float(np.dot(pos_2d - closest_pt, normal))

    @staticmethod
    def _orientation_error(yaw: float, tangent: np.ndarray) -> float:
        """
        Continuous orientation error in (-π, π].
        Matches sim: atan2(sin(θ - θ_path), cos(θ - θ_path))
        """
        desired = math.atan2(tangent[1], tangent[0])
        return math.atan2(math.sin(yaw - desired), math.cos(yaw - desired))


# ------------------------------------------------------------------
# Module-level utility
# ------------------------------------------------------------------

def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi