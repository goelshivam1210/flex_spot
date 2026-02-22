"""
force_prober.py — estimates norm_reactive_force = μmg / F_max via impedance probing.

Ramps a virtual spring equilibrium forward along a probe axis while monitoring the
wrist F/T sensor at 50 Hz. Breakaway (static → kinetic transition) shows as a force
drop; returns the peak force before the drop normalised by F_max.
"""

import time
import math
import numpy as np

try:
    from bosdyn.api import robot_command_pb2, geometry_pb2, trajectory_pb2
    from bosdyn.client.frame_helpers import get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME
    from bosdyn.client.math_helpers import SE3Pose, Quat
    _SPOT_AVAILABLE = True
except ImportError:
    _SPOT_AVAILABLE = False

_DEFAULTS = {
    ("edge_grasp",   "floor"): 0.35,
    ("handle_grasp", "floor"): 0.30,
    ("edge_grasp",   "mat"):   0.45,
    ("handle_grasp", "mat"):   0.40,
    "fallback": 0.35,
}


class ForceProber:
    def __init__(
        self,
        max_force: float = 250.0,           # F_max for normalisation (N)
        probe_step_m: float = 0.005,        # equilibrium advance per step (m)
        max_probe_steps: int = 30,          # steps before giving up
        settle_time_s: float = 1,         # sample window per step (s)
        force_drop_threshold: float = 4.0,  # N drop that signals breakaway
        impedance_stiffness: float = 250.0, # translational stiffness (N/m)
        impedance_damping: float = 30.0,    # translational damping (Ns/m)
    ):
        self.max_force            = float(max_force)
        self.probe_step_m         = float(probe_step_m)
        self.max_probe_steps      = int(max_probe_steps)
        self.settle_time_s        = float(settle_time_s)
        self.force_drop_threshold = float(force_drop_threshold)
        self.impedance_stiffness  = float(impedance_stiffness)
        self.impedance_damping    = float(impedance_damping)
        self._last_result: float  = None

    @property
    def last_result(self) -> float:
        return self._last_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, spot, probe_axis=None, grasp_strategy="edge_grasp",
              surface_type="floor", robot_side="left") -> float:
        """
        Actively probe and return norm_reactive_force in [0, 1].
        Falls back to a calibrated default on any failure.
        """
        if not _SPOT_AVAILABLE:
            return self._use_default(grasp_strategy, surface_type)

        if probe_axis is None:
            _, _, yaw = spot.get_current_pose()
            yaw += -math.pi / 4 if robot_side == "left" else math.pi / 4
            probe_axis = np.array([math.cos(yaw), math.sin(yaw)])

        probe_axis = np.asarray(probe_axis[:2], dtype=float)
        probe_axis /= np.linalg.norm(probe_axis)

        try:
            f_react = self._impedance_probe(spot, probe_axis)
            self._last_result = float(np.clip(f_react / self.max_force, 0.0, 1.0))
            print(f"[ForceProber] F_react={f_react:.1f} N → norm={self._last_result:.3f}")
            return self._last_result
        except Exception as e:
            print(f"[ForceProber] Probing failed ({e}), using default.")
            return self._use_default(grasp_strategy, surface_type)

    def _use_default(self, grasp_strategy="edge_grasp", surface_type="floor") -> float:
        val = _DEFAULTS.get((grasp_strategy, surface_type), _DEFAULTS["fallback"])
        self._last_result = float(val)
        print(f"[ForceProber] Default: {self._last_result:.3f}")
        return self._last_result

    # ------------------------------------------------------------------
    # Impedance probe
    # ------------------------------------------------------------------

    def _impedance_probe(self, spot, probe_axis: np.ndarray) -> float:
        """
        Ramp spring equilibrium along probe_axis in GRAV_ALIGNED_BODY_FRAME x-y plane.
        Returns peak wrist force (net of baseline) observed before breakaway.
        """
        sc = spot._client._state_client
        cc = spot._client._command_client

        snap        = sc.get_robot_state().kinematic_state.transforms_snapshot
        body_T_hand = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        if body_T_hand is None:
            raise RuntimeError("Cannot get hand pose.")

        k, d = self.impedance_stiffness, self.impedance_damping

        def _send(offset_m: float):
            """Send impedance command with equilibrium offset_m ahead along probe_axis."""
            cmd = robot_command_pb2.RobotCommand()
            imp = cmd.synchronized_command.arm_command.arm_impedance_command
            imp.root_frame_name = GRAV_ALIGNED_BODY_FRAME_NAME
            # Task frame = body frame (identity root_tform_task)
            imp.root_tform_task.CopyFrom(SE3Pose(0, 0, 0, Quat()).to_proto())
            imp.wrist_tform_tool.CopyFrom(SE3Pose(0, 0, 0, Quat()).to_proto())
            # High z stiffness keeps hand at same height; x-y stiffness drives the push
            imp.diagonal_stiffness_matrix.CopyFrom(
                geometry_pb2.Vector(values=[k, k, 500.0, 20.0, 20.0, 20.0])
            )
            imp.diagonal_damping_matrix.CopyFrom(
                geometry_pb2.Vector(values=[d, d, d, 1.0, 1.0, 1.0])
            )
            target = SE3Pose(
                x=body_T_hand.x + offset_m * probe_axis[0],
                y=body_T_hand.y + offset_m * probe_axis[1],
                z=body_T_hand.z,
                rot=body_T_hand.rot,
            )
            pt = trajectory_pb2.SE3TrajectoryPoint()
            pt.pose.CopyFrom(target.to_proto())
            traj = trajectory_pb2.SE3Trajectory()
            traj.points.append(pt)
            imp.task_tform_desired_tool.CopyFrom(traj)
            cc.robot_command(cmd)

        def _read_force() -> float:
            ft = sc.get_robot_state().manipulator_state.estimated_end_effector_force_in_hand
            # Rotate force from hand frame into body frame via quaternion sandwich
            f = np.array([ft.x, ft.y, ft.z])
            q = body_T_hand.rot
            t = 2.0 * np.cross([q.x, q.y, q.z], f)
            f_body = f + q.w * t + np.cross([q.x, q.y, q.z], t)
            # Project onto probe axis — isolates push force, rejects clamping force.
            # Negate: box pushes arm back (Newton's 3rd), so raw projection is negative.
            return -float(probe_axis[0] * f_body[0] + probe_axis[1] * f_body[1])

        def _sample(duration_s: float):
            """Sample F/T at ~50 Hz. Returns (mean, peak) of projected force."""
            samples = []
            t_end = time.time() + duration_s
            while time.time() < t_end:
                try:
                    samples.append(_read_force())
                except Exception:
                    pass
                time.sleep(0.02)
            if not samples:
                raise RuntimeError("F/T sensor unavailable.")
            return float(np.mean(samples)), float(max(samples))

        # Activate impedance at 0 and let it settle before baseline
        print("[ForceProber] Settling...")
        _send(0.0)
        time.sleep(3.0)
        baseline, _ = _sample(self.settle_time_s)
        print(f"[ForceProber] Baseline: {baseline:.1f} N")

        max_peak = 0.0
        steps_below = 0  # consecutive steps with force < max_peak - threshold

        for step in range(1, self.max_probe_steps + 1):
            _send(step * self.probe_step_m)
            _, peak_abs = _sample(self.settle_time_s)
            max_peak = max(max_peak, peak_abs)

            eq_mm = step * self.probe_step_m * 1000
            print(f"[ForceProber] Step {step}: {peak_abs:.1f} N  "
                  f"(max={max_peak:.1f} N, eq={eq_mm:.0f} mm)")

            # Require 2 consecutive steps below max_peak to confirm breakaway
            # (single-step spike followed by a slight drop should not trigger)
            if (max_peak - peak_abs) > self.force_drop_threshold:
                steps_below += 1
                if steps_below >= 2:
                    print(f"[ForceProber] Breakaway confirmed: peak={max_peak:.1f} N  "
                          f"current={peak_abs:.1f} N")
                    return max_peak
            else:
                steps_below = 0

        print(f"[ForceProber] No clean breakaway, peak={max_peak:.1f} N")
        return max_peak
