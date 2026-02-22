"""
force_prober.py

Estimates the reactive force feature (norm_reactive_force = μmg / F_max)
by commanding small incremental pushes while grasping the object and
monitoring when it breaks static friction.

The probing procedure:
  1. Command a sequence of small positional displacements along the probe axis.
  2. After each displacement, read the wrist force-torque sensor.
  3. Detect the static-friction breakaway: the step where the measured force
     drops (the object started moving) or the object position changes.
  4. The breakaway force magnitude is F_react ≈ μmg.
  5. Return norm_reactive_force = F_react / F_max.

If the Spot SDK force-torque API is unavailable or probing fails for any
reason, a safe calibrated default is returned instead.

Date: February 2026
"""

import time
import numpy as np

# Spot SDK imports — guarded so the module can be imported in simulation too
try:
    from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
    from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
    from bosdyn.client.math_helpers import SE3Pose
    _SPOT_AVAILABLE = True
except ImportError:
    _SPOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default reactive force calibration table.
# Used when probing is skipped or fails.
# Keys are (grasp_strategy, surface_type), values are norm_reactive_force.
# Tune these from offline experiments.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    ("edge_grasp",   "floor"): 0.35,
    ("handle_grasp", "floor"): 0.30,
    ("edge_grasp",   "mat"):   0.45,
    ("handle_grasp", "mat"):   0.40,
    "fallback": 0.35,
}


class ForceProber:
    """
    Estimates norm_reactive_force = μmg / F_max for the current episode.

    Two modes:
      - probe()  : Active estimation via EEF force-torque sensing on the robot.
      - default() : Returns a calibrated fallback value without touching the robot.

    The result from either method can be passed directly to StateEstimator.
    """

    def __init__(
        self,
        max_force: float = 400.0,
        probe_step_m: float = 0.005,
        max_probe_steps: int = 20,
        settle_time_s: float = 0.3,
        force_drop_threshold: float = 5.0,
        position_move_threshold_m: float = 0.005,
    ):
        """
        Args:
            max_force:                 F_max used to normalise the result (N).
            probe_step_m:              EEF displacement per probe step (m).
                                       Small enough to not disturb the object much.
            max_probe_steps:           Maximum number of probe steps before giving up.
            settle_time_s:             Wait time after each step before reading sensor (s).
            force_drop_threshold:      Force drop (N) that signals breakaway.
            position_move_threshold_m: Object movement (m) that signals breakaway.
        """
        self.max_force                = float(max_force)
        self.probe_step_m             = float(probe_step_m)
        self.max_probe_steps          = int(max_probe_steps)
        self.settle_time_s            = float(settle_time_s)
        self.force_drop_threshold     = float(force_drop_threshold)
        self.position_move_threshold_m = float(position_move_threshold_m)

        self._last_result: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> float | None:
        """The norm_reactive_force estimated in the most recent probe() call."""
        return self._last_result

    def default(
        self,
        grasp_strategy: str = "edge_grasp",
        surface_type: str = "floor",
    ) -> float:
        """
        Return a calibrated default norm_reactive_force without robot interaction.

        Args:
            grasp_strategy: "edge_grasp" or "handle_grasp".
            surface_type:   "floor" or "mat".

        Returns:
            float in [0, 1].
        """
        key = (grasp_strategy, surface_type)
        value = _DEFAULTS.get(key, _DEFAULTS["fallback"])
        self._last_result = float(value)
        print(f"[ForceProber] Using calibrated default: {self._last_result:.3f} "
              f"(strategy={grasp_strategy}, surface={surface_type})")
        return self._last_result

    def probe(
        self,
        spot,
        probe_axis: np.ndarray = None,
        grasp_strategy: str = "edge_grasp",
        surface_type: str = "floor",
    ) -> float:
        """
        Actively estimate norm_reactive_force by probing on the robot.

        The robot commands a sequence of small EEF displacements along
        probe_axis while holding the object. When the wrist force sensor
        detects breakaway (force drop or object movement), the applied
        force at that step is used to estimate F_react.

        Falls back to default() if:
          - Spot SDK is not available.
          - Force-torque sensor is unavailable.
          - No breakaway detected within max_probe_steps.
          - Any exception occurs.

        Args:
            spot:           Spot robot instance (spot.py Spot class).
            probe_axis:     Unit vector [x, y] in vision frame to push along.
                            Defaults to robot forward direction.
            grasp_strategy: Used for fallback default lookup.
            surface_type:   Used for fallback default lookup.

        Returns:
            float in [0, 1] — norm_reactive_force for this episode.
        """
        if not _SPOT_AVAILABLE:
            print("[ForceProber] Spot SDK not available, using default.")
            return self.default(grasp_strategy, surface_type)

        if probe_axis is None:
            # Default: push along robot forward direction in vision frame
            import math
            _, _, yaw = spot.get_current_pose()
            probe_axis = np.array([math.cos(yaw), math.sin(yaw)])

        probe_axis = np.array(probe_axis[:2], dtype=float)
        axis_norm = np.linalg.norm(probe_axis)
        if axis_norm < 1e-8:
            print("[ForceProber] probe_axis is zero vector, using robot forward direction.")
            import math
            _, _, yaw = spot.get_current_pose()
            probe_axis = np.array([math.cos(yaw), math.sin(yaw)])
        else:
            probe_axis /= axis_norm

        try:
            result = self._run_probe(spot, probe_axis)
            if result is None:
                print("[ForceProber] Breakaway not detected, using default.")
                return self.default(grasp_strategy, surface_type)

            self._last_result = float(np.clip(result / self.max_force, 0.0, 1.0))
            print(f"[ForceProber] Probed F_react={result:.1f}N → "
                  f"norm_reactive_force={self._last_result:.3f}")
            return self._last_result

        except Exception as e:
            print(f"[ForceProber] Probing failed ({e}), using default.")
            return self.default(grasp_strategy, surface_type)

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _run_probe(self, spot, probe_axis: np.ndarray) -> float | None:
        """
        Execute incremental probe steps and detect breakaway.

        Returns the estimated breakaway force in Newtons, or None if not found.
        """
        state_client  = spot._client._state_client
        command_client = spot._client._command_client

        # Get initial hand pose in vision frame
        snapshot      = state_client.get_robot_state().kinematic_state.transforms_snapshot
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        if vision_T_hand is None:
            raise RuntimeError("Cannot get hand pose for probing.")

        start_hand_pos = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
        current_target = start_hand_pos.copy()

        prev_force_mag = None

        for step in range(1, self.max_probe_steps + 1):
            # --- Command small displacement along probe axis ---
            current_target[0] += self.probe_step_m * probe_axis[0]
            current_target[1] += self.probe_step_m * probe_axis[1]

            target_pose = SE3Pose(
                x=current_target[0],
                y=current_target[1],
                z=current_target[2],
                rot=vision_T_hand.rot,
            )
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                target_pose.to_proto(), VISION_FRAME_NAME, seconds=1.0
            )
            cmd_id = command_client.robot_command(arm_cmd)
            block_until_arm_arrives(command_client, cmd_id, timeout_sec=2.0)

            # Let the system settle before reading
            time.sleep(self.settle_time_s)

            # --- Read wrist force-torque sensor ---
            force_mag = self._read_wrist_force(state_client)
            if force_mag is None:
                # Sensor unavailable — fall back
                raise RuntimeError("Force-torque sensor unavailable.")

            # --- Applied force estimate (NEEDS CALIBRATION) ---
            #
            # What we want: the actual force being transmitted through the gripper
            # to the object at this probe step.
            #
            # What we have: the wrist F/T sensor reading (force_mag), which measures
            # the reaction force at the wrist joint — this includes both the force
            # on the object AND the arm's own inertia/weight components.
            #
            # Current approach: we use force_mag directly as the F_react estimate,
            # under the assumption that:
            #   (a) the arm is quasi-static (slow probe steps, so inertia is small)
            #   (b) gravity compensation is handled by the Spot SDK internally
            #   (c) therefore force_mag ≈ contact force on object ≈ μmg at breakaway
            #
            # This is a rough approximation. In practice you may need to:
            #   1. Subtract a baseline: read force_mag BEFORE contact and subtract it
            #      to remove gravity/arm-weight bias.
            #   2. Scale by a calibration factor: run known-mass objects and compare
            #      force_mag at breakaway to the known μmg value.
            #   3. Project onto the probe axis: use the raw [fx, fy, fz] vector and
            #      dot it with probe_axis rather than taking the full magnitude, to
            #      isolate the contact force component from side-load noise.
            #
            # The applied_force_estimate variable below is an alternative linear ramp
            # model (stiffness * displacement). It is NOT currently used in breakaway
            # detection — we kept it here as a reference for future calibration work.
            # Once you have real data, you can compare force_mag vs applied_force_estimate
            # per step to back out the effective contact stiffness of the setup.
            #
            # TO CALIBRATE:
            #   1. Run probe() on objects of known mass m and friction μ.
            #   2. Record force_mag at the breakaway step.
            #   3. Compute calibration_factor = (μ * m * 9.81) / force_mag_at_breakaway
            #   4. Apply: F_react = force_mag * calibration_factor
            #   5. Update _DEFAULTS table with values from this procedure.
            applied_force_estimate = step * self.probe_step_m * self.max_force / 0.1
            # ^ linear ramp model: assumes F_max would be reached over 0.1m of displacement.
            # 0.1m is a placeholder — replace with the measured compliance of your setup.
            # Not used in detection logic currently; kept for calibration reference only.

            # --- Detect breakaway ---
            # Condition 1: force DROP between consecutive steps.
            #
            # Physics: as we incrementally push, the F/T sensor reading ramps up
            # while the object is stationary (static friction building). At breakaway,
            # static friction is overcome and the object starts sliding. Kinetic
            # friction is lower than static, so the measured resistance force drops.
            # The PEAK force (prev_force_mag, the step before the drop) is our best
            # estimate of F_react = μ_static * m * g.
            # We return prev_force_mag, not force_mag, for this reason.
            if prev_force_mag is not None and (prev_force_mag - force_mag) > self.force_drop_threshold:
                print(f"[ForceProber] Breakaway detected at step {step} "
                      f"(force drop {prev_force_mag:.1f} → {force_mag:.1f} N)")
                return prev_force_mag  # peak force before drop ≈ μ_static * m * g

            # Condition 2: hand REACHED commanded position but force is still high.
            #
            # Physics: if the hand arrives at the commanded target (small slip between
            # actual and commanded position), but force_mag is still substantial, it
            # means the object moved with the hand — breakaway already happened earlier
            # and we missed the force drop (e.g. due to noisy sensor or coarse steps).
            # In this case, use the current force_mag as the best available estimate.
            #
            # NOTE: This is the OPPOSITE of "the hand didn't reach commanded pos".
            # If the hand DIDN'T reach commanded pos (large slip), that means the
            # object is resisting and we are still in the pre-breakaway ramp — do NOT
            # trigger breakaway detection in that case.
            snapshot_now = state_client.get_robot_state().kinematic_state.transforms_snapshot
            hand_now     = get_a_tform_b(snapshot_now, VISION_FRAME_NAME, "hand")
            if hand_now is not None:
                actual_pos    = np.array([hand_now.x, hand_now.y])
                commanded_pos = current_target[:2]
                # slip = how far the hand fell short of the commanded target
                slip = np.linalg.norm(actual_pos - commanded_pos)
                # Small slip = hand arrived → object may have moved with it
                if slip < self.position_move_threshold_m and force_mag > self.force_drop_threshold:
                    print(f"[ForceProber] Object moved with hand at step {step} "
                          f"(slip={slip:.4f} m, force={force_mag:.1f} N)")
                    return force_mag

            prev_force_mag = force_mag
            print(f"[ForceProber] Step {step}: force={force_mag:.1f} N, no breakaway yet")

        # No breakaway found
        return None

    @staticmethod
    def _read_wrist_force(state_client) -> float | None:
        """
        Read the wrist force-torque sensor magnitude from robot state.

        Returns force magnitude in Newtons, or None if unavailable.
        """
        try:
            robot_state = state_client.get_robot_state()
            # Spot SDK: manipulator_state.estimated_end_effector_force_in_hand
            ft = robot_state.manipulator_state.estimated_end_effector_force_in_hand
            force_vec = np.array([ft.x, ft.y, ft.z])
            return float(np.linalg.norm(force_vec))
        except Exception:
            return None