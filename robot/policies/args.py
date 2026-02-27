"""
Push policy CLI arguments.

Uses tyro for type-safe CLI with automatic --help generation.
Arguments are organized into logical sections via docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class PushArgs:
    """
    Push an object along a path with Spot using a TD3 policy.

    Connection
    ----------
    hostname : Robot IP or hostname
    dock_id : Dock ID for docking at end of run

    Perception
    ----------
    image_source, depth_source : Camera sources
    autonomous_detection : Use SAM for edge detection
    robot_side : left or right

    Box (for box-center state)
    --------------------------
    use_box_center, box_width, box_depth, box_height

    Path
    ----
    path_type, arc_radius, arc_angle, length, amplitude

    Policy
    ------
    model_dir, model_name, push_from_edge

    Force
    -----
    probe_force, surface_type, max_force

    Execution
    ---------
    max_steps, action_scale, yaw_scale, success_distance, log_every, walk_back

    Impedance
    ---------
    use_impedance, impedance_stiffness, impedance_damping, impedance_two_phase
    """

    # --- Connection ---
    hostname: str = "192.168.1.101"
    """Robot IP address or hostname."""
    dock_id: int = 521
    """Dock ID for docking at end of run."""

    # --- Perception ---
    image_source: str = "hand_color_image"
    """Image source for grasp point selection."""
    depth_source: str = "hand_depth_in_hand_color_frame"
    """Depth source for 3D projection."""
    autonomous_detection: bool = False
    """Use SAM for autonomous edge detection; else manual click."""
    robot_side: Literal["left", "right"] = "right"
    """Which side of the robot is used for grasping."""

    # --- Box (for box-center state estimation; matches sim) ---
    use_box_center: bool = False
    """Use box-center state/path; default is EEF pose only."""
    box_width: float = 0.465
    """Box width (m)."""
    box_depth: float = 0.61
    """Box depth (m)."""
    box_height: float = 0.63
    """Box height (m)."""

    # --- Path ---
    path_type: Literal["arc", "straight", "s_curve", "meander", "triple_s"] = "arc"
    """Path shape."""
    arc_radius: float = 1.5
    """Arc radius (m) for arc path."""
    arc_angle: float = 60.0
    """Arc angle (deg) for arc path."""
    length: float = 1.5
    """Path length (m) for straight/s_curve/meander/triple_s."""
    amplitude: float = 0.5
    """Amplitude (m) for s_curve/meander/triple_s."""

    # --- Policy ---
    model_dir: str = "models/push_from_edge"
    """Directory containing TD3 weights."""
    model_name: str = "final_model"
    """Name prefix for saved weights, e.g. 'best' → best_actor.pth."""
    push_from_edge: bool = False
    """Use 2D policy (Fx, Fy only) trained with push_from_edge."""
    state_dim: int = 6
    """State dimension for TD3 (must match training)."""
    action_dim: int = 2
    """Action dimension for center-push TD3 policy."""

    # --- Reactive force ---
    probe_force: bool = True
    """Actively probe for reactive force; else use calibrated default."""
    surface_type: Literal["floor", "mat"] = "floor"
    """Surface type for default force calibration."""
    max_force: float = 200.0
    """F_max for norm_reactive_force (N)."""

    # --- Execution ---
    max_steps: int = 30
    """Maximum policy steps before termination."""
    step_duration_s: float = 2.0
    """Seconds per control step (Δt_ctrl) for push loop."""
    action_scale: float = 0.75
    """Scale for policy action → metres (admittance)."""
    yaw_scale: float = 0.05
    """Scale for yaw correction (rad per unit torque)."""
    success_progress: float = 0.95
    """Fraction of path completed (0–1) required for success."""
    deviation_tolerance: float = 0.7
    """Abort if deviation from path (m) exceeds this tolerance."""
    success_distance: float = 0.8
    """Max deviation (m) at path end to count as success."""
    log_every: int = 1
    """Print detailed diagnostics every N policy steps."""
    walk_back: float = 0.0
    """Walk backwards this many metres after standing (0 = disabled)."""
    settle_between_steps: bool = True
    """Between steps: hold pose with impedance and settle so gripper is against box without smashing."""
    settle_duration_s: float = 3.0
    """Duration (s) of impedance settle between policy steps."""

    # --- Impedance control ---
    use_impedance: bool = True
    """Use arm impedance control instead of base mobility for pushing."""
    impedance_stiffness: float = 300.0
    """Impedance stiffness N/m when use_impedance."""
    impedance_damping: float = 45.0
    """Impedance damping Ns/m when use_impedance."""
    impedance_two_phase: bool = True
    """When use_impedance: arm extends first, then body catches up. Keeps hand in front (avoids gripper drifting under body)."""

    def to_dict(self) -> dict:
        """Convert to dict for save_run_data (config)."""
        return {
            "hostname": self.hostname,
            "dock_id": self.dock_id,
            "image_source": self.image_source,
            "depth_source": self.depth_source,
            "autonomous_detection": self.autonomous_detection,
            "robot_side": self.robot_side,
            "use_box_center": self.use_box_center,
            "box_width": self.box_width,
            "box_depth": self.box_depth,
            "box_height": self.box_height,
            "path_type": self.path_type,
            "arc_radius": self.arc_radius,
            "arc_angle": self.arc_angle,
            "length": self.length,
            "amplitude": self.amplitude,
            "model_dir": self.model_dir,
            "model_name": self.model_name,
            "push_from_edge": self.push_from_edge,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "probe_force": self.probe_force,
            "surface_type": self.surface_type,
            "max_force": self.max_force,
            "max_steps": self.max_steps,
            "step_duration_s": self.step_duration_s,
            "action_scale": self.action_scale,
            "yaw_scale": self.yaw_scale,
            "success_progress": self.success_progress,
            "deviation_tolerance": self.deviation_tolerance,
            "success_distance": self.success_distance,
            "log_every": self.log_every,
            "walk_back": self.walk_back,
            "settle_between_steps": self.settle_between_steps,
            "settle_duration_s": self.settle_duration_s,
            "use_impedance": self.use_impedance,
            "impedance_stiffness": self.impedance_stiffness,
            "impedance_damping": self.impedance_damping,
            "impedance_two_phase": self.impedance_two_phase,
        }


def parse_push_args() -> PushArgs:
    """Parse CLI arguments via tyro."""
    return tyro.cli(PushArgs)


if __name__ == "__main__":
    args = parse_push_args()
    print(args)
