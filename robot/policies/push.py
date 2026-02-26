"""
push.py

Push manipulation with Spot robot using the path-following TD3 policy.

Pipeline:
    1. Initialise robot and save starting yaw.
    2. Detect and grasp the object (manual pixel selection or autonomous).
    3. Estimate reactive force (probe or calibrated default).
    4. Define path (arc, straight, s-curve, meander, or triple-s).
    5. Execute path-following policy loop — push only.
    6. Release, stow arm, dock.

The policy receives the 6D egocentric state produced by react.StateEstimator,
which exactly mirrors the simulation training environment. Actions are 3D
normalised wrenches [Fx, Fy, tau] that are scaled and executed as robot
body displacements via Spot.push_object_from_sim().

Usage:
    python push.py --hostname 192.168.1.100 --path-type arc
    python push.py --hostname 192.168.1.100 --path-type straight --length 1.5
    python push.py --hostname 192.168.1.100 --path-type arc --arc-radius 1.5 \\
                   --arc-angle 60 --autonomous-detection
    python push.py --hostname 192.168.1.100 --path-type s_curve \\
                   --length 3.0 --amplitude 0.5 --probe-force

Date: February 2026
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

# Spot SDK
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.robot_command import (
  RobotCommandBuilder,
  blocking_stand,
  block_until_arm_arrives
)

# Local robot modules
from spot.spot import Spot, SpotPerception

# react/ execution layer
from react.state_estimator import StateEstimator, _estimate_box_center_from_grasp
from react.path_generator  import PathGenerator
from react.force_prober    import ForceProber

# TD3 policy
from react.td3 import TD3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM  = 6
ACTION_DIM = 3

# Admittance gains: scale policy action [−1,1] → metres / radians
DEFAULT_ACTION_SCALE = 0.03   # metres per unit normalised force
DEFAULT_YAW_SCALE    = 0.05   # radians per unit normalised torque

# Robot execution timing
STEP_DURATION_S = 3.0   # seconds per control step (matches sim Δt_ctrl)

# Termination thresholds
SUCCESS_PROGRESS    = 0.95   # fraction of path completed
DEVIATION_TOLERANCE = 0.5   # metres — abort if EEF drifts beyond this


# ---------------------------------------------------------------------------
# Lightweight console logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(msg: str) -> None:
    print(f"[push][{_ts()}] {msg}")


def _fmt_xy(xy, prec: int = 3) -> str:
    return f"({float(xy[0]):+.{prec}f}, {float(xy[1]):+.{prec}f})"


def _fmt_vec(v, prec: int = 3) -> str:
    v = np.asarray(v).reshape(-1)
    return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in v) + "]"


def _deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-12 else float(default)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(model_dir: str, model_name: str,
                max_action: float = 1.0, max_torque: float = 1.0,
                action_dim: int = None) -> TD3:
    """
    Load a trained TD3 actor from models/react/<model_name>_actor.pth.

    Args:
        model_dir:  Path to directory containing saved weights, e.g. 'models/react'.
        model_name: Name prefix used when saving, e.g. 'best' → loads 'best_actor.pth'.
        max_action: Max force magnitude (passed to Actor constructor — must match training).
        max_torque: Max torque magnitude (passed to Actor constructor — must match training).
        action_dim: 2 for push-from-edge (Fx, Fy only), 3 for center-push (Fx, Fy, τz). Default: ACTION_DIM.

    Returns:
        TD3 instance with actor loaded and set to eval mode.
    """
    adim = action_dim if action_dim is not None else ACTION_DIM
    policy = TD3(
        lr=1e-3,              # lr irrelevant at inference — optimizer not used
        state_dim=STATE_DIM,
        action_dim=adim,
        max_action=max_action,
        max_torque=max_torque,
    )
    policy.load_actor(model_dir, model_name)
    policy.actor.eval()
    policy.actor_target.eval()
    print(f"[push] Policy loaded: {model_dir}/{model_name}_actor.pth")
    return policy


# ---------------------------------------------------------------------------
# EEF pose helper
# ---------------------------------------------------------------------------

def get_eef_pose(spot):
    """
    Query current EEF (hand) pose from Spot SDK.

    Returns:
        hand_pos: np.ndarray [x, y, z] in vision frame.
        yaw:      float, EEF yaw angle in vision frame (radians).
    """
    state    = spot._client._state_client.get_robot_state()
    snapshot = state.kinematic_state.transforms_snapshot
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")

    if vision_T_hand is None:
        raise RuntimeError("Could not get hand pose from Spot SDK.")

    hand_pos = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])

    q = vision_T_hand.rot
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return hand_pos, yaw


# ---------------------------------------------------------------------------
# Path progress helper
# ---------------------------------------------------------------------------

def compute_progress(hand_pos, path_points):
    """
    Return (progress, deviation, idx) for the current hand position.

    progress:  float in [0, 1] — fraction of path completed.
    deviation: float — Euclidean distance from EEF to closest path point (m).
    idx:       int   — index of closest path point.
    """
    pts_2d    = path_points[:, :2]
    pos_2d    = hand_pos[:2]
    dists     = np.linalg.norm(pts_2d - pos_2d, axis=1)
    idx       = int(np.argmin(dists))
    progress  = idx / (len(path_points) - 1) if len(path_points) > 1 else 0.0
    deviation = float(dists[idx])
    return progress, deviation, idx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(robot_positions, path_points, run_dir, experiment_name):
    positions = np.array(robot_positions)
    rx, ry, ryaw = positions[:, 0], positions[:, 1], positions[:, 2]

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Trajectory vs path ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(path_points[:, 0], path_points[:, 1], "b-", lw=2,
            label="Planned path", alpha=0.7)
    ax.scatter(*path_points[0, :2],  color="green", s=120, zorder=5, label="Path start")
    ax.scatter(*path_points[-1, :2], color="red",   s=120, marker="s", zorder=5, label="Path end")
    ax.plot(rx, ry, "r-", lw=2, label="EEF trajectory", alpha=0.8)
    ax.scatter(rx[0],  ry[0],  color="darkgreen", s=100, marker="^", zorder=5, label="EEF start")
    ax.scatter(rx[-1], ry[-1], color="darkred",   s=100, marker="v", zorder=5, label="EEF end")

    step = max(1, len(rx) // 10)
    for i in range(0, len(rx), step):
        ax.arrow(rx[i], ry[i],
                 0.08 * math.cos(ryaw[i]), 0.08 * math.sin(ryaw[i]),
                 head_width=0.03, head_length=0.02,
                 fc="orange", ec="orange", alpha=0.7)

    total_dist = float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))
    path_len   = float(np.sum(np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)))
    final_err  = float(np.linalg.norm(positions[-1, :2] - path_points[-1, :2]))
    ax.text(0.02, 0.98,
            f"EEF distance: {total_dist:.2f}m\n"
            f"Path length:  {path_len:.2f}m\n"
            f"Final error:  {final_err:.3f}m\n"
            f"Steps:        {len(rx)}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"EEF Trajectory vs Planned Path — {experiment_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(plots_dir, "trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Position + yaw over time ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    steps = np.arange(len(rx))
    axes[0].plot(steps, rx, "b-", lw=2); axes[0].set_ylabel("X (m)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, ry, "g-", lw=2); axes[1].set_ylabel("Y (m)"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, np.degrees(ryaw), "r-", lw=2)
    axes[2].set_ylabel("Yaw (deg)"); axes[2].set_xlabel("Step"); axes[2].grid(True, alpha=0.3)
    axes[0].set_title(f"EEF Position Over Time — {experiment_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "position_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[push] Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------

def save_run_data(run_dir, config, robot_positions, path_points,
                  state_log, action_log, duration_s, success):
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump({**config, "duration_s": duration_s,
                   "success": success, "steps": len(robot_positions)}, f, indent=2)

    with open(os.path.join(run_dir, "eef_trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "yaw"])
        for i, (x, y, yaw) in enumerate(robot_positions):
            w.writerow([i, x, y, yaw])

    with open(os.path.join(run_dir, "planned_path.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x", "y", "z"])
        for i, pt in enumerate(path_points):
            w.writerow([i, pt[0], pt[1], pt[2]])

    with open(os.path.join(run_dir, "state_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lateral_err", "orientation_err",
                    "speed_fwd", "speed_lat", "angular_vel", "norm_rf"])
        for i, s in enumerate(state_log):
            w.writerow([i] + [f"{v:.6f}" for v in s])

    with open(os.path.join(run_dir, "action_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "Fx", "Fy", "tau"])
        for i, a in enumerate(action_log):
            w.writerow([i] + [f"{v:.6f}" for v in a])

    print(f"[push] Run data saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Core push execution
# ---------------------------------------------------------------------------

def run_push(spot, policy, path_points, norm_reactive_force, args):
    """
    Execute the path-following push policy loop.

    Returns:
        success, robot_positions, state_log, action_log, duration_s
    """
    hand_pos, hand_yaw = get_eef_pose(spot)
    init_time = time.time()

    use_box_center = getattr(args, "use_box_center", False)
    box_dims = None if not use_box_center else {
        "width": args.box_width,
        "depth": args.box_depth,
        "height": args.box_height,
    }
    estimator = StateEstimator(
        init_hand_pos=hand_pos,
        init_yaw=hand_yaw,
        init_time=init_time,
        norm_reactive_force=norm_reactive_force,
        grasp_strategy="edge_grasp",
        box_dimensions=box_dims,
        robot_side=getattr(args, "robot_side", "right"),
    )

    robot_positions, state_log, action_log = [], [], []
    success = False

    path_len = float(np.sum(np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)))
    log_every = max(1, int(getattr(args, "log_every", 1)))
    ref_mode = "box_center" if use_box_center else "eef"

    print("")  # spacing for readability in the terminal
    _log(f"Starting policy loop: max_steps={args.max_steps} step_dt={STEP_DURATION_S:.2f}s log_every={log_every}")
    _log(f"Path: waypoints={len(path_points)} length≈{path_len:.2f}m start={_fmt_xy(path_points[0, :2])} end={_fmt_xy(path_points[-1, :2])}")
    _log(
        "Termination: "
        f"success(progress>={SUCCESS_PROGRESS:.2f} & dev<{args.success_distance:.2f}m), "
        f"abort(dev>{DEVIATION_TOLERANCE:.2f}m)"
    )
    _log(
        f"Reference mode={ref_mode} robot_side={args.robot_side} "
        f"push_from_edge={getattr(args, 'push_from_edge', False)} "
        f"scales: action_scale={args.action_scale:.4f}m yaw_scale={args.yaw_scale:.4f}rad "
        f"norm_reactive_force={float(norm_reactive_force):.3f}"
    )
    _log(f"Start EEF: pos={_fmt_vec(hand_pos, prec=3)} yaw={hand_yaw:+.3f}rad ({_deg(hand_yaw):+.1f}deg)")

    start_time = time.time()
    prev_idx = None
    prev_progress = None

    for step in range(args.max_steps):

        # 1. Observe
        hand_pos, hand_yaw = get_eef_pose(spot)
        current_time = time.time()
        robot_positions.append((hand_pos[0], hand_pos[1], hand_yaw))

        # 2. Compute state
        state = estimator.compute(hand_pos, hand_yaw, current_time, path_points)
        state_log.append(state.copy())

        # 3. Check termination (use box center for progress/deviation when box_dimensions set)
        ref_pos = estimator.reference_pos_2d
        progress, deviation, idx = compute_progress(
            np.array([ref_pos[0], ref_pos[1], hand_pos[2]]), path_points
        )
        progress_eef, deviation_eef, idx_eef = compute_progress(hand_pos, path_points)

        if deviation > DEVIATION_TOLERANCE:
            _log(f"ABORT step {step+1}: deviation(ref)={deviation:.3f}m > {DEVIATION_TOLERANCE:.3f}m")
            break

        if progress >= SUCCESS_PROGRESS and deviation < args.success_distance:
            _log(f"SUCCESS step {step+1}: progress(ref)={progress:.3f} deviation(ref)={deviation:.3f}m")
            success = True
            break

        # 4. Query policy
        # policy.select_action returns shape (1, action_dim) — flatten
        raw_action = policy.select_action(state).flatten()
        action = np.clip(raw_action, -1.0, 1.0)
        clipped = bool(np.any(np.abs(raw_action) > 1.0 + 1e-6))
        action_log.append(action.copy())

        # 5. Scale to physical displacements
        # Edge-push policy (2D): only Fx, Fy; d_yaw=0 (torque from r×F on robot)
        dx    = float(action[0]) * args.action_scale
        dy    = float(action[1]) * args.action_scale
        d_yaw = float(action[2]) * args.yaw_scale if len(action) >= 3 else 0.0

        vx    = max(abs(dx)    / STEP_DURATION_S, 0.01)
        vy    = max(abs(dy)    / STEP_DURATION_S, 0.01)
        v_yaw = max(abs(d_yaw) / STEP_DURATION_S, 0.01)

        # --- Diagnostics: waypoint alignment + progress sanity ---
        next_idx = min(idx + 1, len(path_points) - 1)
        to_next = path_points[next_idx, :2] - np.array([ref_pos[0], ref_pos[1]])
        move_xy = np.array([dx, dy], dtype=float)
        align = _safe_div(
            float(np.dot(move_xy, to_next)),
            float(np.linalg.norm(move_xy) * np.linalg.norm(to_next)),
            default=0.0,
        )
        align_deg = math.degrees(math.acos(np.clip(align, -1.0, 1.0))) if np.linalg.norm(move_xy) > 1e-9 and np.linalg.norm(to_next) > 1e-9 else 0.0

        if prev_idx is not None:
            if idx < prev_idx - 2:
                _log(f"WARNING: closest_idx regressed {prev_idx} -> {idx} (progress {prev_progress:.3f} -> {progress:.3f})")
        prev_idx = idx
        prev_progress = progress

        if step % log_every == 0 or step < 3:
            _log(
                f"step={step+1:03d}/{args.max_steps} "
                f"idx(ref)={idx:04d}/{len(path_points)-1} prog(ref)={progress:.3f} dev(ref)={deviation:.3f}m "
                f"| idx(eef)={idx_eef:04d} prog(eef)={progress_eef:.3f} dev(eef)={deviation_eef:.3f}m"
            )
            _log(
                f"  ref_xy={_fmt_xy(ref_pos)} eef_xy={_fmt_xy(hand_pos[:2])} "
                f"yaw={hand_yaw:+.3f}rad ({_deg(hand_yaw):+.1f}deg)"
            )
            _log(
                f"  state: lat_err={state[0]:+.3f}m ori_err={state[1]:+.3f}rad({_deg(state[1]):+.1f}deg) "
                f"v_fwd={state[2]:+.3f} v_lat={state[3]:+.3f} yaw_rate={state[4]:+.3f} rf={state[5]:.3f}"
            )
            _log(
                f"  action_raw={_fmt_vec(raw_action)} action={_fmt_vec(action)}"
                + (" (CLIPPED)" if clipped else "")
                + f" -> dxdy={_fmt_xy(move_xy)} dyaw={d_yaw:+.3f}rad"
            )
            _log(
                f"  to_next={_fmt_vec(to_next, prec=3)} align=cos={align:+.3f} (~{align_deg:.1f}deg) "
                f"cmd_v: vx={vx:.3f} vy={vy:.3f} vyaw={v_yaw:.3f}"
            )

        # 6. Execute
        try:
            spot.push_object_from_sim(
                dx=dx, dy=dy, d_yaw=d_yaw,
                vx=vx, vy=vy, v_yaw=v_yaw,
                dt=STEP_DURATION_S,
            )
            # spot.push_object_from_sim(
            #     dx=0.5, dy=0, d_yaw=0,
            #     vx=0.5, vy=0, v_yaw=0,
            #     dt=STEP_DURATION_S,
            # )
        except Exception as exc:
            _log(
                "ERROR during push_object_from_sim: "
                f"{type(exc).__name__}: {exc} | "
                f"dx={dx:+.3f} dy={dy:+.3f} dyaw={d_yaw:+.3f} "
                f"vx={vx:.3f} vy={vy:.3f} vyaw={v_yaw:.3f}"
            )
            raise

    duration_s = time.time() - start_time
    print("")
    _log(f"Loop done: steps={len(robot_positions)} duration={duration_s:.1f}s success={success}")

    return success, robot_positions, state_log, action_log, duration_s


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def push_main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"push_{args.path_type}_{timestamp}"
    run_dir   = os.path.join("experiment_logs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    spot = Spot(id="Pusher", hostname=args.hostname)
    spot.start()

    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            print("[push] Powering on and standing up...")
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()
            #  2. Walk forward by 1 meter (no rotation)
            # walk_distance = 1  # meters
            # command_client = spot._client._command_client
            # state_client = spot._client._state_client
            # robot_state = state_client.get_robot_state()
            # transforms = robot_state.kinematic_state.transforms_snapshot

            # duration = 4.0
            # end_time = time.time() + duration

            # # Walk forward
            # traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            #     walk_distance, 0.0, 0.0, transforms
            # )
            # cmd_id = command_client.robot_command(traj_cmd, end_time_secs=end_time)
            # time.sleep(duration)

            # print("finished walking forwards")
            # spot.unstow_arm()
            



            saved_yaw = spot.save_initial_yaw()

            # ----------------------------------------------------------
            # Step 1: Grasp
            # ----------------------------------------------------------
            print("\n[push] === STEP 1: Grasp ===")
            color_img, depth_img = spot.take_picture(
                color_src=args.image_source,
                depth_src=args.depth_source,
                save_images=True,
            )
            if color_img is None:
                raise RuntimeError("Failed to capture image.")

            if args.autonomous_detection:
                print("[push] Autonomous edge detection (SAM)...")
                target_pixel = SpotPerception.find_grasp_sam(
                    color_img, depth_img,
                    left=(args.robot_side == "left"),
                    conf=0.15, max_distance_m=3.0,
                )
                if target_pixel is None:
                    print("[push] Autonomous detection failed — falling back to manual.")
                    target_pixel = SpotPerception.get_target_from_user(color_img)
            else:
                print("[push] Manual grasp point selection — click on the box edge.")
                target_pixel = SpotPerception.get_target_from_user(color_img)

            if target_pixel is None:
                raise RuntimeError("No grasp target selected.")

            spot.open_gripper()
            if not spot.grasp_edge(target_pixel, img_src=args.image_source):
                raise RuntimeError("Grasp command failed.")
            if not spot.check_grip():
                raise RuntimeError("Grip check failed — object not held.")

            print("[push] Grasp confirmed. Settling 2s...")
            time.sleep(2.0)

            # Capture push direction yaw HERE — robot is in correct position
            _, _, push_yaw = spot.get_current_pose()
            # print(f"[push] Push yaw captured: {push_yaw:.3f} rad")

            # # ----------------------------------------------------------
            # # Step 2: Reactive force estimation
            # # ----------------------------------------------------------
            # print("\n[push] === STEP 2: Reactive Force Estimation ===")
            prober = ForceProber()
            spot.open_gripper()  # Ensure gripper is open for probing

            if args.probe_force:
                norm_rf = prober.probe(
                    spot,
                    grasp_strategy="edge_grasp",
                    surface_type=args.surface_type,
                    robot_side=args.robot_side,
                )
            else:
                norm_rf = prober._use_default(
                    grasp_strategy="edge_grasp",
                    surface_type=args.surface_type,
                )

            print(f"[push] norm_reactive_force = {norm_rf:.3f}")

            # Re-grasp after probing
            # Close gripper to re-establish grasp
            spot.close_gripper()
            time.sleep(1.0)

            # print("[push] Returning to saved yaw before path definition...")
            # spot.return_to_saved_yaw(saved_yaw)
            # time.sleep(1.0)

            # ----------------------------------------------------------
            # Step 3: Path definition
            # ----------------------------------------------------------
            print("\n[push] === STEP 3: Path Definition ===")
            hand_pos, hand_yaw = get_eef_pose(spot)
            # Define path origin: default is EEF pose; optionally box center when requested.
            if not getattr(args, "use_box_center", False):
                origin_xy = hand_pos[:2]
                z_height  = hand_pos[2]
            else:
                box_dims = {
                    "width": args.box_width,
                    "depth": args.box_depth,
                    "height": args.box_height,
                }
                box_center = _estimate_box_center_from_grasp(
                    gripper_pos=hand_pos,
                    box_dimensions=box_dims,
                    current_yaw=hand_yaw,
                    robot_side=args.robot_side,
                )
                origin_xy = box_center[:2]
                z_height  = box_center[2]

            gen = PathGenerator(
                saved_yaw=push_yaw,  # Align path with push direction
                origin_xy=origin_xy,
                z_height=z_height,
            )

            path_type = args.path_type.lower()
            if path_type == "arc":
                path_points = gen.arc(
                    radius=args.arc_radius,
                    start_angle=math.pi,
                    end_angle=math.pi + math.radians(args.arc_angle),
                    num_points=max(10, int(args.arc_radius * 20)),
                )
            elif path_type == "straight":
                path_points = gen.straight(
                    length=args.length,
                    num_points=max(10, int(args.length * 20)),
                )
            elif path_type == "s_curve":
                path_points = gen.s_curve(length=args.length, amplitude=args.amplitude)
            elif path_type == "meander":
                path_points = gen.meander(length=args.length, amplitude=args.amplitude)
            elif path_type == "triple_s":
                path_points = gen.triple_s(length=args.length, amplitude=args.amplitude)
            else:
                raise ValueError(f"Unknown path type: {args.path_type}")

            path_len = float(np.sum(np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)))
            print(f"[push] Path: {path_type}, {len(path_points)} pts, length ≈ {path_len:.2f}m")

            # ----------------------------------------------------------
            # Step 4: Load policy
            # ----------------------------------------------------------
            print("\n[push] === STEP 4: Loading Policy ===")
            action_dim = 2 if getattr(args, "push_from_edge", False) else ACTION_DIM
            policy = load_policy(
                model_dir=args.model_dir,
                model_name=args.model_name,
                max_action=1.0,
                max_torque=1.0,
                action_dim=action_dim,
            )

            # ----------------------------------------------------------
            # Step 5: Push execution
            # ----------------------------------------------------------
            print("\n[push] === STEP 5: Push Execution ===")
            success, robot_positions, state_log, action_log, duration_s = run_push(
                spot=spot,
                policy=policy,
                path_points=path_points,
                norm_reactive_force=norm_rf,
                args=args,
            )

            # ----------------------------------------------------------
            # Step 6: Release and dock
            # ----------------------------------------------------------
            print("\n[push] === STEP 6: Release and Dock ===")
            spot.open_gripper()
            time.sleep(1.0)
            spot.stow_arm()
            spot.dock(dock_id=args.dock_id)

            # Save
            config = vars(args)
            config["norm_reactive_force"] = float(norm_rf)
            save_run_data(run_dir, config, robot_positions, path_points,
                          state_log, action_log, duration_s, success)
            save_plots(robot_positions, path_points, run_dir, run_name)

            print(f"\n[push] === {'SUCCESS' if success else 'FAILED'} ===")
            return success

        except KeyboardInterrupt:
            print("\n[push] Interrupted — shutting down safely...")
            _safe_shutdown(spot, args.dock_id)
            return False

        except Exception as e:
            print(f"\n[push] Error: {e}")
            _safe_shutdown(spot, args.dock_id)
            raise


def _safe_shutdown(spot, dock_id):
    for fn in [spot.open_gripper, spot.stow_arm, lambda: spot.dock(dock_id=dock_id)]:
        try:
            fn()
            time.sleep(0.5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push an object along a path with Spot using a TD3 policy."
    )

    # Connection
    parser.add_argument("--hostname",  required=True)
    parser.add_argument("--dock-id",   type=int, default=521)

    # Perception
    parser.add_argument("--image-source",  default="hand_color_image")
    parser.add_argument("--depth-source",  default="hand_depth_in_hand_color_frame")
    parser.add_argument("--autonomous-detection", action="store_true")
    parser.add_argument("--robot-side", choices=["left", "right"], default="right")

    # Box dimensions (for box-center state estimation; matches sim)
    parser.add_argument("--use-box-center", action="store_true",
                        help="Use box-center state/path; default is EEF pose only")
    parser.add_argument("--box-width",   type=float, default=0.4, help="Box width (m)")
    parser.add_argument("--box-depth",   type=float, default=0.4, help="Box depth (m)")
    parser.add_argument("--box-height",  type=float, default=0.3, help="Box height (m)")

    # Path
    parser.add_argument("--path-type",
                        choices=["arc", "straight", "s_curve", "meander", "triple_s"],
                        default="arc")
    parser.add_argument("--arc-radius",  type=float, default=1.5)
    parser.add_argument("--arc-angle",   type=float, default=60.0)
    parser.add_argument("--length",      type=float, default=3.0)
    parser.add_argument("--amplitude",   type=float, default=0.5)

    # Policy — points directly to models/react/<model_name>_actor.pth
    parser.add_argument("--model-dir",  default="models/react",
                        help="Directory containing TD3 weights (default: models/react)")
    parser.add_argument("--model-name", default="final_model",
                        help="Name prefix for saved weights, e.g. 'best' → best_actor.pth "
                             "(default: best)")
    parser.add_argument("--push-from-edge", action="store_true",
                        help="Use 2D policy (Fx, Fy only) trained with push_from_edge")

    # Reactive force
    parser.add_argument("--probe-force",  action="store_true",
                        help="Actively probe for reactive force. If not set, uses calibrated default.")
    parser.add_argument("--surface-type", choices=["floor", "mat"], default="floor")
    parser.add_argument("--max-force",    type=float, default=400.0,
                        help="F_max for norm_reactive_force (N, default: 400)")

    # Execution
    parser.add_argument("--max-steps",        type=int,   default=50)
    parser.add_argument("--action-scale",      type=float, default=DEFAULT_ACTION_SCALE)
    parser.add_argument("--yaw-scale",         type=float, default=DEFAULT_YAW_SCALE)
    parser.add_argument("--success-distance",  type=float, default=0.8)
    parser.add_argument("--log-every",         type=int, default=1,
                        help="Print detailed diagnostics every N policy steps (default: 1).")

    args = parser.parse_args()

    try:
        ok = push_main(args)
        sys.exit(0 if ok else 1)
    except Exception as exc:
        print(f"[push] Fatal: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()