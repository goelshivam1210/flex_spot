"""
push_baseline.py

Trajectory-following baseline for push manipulation with Spot.

Instead of a learned policy, the robot simply walks along the planned
path waypoints one by one while holding the grasped object with a
frozen arm. This serves as a comparison baseline for the TD3 policy.

Pipeline:
    1. Initialise robot and save starting yaw.
    2. Detect and grasp the object (manual pixel selection or autonomous).
    3. Define path (arc, straight, s-curve, meander, or triple-s).
    4. Walk through waypoints with arm joints frozen.
    5. Release, stow arm, dock.

Usage:
    python -m policies.push_baseline --hostname 192.168.1.100 --path-type arc
    python -m policies.push_baseline --hostname 192.168.1.100 --path-type straight --length 1.5
    python -m policies.push_baseline --hostname 192.168.1.100 --path-type arc \\
                   --arc-radius 1.5 --arc-angle 60

Date: February 2026
"""

import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.keepalive import KeepaliveClient
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2

from spot.spot import Spot, SpotPerception
from react.state_estimator import _estimate_box_center_from_grasp
from react.path_generator import PathGenerator

from .args import parse_push_args, PushArgs


# ---------------------------------------------------------------------------
# Logging helpers (reused from push.py)
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _log(msg: str) -> None:
    print(f"[baseline][{_ts()}] {msg}")

def _fmt_xy(xy, prec: int = 3) -> str:
    return f"({float(xy[0]):+.{prec}f}, {float(xy[1]):+.{prec}f})"


# ---------------------------------------------------------------------------
# EEF pose helper (same as push.py)
# ---------------------------------------------------------------------------

def get_eef_pose(spot):
    state = spot._client._state_client.get_robot_state()
    snapshot = state.kinematic_state.transforms_snapshot
    vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
    if vision_T_hand is None:
        raise RuntimeError("Could not get hand pose from Spot SDK.")
    hand_pos = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
    q = vision_T_hand.rot
    hand_z_x = 2.0 * (q.x * q.z + q.y * q.w)
    hand_z_y = 2.0 * (q.y * q.z - q.x * q.w)
    yaw = math.atan2(hand_z_y, hand_z_x)
    return hand_pos, yaw


# ---------------------------------------------------------------------------
# Progress helper (same as push.py)
# ---------------------------------------------------------------------------

def compute_progress(hand_pos, path_points):
    pts_2d = path_points[:, :2]
    pos_2d = hand_pos[:2]
    dists = np.linalg.norm(pts_2d - pos_2d, axis=1)
    idx = int(np.argmin(dists))
    progress = idx / (len(path_points) - 1) if len(path_points) > 1 else 0.0
    deviation = float(dists[idx])
    return progress, deviation, idx


# ---------------------------------------------------------------------------
# Plotting (same as push.py)
# ---------------------------------------------------------------------------

def save_plots(robot_positions, path_points, run_dir, experiment_name):
    positions = np.array(robot_positions)
    rx, ry, ryaw = positions[:, 0], positions[:, 1], positions[:, 2]
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(path_points[:, 0], path_points[:, 1], "b-", lw=2, label="Planned path", alpha=0.7)
    ax.scatter(*path_points[0, :2], color="green", s=120, zorder=5, label="Path start")
    ax.scatter(*path_points[-1, :2], color="red", s=120, marker="s", zorder=5, label="Path end")
    ax.plot(rx, ry, "r-", lw=2, label="EEF trajectory", alpha=0.8)
    ax.scatter(rx[0], ry[0], color="darkgreen", s=100, marker="^", zorder=5, label="EEF start")
    ax.scatter(rx[-1], ry[-1], color="darkred", s=100, marker="v", zorder=5, label="EEF end")

    total_dist = float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))
    path_len = float(np.sum(np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)))
    final_err = float(np.linalg.norm(positions[-1, :2] - path_points[-1, :2]))
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
    print(f"[baseline] Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Data saving (simplified — no state/action logs)
# ---------------------------------------------------------------------------

def save_run_data(run_dir, config, robot_positions, path_points, duration_s, success):
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
    print(f"[baseline] Run data saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Core trajectory-following loop
# ---------------------------------------------------------------------------

def run_trajectory(spot, path_points, args: PushArgs):
    """
    Walk Spot through path waypoints with arm joints frozen.

    At each step, compute the delta from the current body pose to the next
    waypoint(s) and issue a trajectory command. The arm is frozen via
    arm_joint_freeze_command so it holds the grasp rigidly.

    Returns:
        success, robot_positions, duration_s
    """
    hand_pos, hand_yaw = get_eef_pose(spot)
    robot_positions = []
    success = False

    path_len = float(np.sum(np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)))
    waypoint_skip = max(1, args.waypoint_skip)
    step_duration = args.step_duration_s

    _log(f"Starting trajectory loop: {len(path_points)} waypoints, "
         f"skip={waypoint_skip}, step_dt={step_duration:.1f}s")
    _log(f"Path length≈{path_len:.2f}m "
         f"start={_fmt_xy(path_points[0, :2])} end={_fmt_xy(path_points[-1, :2])}")

    obstacles = spot_command_pb2.ObstacleParams(
        disable_vision_body_obstacle_avoidance=True,
        disable_vision_foot_obstacle_avoidance=True,
        disable_vision_foot_constraint_avoidance=True,
        obstacle_avoidance_padding=0.001,
    )
    speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
        linear=Vec2(x=0.3, y=0.3), angular=0.3
    ))
    mobility_params = spot_command_pb2.MobilityParams(
        obstacle_params=obstacles,
        vel_limit=speed_limit,
        locomotion_hint=spot_command_pb2.HINT_AUTO,
    )

    start_time = time.time()
    command_client = spot._client._command_client
    state_client = spot._client._state_client

    waypoint_indices = list(range(0, len(path_points), waypoint_skip))
    if waypoint_indices[-1] != len(path_points) - 1:
        waypoint_indices.append(len(path_points) - 1)

    for step, wp_idx in enumerate(waypoint_indices):
        hand_pos, hand_yaw = get_eef_pose(spot)
        robot_positions.append((hand_pos[0], hand_pos[1], hand_yaw))

        progress, deviation, closest_idx = compute_progress(hand_pos, path_points)
        _log(f"step={step+1:03d} wp={wp_idx:04d}/{len(path_points)-1} "
             f"prog={progress:.3f} dev={deviation:.3f}m "
             f"eef={_fmt_xy(hand_pos[:2])}")

        if progress >= args.success_progress and deviation < args.success_distance:
            _log(f"SUCCESS: progress={progress:.3f} deviation={deviation:.3f}m")
            success = True
            break

        target_xy = path_points[wp_idx, :2]

        snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot
        from bosdyn.client.frame_helpers import get_se2_a_tform_b, BODY_FRAME_NAME
        vision_tform_body = get_se2_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

        dx = target_xy[0] - vision_tform_body.x
        dy = target_xy[1] - vision_tform_body.y
        target_heading = math.atan2(dy, dx)

        arm_cmd = RobotCommandBuilder.arm_joint_freeze_command()
        walk_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=float(target_xy[0]),
            goal_y=float(target_xy[1]),
            goal_heading=target_heading,
            frame_name=VISION_FRAME_NAME,
            params=mobility_params,
            build_on_command=arm_cmd,
        )
        end_t = time.time() + step_duration
        command_client.robot_command(walk_cmd, end_time_secs=end_t)
        time.sleep(step_duration)

    hand_pos, hand_yaw = get_eef_pose(spot)
    robot_positions.append((hand_pos[0], hand_pos[1], hand_yaw))
    progress, deviation, _ = compute_progress(hand_pos, path_points)
    if not success and progress >= args.success_progress and deviation < args.success_distance:
        success = True

    duration_s = time.time() - start_time
    _log(f"Loop done: steps={len(robot_positions)} duration={duration_s:.1f}s success={success}")
    return success, robot_positions, duration_s


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def baseline_main(args: PushArgs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"baseline_{args.path_type}_{timestamp}"
    run_dir = os.path.join("experiment_logs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    spot = Spot(id="Baseline", hostname=args.hostname)
    spot.start()

    with LeaseKeepAlive(spot.lease_client, must_acquire=True, return_at_exit=True):
        try:
            print("[baseline] Powering on and standing up...")
            keepalive_client = spot._client._spot.ensure_client(KeepaliveClient.default_service_name)
            status = keepalive_client.get_status()
            policy_ids = [p.policy_id for p in status.status]
            _log(f"Keepalive: {len(policy_ids)} policy(ies) found: {policy_ids}")
            if policy_ids:
                try:
                    keepalive_client.modify_policy(policy_ids_to_remove=policy_ids)
                    _log("Keepalive: Removed all policies.")
                except Exception as e:
                    _log(f"Keepalive: Could not remove policies: {e}")
            if policy_ids:
                time.sleep(1.0)
            spot.power_on()
            spot.stand_up()
            spot.open_gripper()

            if args.walk_back > 0:
                _log(f"Walking backwards {args.walk_back:.2f}m...")
                command_client = spot._client._command_client
                state_client = spot._client._state_client
                robot_state = state_client.get_robot_state()
                transforms = robot_state.kinematic_state.transforms_snapshot
                duration = max(4.0, args.walk_back / 0.25)
                end_time = time.time() + duration
                traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    -args.walk_back, 0.0, 0.0, transforms
                )
                command_client.robot_command(traj_cmd, end_time_secs=end_time)
                time.sleep(duration)
                _log("Finished walking backwards.")

            saved_yaw = spot.save_initial_yaw()

            # Step 1: Grasp
            print("\n[baseline] === STEP 1: Grasp ===")
            color_img, depth_img = spot.take_picture(
                color_src=args.image_source,
                depth_src=args.depth_source,
                save_images=True,
            )
            if color_img is None:
                raise RuntimeError("Failed to capture image.")

            if args.autonomous_detection:
                print("[baseline] Autonomous edge detection (SAM)...")
                target_pixel = SpotPerception.find_grasp_sam(
                    color_img, depth_img,
                    left=(args.robot_side == "left"),
                    conf=0.15, max_distance_m=3.0,
                )
                if target_pixel is None:
                    print("[baseline] Autonomous detection failed — falling back to manual.")
                    target_pixel = SpotPerception.get_target_from_user(color_img)
            else:
                print("[baseline] Manual grasp point selection — click on the box edge.")
                target_pixel = SpotPerception.get_target_from_user(color_img)

            if target_pixel is None:
                raise RuntimeError("No grasp target selected.")

            spot.open_gripper()
            if not spot.grasp_edge(target_pixel, img_src=args.image_source):
                raise RuntimeError("Grasp command failed.")
            if not spot.check_grip():
                raise RuntimeError("Grip check failed — object not held.")

            print("[baseline] Grasp confirmed. Settling 2s...")
            time.sleep(2.0)

            # Step 2: Path definition
            print("\n[baseline] === STEP 2: Path Definition ===")
            hand_pos, hand_yaw = get_eef_pose(spot)
            if not args.use_box_center:
                origin_xy = hand_pos[:2]
                z_height = hand_pos[2]
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
                z_height = box_center[2]

            gen = PathGenerator(saved_yaw=0, origin_xy=origin_xy, z_height=z_height)
            path_type = args.path_type.lower()
            if path_type == "arc":
                path_points = gen.arc(
                    radius=args.arc_radius,
                    start_angle=3 * math.pi / 2,
                    end_angle=3 * math.pi / 2 + math.radians(args.arc_angle),
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
            print(f"[baseline] Path: {path_type}, {len(path_points)} pts, length ≈ {path_len:.2f}m")

            spot.open_gripper()
            # Step 3: Trajectory execution
            print("\n[baseline] === STEP 3: Trajectory Execution ===")
            success, robot_positions, duration_s = run_trajectory(
                spot=spot,
                path_points=path_points,
                args=args,
            )

            # Step 4: Release and dock
            print("\n[baseline] === STEP 4: Release and Dock ===")
            spot.open_gripper()
            time.sleep(1.0)
            spot.stow_arm()
            time.sleep(2.0)
            spot.return_to_saved_yaw(saved_yaw)
            time.sleep(1.0)
            _log(f"Walking backwards {args.walk_back:.2f}m...")
            command_client = spot._client._command_client
            state_client = spot._client._state_client
            robot_state = state_client.get_robot_state()
            transforms = robot_state.kinematic_state.transforms_snapshot
            duration = max(4.0, args.walk_back / 0.25)
            end_time = time.time() + duration
            traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                -2.5, 0.0, 0.0, transforms
            )
            command_client.robot_command(traj_cmd, end_time_secs=end_time)
            time.sleep(duration)
            _log("Finished walking backwards.")
            spot.dock(dock_id=args.dock_id)

            config = args.to_dict()
            config["method"] = "trajectory_baseline"
            save_run_data(run_dir, config, robot_positions, path_points, duration_s, success)
            save_plots(robot_positions, path_points, run_dir, run_name)

            print(f"\n[baseline] === {'SUCCESS' if success else 'FAILED'} ===")
            return success

        except KeyboardInterrupt:
            print("\n[baseline] Interrupted — shutting down safely...")
            _safe_shutdown(spot, args.dock_id)
            return False

        except Exception as e:
            print(f"\n[baseline] Error: {e}")
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
    args = parse_push_args()
    try:
        ok = baseline_main(args)
        sys.exit(0 if ok else 1)
    except Exception as exc:
        print(f"[baseline] Fatal: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
