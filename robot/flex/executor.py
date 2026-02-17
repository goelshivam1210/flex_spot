"""
FlexExecutor - policy execution system for robotic manipulation.

This class is platform-agnostic and can work with any gym environment.
"""

import math
import time
import numpy as np
from scipy.optimize import least_squares
import logging
from typing import Tuple, Dict, Any

from flex.config import FlexConfig
from flex.interactive_perception import InteractivePerception
from flex.policy_manager import PolicyManager

logger = logging.getLogger(__name__)


class FlexExecutor:
    """
    executor for FLEX manipulation policies.
    
    Usage:
        executor = FlexExecutor(config)
        joint_type, params = executor.analyze_joint(analysis_env)
        success, msg = executor.execute_policy(prismatic_env, joint_type, params)
    """
    
    def __init__(self, config: FlexConfig):
        self.config = config
        self.perception = InteractivePerception(
            movement_distance=config.movement_distance
        )
        self.policy_manager = PolicyManager(
            models_dir=config.prismatic_policy_path.rsplit('/', 1)[0]  # Get base models dir
        )
        self._policy_cache = {}

        # used to cancel an executing policy
        self.should_cancel=False

    #     self.theta = 0
    #     self.radius_est = 1000
    #
    # def analyze_and_execute_policy(self, env, joint_type: str,
    #                   joint_params: Dict[str, Any]) -> Tuple[bool, str]:
    #     """
    #             Complete policy execution pipeline for manipulation.
    #
    #             Returns:
    #                 Tuple of (success: bool, message: str)
    #             """
    #     logger.info("=" * 60)
    #     logger.info(f"PHASE 0: Analysis and Policy Execution ({joint_type})")
    #     logger.info("=" * 60)
    #
    #     x_points = []
    #     y_points = []
    #
    #     # reset cancel flag
    #     self.should_cancel = False
    #
    #
    #     x = np.array(x_points)
    #     y = np.array(y_points)
    #     n = len(x)
    #
    #
    #
    #     # Set joint params in env
    #     joint_type = "linear"
    #     env.joint_params = joint_params
    #
    #
    #
    #     # Reset environment
    #     obs, info = env.reset()
    #     logger.info(f"Environment reset. Observation shape: {obs.shape}")
    #
    #     # Track initial position (from env's initial state)
    #     initial_position = self._get_initial_position(env)
    #     logger.info(f"Initial position: {initial_position}")
    #
    #     # Load policy
    #     policy = self._get_policy(joint_type)
    #     logger.info(f"Loaded {joint_type} policy")
    #
    #     # Execution loop
    #     success = False
    #     step = 0
    #
    #     for step in range(self.config.max_steps):
    #         if self.should_cancel:
    #             logger.info("execute_policy cancelled by user")
    #             return None, None
    #
    #         logger.info(f"\nStep {step + 1}/{self.config.max_steps}")
    #
    #         # Get action from policy
    #         raw_action = policy.select_action(obs)
    #         logger.info(f"  Raw action: {raw_action}")
    #
    #         # Process action based on joint type
    #         processed_action = self._process_action(raw_action, joint_type, joint_params)
    #         logger.info(f"  Processed action: {processed_action}")
    #
    #         # Execute via environment
    #         obs, reward, terminated, truncated, info = env.step(processed_action)
    #
    #         # Get current position and check success
    #         current_position = self._get_current_position(env)
    #         success = self._check_success(joint_type, joint_params,
    #                                       initial_position, current_position)
    #
    #         if success or terminated or truncated:
    #             break
    #
    #     # Results
    #     if success:
    #         msg = f"Successfully completed in {step + 1} steps"
    #     else:
    #         msg = f"Did not reach success threshold after {step + 1} steps"
    #
    #     logger.info("=" * 60)
    #     logger.info(msg)
    #     logger.info("=" * 60)
    #
    #     return success, msg
    #
    #
    #
    # def unified_geometry_estimator(x_points, y_points):
    #     """
    #     Robustly estimates geometry (Line vs Circle) and returns a Confidence score.
    #     Returns:
    #         shape_type: "LINEAR" or "CIRCULAR"
    #         params: (xc, yc, R) or (vx, vy, inf) for line
    #         confidence: 0.0 to 1.0
    #     """
    #
    #
    #     # 1. Safety: Need minimal movement to estimate anything
    #     # Calculate spatial span (distance between start and end)
    #     span = np.hypot(x[-1] - x[0], y[-1] - y[0])
    #     if n < 5 or span < 0.01:  # Less than 1cm movement
    #         return "UNDEFINED", (0, 0, 0), 0.0
    #
    #     # --- PHASE 1: FIT BOTH MODELS ---
    #
    #     # Model A: Line Fit (PCA / SVD is better than polyfit for vertical lines)
    #     # We center data to avoid offset issues
    #     x_mean, y_mean = np.mean(x), np.mean(y)
    #     u, s, vh = np.linalg.svd(np.vstack([x - x_mean, y - y_mean]).T)
    #     # Direction vector is the first row of vh
    #     dir_x, dir_y = vh[0]
    #     # Calculate residuals (perpendicular distance to line)
    #     # distance = |(p - mean) dot normal|
    #     normal_x, normal_y = -dir_y, dir_x
    #     line_residuals = (x - x_mean) * normal_x + (y - y_mean) * normal_y
    #     rss_line = np.sum(line_residuals ** 2)
    #
    #     # Model B: Circle Fit
    #     def circle_residuals(params, x, y):
    #         xc, yc, r = params
    #         return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r
    #
    #     # Initial guess
    #     initial_R = span / 2 + 0.1
    #     # Optimization
    #     try:
    #         res = least_squares(circle_residuals, [x_mean, y_mean, initial_R], args=(x, y))
    #         rss_circle = np.sum(res.fun ** 2)
    #         circle_params = res.x  # xc, yc, R
    #     except Exception:
    #         rss_circle = np.inf  # Fit failed
    #
    #     # --- PHASE 2: AIC SELECTION ---
    #
    #     # AIC = n * ln(RSS/n) + 2k
    #     aic_line = n * np.log(rss_line / n + 1e-9) + 2 * 2
    #     aic_circle = n * np.log(rss_circle / n + 1e-9) + 2 * 3
    #
    #     # Decision: Is Circle SIGNIFICANTLY better? (Lower is better)
    #     # We add a buffer (e.g., 2.0) to prefer the simpler Line model when ambiguous
    #     is_circular = aic_circle < (aic_line - 2.0)
    #
    #     # --- PHASE 3: CONFIDENCE CALCULATION ---
    #
    #     if not is_circular:
    #         # === LINEAR CASE ===
    #         # Confidence depends on:
    #         # 1. Straightness (RMSE) - Are points actually on the line?
    #         rmse = np.sqrt(rss_line / n)
    #         score_fit = np.clip(1.0 - (rmse / 0.02), 0.0, 1.0)  # 0.02m (2cm) tolerance
    #
    #         # 2. Length Span - Have we moved enough to be sure?
    #         # If we moved 10cm, we are confident. If 1cm, not confident.
    #         score_span = np.clip(span / 0.10, 0.0, 1.0)
    #
    #         confidence = score_fit * score_span
    #
    #         # Return direction vector as params
    #         return "LINEAR", (dir_x, dir_y, np.inf), confidence
    #
    #     else:
    #         # === CIRCULAR CASE ===
    #         xc, yc, R_est = circle_params
    #
    #         # 1. Covariance Score (Math Uncertainty)
    #         mse = rss_circle / (n - 3)
    #         try:
    #             # Calculate Covariance
    #             J = res.jac
    #             cov = np.linalg.pinv(J.T @ J) * mse
    #             sigma_R = np.sqrt(cov[2, 2])
    #
    #             # Relative Error in Radius
    #             score_math = np.clip(1.0 - (sigma_R / R_est), 0.0, 1.0)
    #         except:
    #             score_math = 0.0
    #
    #         # 2. Angular Span Score (Geometric Diversity)
    #         # Calculate start and end angles
    #         angles = np.arctan2(y - yc, x - xc)
    #         # Unwrap to handle crossing the -pi/pi boundary
    #         angles = np.unwrap(angles)
    #         angle_span = np.abs(angles[-1] - angles[0])
    #
    #         # We want at least 15 degrees (0.26 rad) to be confident
    #         score_span = np.clip(angle_span / 0.26, 0.0, 1.0)
    #
    #         confidence = score_math * score_span
    #
    #         return "CIRCULAR", (xc, yc, R_est), confidence

    def analyze_joint(self, env) -> Tuple[str, Dict[str, Any]]:
        """
        Complete joint analysis pipeline using wiggle exploration.
        
        Returns:
            Tuple of (joint_type, joint_params)
        """
        logger.info("="*60)
        logger.info("PHASE 1: Joint Analysis")
        logger.info("="*60)

        # reset cancel flag
        self.should_cancel=False

        # Reset and get starting position
        obs, info = env.reset()
        start_position = obs
        logger.info(f"Starting position: {start_position}")
        
        # Generate wiggle waypoints
        wiggle_positions = self.perception.generate_wiggle_positions(start_position)
        logger.info(f"Generated {len(wiggle_positions)} wiggle waypoints")
        
        # Execute wiggle and collect trajectory
        trajectory = [start_position.copy()]
        
        for i, target_pos in enumerate(wiggle_positions):
            if self.should_cancel:
                logger.info("analyze_joint cancelled by user")
                return None, None

            logger.info(f"Wiggle movement {i+1}/{len(wiggle_positions)}")
            obs, reward, terminated, truncated, info = env.step(target_pos)
            trajectory.append(obs.copy())
            
            if terminated or truncated:
                logger.warning(f"Environment terminated early at step {i+1}")
                break
        
        trajectory = np.array(trajectory)
        logger.info(f"Collected trajectory with {len(trajectory)} points")
        
        # Analyze trajectory
        joint_type, joint_params = self.perception.analyze_trajectory_and_estimate_joint(trajectory)
        
        # Apply force override if specified
        if self.config.force_joint_type:
            logger.info(f"Applying joint type override: {self.config.force_joint_type}")
            joint_type, joint_params = self._apply_joint_override(
                trajectory, self.config.force_joint_type
            )
        
        logger.info(f"Analysis complete: {joint_type}")
        logger.info(f"Parameters: {joint_params}")
        logger.info("="*60)

        return joint_type, joint_params

    def estimate_trajectory_2d_via_probing(
        self,
        env,
        direction=np.array([0.0, -1.0]),
        step_size=0.04,  # meters
        phi_max=np.pi/4,
        phi_min=0.0,
        min_steps=5,
        confidence_thresh=0.8,
        max_steps=10,
        plot_path=None,
        eps=1e-6,
    ):
        """
        Estimate a 2D trajectory model by probing and adapting direction.

        Args:
            env: environment providing step() and optionally _get_current_position().
            direction: (2,) initial unit direction.
            step_size: step length.
            phi_max: maximum rotation (radians).
            phi_min: minimum rotation (radians).
            min_steps: minimum samples before fitting.
            confidence_thresh: confidence threshold.
            max_steps: hard iteration cap to avoid infinite loops.
            plot_path: optional filepath for fit plot output.
            eps: small constant to avoid division by zero.

        Returns:
            (joint_type, joint_params) to match analyze_joint().
        """
        self.should_cancel = False
        logger.info("=" * 60)
        logger.info("PHASE 1B: 2D Probing Trajectory Estimation")
        logger.info("=" * 60)
        obs, _ = env.reset()
        start_position = obs
        start_position = np.asarray(start_position, dtype=float)
        x = start_position[:2].reshape(2)
        z = start_position[2] if start_position.shape[0] >= 3 else 0.0
        logger.info(f"Starting position: {start_position}")

        theta = np.asarray(direction, dtype=float).reshape(2)
        theta_norm = np.linalg.norm(theta)
        if theta_norm < eps:
            raise ValueError("theta0 must be a non-zero 2D vector.")
        theta = theta / theta_norm
        logger.info(f"Initial direction theta0: {theta}")
        logger.info(f"Step length d: {step_size}, phi_max: {phi_max}, phi_min: {phi_min}, n_iters: {max_steps}")

        X = [x.copy()]
        trajectory = [start_position.copy()]
        p = x + step_size * theta
        model = ("UNDEFINED", None)
        confidence = 0.0

        def rot2(angle):
            ca = math.cos(angle)
            sa = math.sin(angle)
            return np.array([[ca, -sa], [sa, ca]])

        for _ in range(max_steps):
            if self.should_cancel:
                logger.info("estimate_trajectory_2d_via_probing cancelled by user")
                return None, None
            logger.info(f"Probing target (xy): {p}")
            x_prev = x.copy()
            target = np.array([p[0], p[1], z], dtype=float)
            t_before = time.time()
            logger.info(f"Step start time: {t_before:.3f}")
            obs, _, terminated, truncated, _ = env.step(target)
            t_after = time.time()
            logger.info(f"Step end time: {t_after:.3f}, elapsed: {t_after - t_before:.3f}s")
            obs = np.asarray(obs, dtype=float)
            x = obs[:2].reshape(2)
            if obs.shape[0] >= 3:
                z = obs[2]
            X.append(x.copy())
            trajectory.append(obs.copy())
            logger.info(f"Observed position (xy): {x}")
            if terminated or truncated:
                logger.info("estimate_trajectory_2d_via_probing stopped: environment ended")
                break

            delta = np.linalg.norm(x - x_prev)
            r = min(1.0, delta / step_size) if step_size > 0 else 0.0
            logger.info(f"Progress delta: {delta:.4f}, ratio r: {r:.3f}")

            if len(X) >= min_steps:
                model, confidence = self.perception.fit_model_2d(np.asarray(X))
            else:
                confidence = 0.0
            logger.info(f"Fit model: {model[0]}, confidence: {confidence:.3f}, samples: {len(X)}")

            if confidence >= confidence_thresh:
                model_type, params = model
                if model_type == "LINEAR":
                    direction = params["direction"]
                    direction = direction / max(np.linalg.norm(direction), eps)
                    if np.dot(direction, theta) < 0:
                        direction = -direction
                    theta = direction
                    p = x + step_size * direction
                    logger.info(f"Following line, new theta: {theta}")
                elif model_type == "CIRCULAR":
                    center = params["center"]
                    radius = params["radius"]
                    if radius <= eps:
                        p = x + step_size * theta
                        logger.info("Circle radius too small, keep heading.")
                    else:
                        vec = x - center
                        angle = math.atan2(vec[1], vec[0])
                        tangent_ccw = np.array([-math.sin(angle), math.cos(angle)])
                        tangent_cw = -tangent_ccw
                        if np.dot(tangent_ccw, theta) >= np.dot(tangent_cw, theta):
                            sign = 1.0
                            tangent = tangent_ccw
                        else:
                            sign = -1.0
                            tangent = tangent_cw
                        theta = tangent / max(np.linalg.norm(tangent), eps)
                        angle = angle + sign * (step_size / radius)
                        p = center + radius * np.array([math.cos(angle), math.sin(angle)])
                        logger.info(f"Following circle, center: {center}, radius: {radius:.3f}")
                else:
                    p = x + step_size * theta
            else:
                phi = (1.0 - r) * phi_max
                if phi >= phi_min:
                    u = x - x_prev
                    u = u / max(np.linalg.norm(u), eps)
                    theta_plus = rot2(phi) @ theta
                    theta_minus = rot2(-phi) @ theta
                    if np.dot(theta_plus, u) >= np.dot(theta_minus, u):
                        theta = theta_plus
                    else:
                        theta = theta_minus
                    theta = theta / max(np.linalg.norm(theta), eps)
                    logger.info(f"Low confidence, rotated theta: {theta}, phi: {phi:.3f}")
                p = x + step_size * theta

        if len(trajectory) < 2:
            logger.warning("estimate_trajectory_2d_via_probing collected too few points")
            return None, None

        trajectory = np.asarray(trajectory)
        joint_type, joint_params = self.perception.analyze_trajectory_and_estimate_joint(
            trajectory,
            plot_fits=True,
            plot_path=plot_path,
        )
        logger.info(f"Estimated joint type: {joint_type}")
        logger.info(f"Estimated joint params: {joint_params}")
        if self.config.force_joint_type:
            logger.info(f"Applying joint type override: {self.config.force_joint_type}")
            joint_type, joint_params = self._apply_joint_override(
                trajectory, self.config.force_joint_type
            )
            logger.info(f"Overridden joint type: {joint_type}")
            logger.info(f"Overridden joint params: {joint_params}")

        logger.info("=" * 60)
        return joint_type, joint_params
    
    def execute_policy(self, env, joint_type: str, 
                      joint_params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Complete policy execution pipeline for manipulation.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info("="*60)
        logger.info(f"PHASE 2: Policy Execution ({joint_type})")
        logger.info("="*60)

        # reset cancel flag
        self.should_cancel=False
        
        if joint_type not in ['prismatic', 'revolute']:
            msg = f"Invalid joint type: {joint_type}"
            logger.error(msg)
            return False, msg
        
        # Set joint params in env
        env.joint_params = joint_params
        
        # Reset environment
        obs, info = env.reset()
        logger.info(f"Environment reset. Observation shape: {obs.shape}")
        
        # Track initial position (from env's initial state)
        initial_position = self._get_initial_position(env)
        logger.info(f"Initial position: {initial_position}")
        
        # Load policy
        policy = self._get_policy(joint_type)
        logger.info(f"Loaded {joint_type} policy")
        
        # Execution loop
        success = False
        step = 0
        
        for step in range(self.config.max_steps):
            if self.should_cancel:
                logger.info("execute_policy cancelled by user")
                return None, None

            logger.info(f"\nStep {step+1}/{self.config.max_steps}")
            
            # Get action from policy
            raw_action = policy.select_action(obs)
            logger.info(f"  Raw action: {raw_action}")
            
            # Process action based on joint type
            processed_action = self._process_action(raw_action, joint_type, joint_params)
            logger.info(f"  Processed action: {processed_action}")
            
            # Execute via environment
            obs, reward, terminated, truncated, info = env.step(processed_action)

            # Get current position and check success
            current_position = self._get_current_position(env)
            success = self._check_success(joint_type, joint_params,
                                         initial_position, current_position)

            if success or terminated or truncated:
                break
        
        # Results
        if success:
            msg = f"Successfully completed in {step+1} steps"
        else:
            msg = f"Did not reach success threshold after {step+1} steps"
        
        logger.info("="*60)
        logger.info(msg)
        logger.info("="*60)
        
        return success, msg

    def cancel_policy(self):
        """Cancel current policy execution."""
        logger.info("Cancelling policy execution")
        self.should_cancel=True
    
    def _get_policy(self, joint_type: str):
        """Load policy (uses PolicyManager)."""
        if joint_type not in self._policy_cache:
            self._policy_cache[joint_type] = self.policy_manager.load_policy(joint_type)
        return self._policy_cache[joint_type]
    
    def _get_initial_position(self, env) -> np.ndarray:
        """Get initial position from environment."""
        if hasattr(env, '_get_current_position'):
            return env._get_current_position()
        else:
            # Fallback: assume position is in observation
            return np.zeros(3)
    
    def _get_current_position(self, env) -> np.ndarray:
        """Get current position from environment."""
        if hasattr(env, '_get_current_position'):
            return env._get_current_position()
        else:
            return np.zeros(3)
    
    def _process_action(self, action: np.ndarray, joint_type: str, 
                       joint_params: Dict[str, Any]) -> np.ndarray:
        """Process action based on joint constraints."""
        if len(action.shape) > 1:
            action = action.flatten()
        
        # Spot-specific inversion
        action[1] = -action[1]
        action[0] = -action[0]
        
        if joint_type == "prismatic":
            sliding_axis = joint_params["axis"]
            sliding_axis = sliding_axis / np.linalg.norm(sliding_axis)
            action_magnitude = np.dot(action, sliding_axis)
            return action_magnitude * sliding_axis
        
        return action
    
    def _check_success(self, joint_type: str, joint_params: Dict[str, Any],
                      initial_position: np.ndarray, 
                      current_position: np.ndarray) -> bool:
        """Check if success threshold met."""
        if joint_type == "prismatic":
            distance = np.linalg.norm(current_position - initial_position)
            success = distance >= self.config.success_distance
            logger.info(f"  Distance: {distance:.3f}m (threshold: {self.config.success_distance:.3f}m)")
            return success
        
        elif joint_type == "revolute":
            joint_center = joint_params['center']
            initial_vec = initial_position - joint_center
            current_vec = current_position - joint_center
            
            cos_angle = np.dot(initial_vec, current_vec) / (
                np.linalg.norm(initial_vec) * np.linalg.norm(current_vec)
            )
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            success = angle_deg >= self.config.success_angle
            logger.info(f"  Angle: {angle_deg:.1f}° (threshold: {self.config.success_angle:.1f}°)")
            return success
        
        return False
    
    def _apply_joint_override(self, trajectory: np.ndarray, 
                             forced_type: str) -> Tuple[str, Dict[str, Any]]:
        """Force specific joint type analysis."""
        if forced_type == "prismatic":
            error, axis = self.perception.prismatic_error_analysis(trajectory)
            return "prismatic", {"axis": axis, "error": error}
        elif forced_type == "revolute":
            error, center, radius, axis = self.perception.revolute_error_analysis(trajectory)
            return "revolute", {
                "center": center, "radius": radius, "axis": axis, "error": error
            }
        return self.perception.analyze_trajectory_and_estimate_joint(trajectory)
