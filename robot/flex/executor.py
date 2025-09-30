"""
FlexExecutor - policy execution system for robotic manipulation.

This class is platform-agnostic and can work with any gym environment.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any

from flex.config import FlexConfig
from interactive_perception import InteractivePerception
from policy_manager import PolicyManager

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
    
    def analyze_joint(self, env) -> Tuple[str, Dict[str, Any]]:
        """
        Complete joint analysis pipeline using wiggle exploration.
        
        Returns:
            Tuple of (joint_type, joint_params)
        """
        logger.info("="*60)
        logger.info("PHASE 1: Joint Analysis")
        logger.info("="*60)
        
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
            logger.info(f"\nStep {step+1}/{self.config.max_steps}")
            
            # Get action from policy
            raw_action = policy.select_action(obs)
            logger.debug(f"  Raw action: {raw_action}")
            
            # Process action based on joint type
            processed_action = self._process_action(raw_action, joint_type, joint_params)
            logger.debug(f"  Processed action: {processed_action}")
            
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