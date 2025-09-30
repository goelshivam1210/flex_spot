# flex/config.py
"""Configuration for FLEX policy execution system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FlexConfig:
    """Configuration for FLEX policy execution"""
    
    # Policy execution parameters
    max_steps: int = 50
    action_scale: float = 0.02
    
    # Success thresholds
    success_distance: float = 0.1    # meters, for prismatic joints
    success_angle: float = 60.0       # degrees, for revolute joints
    
    # Analysis parameters
    movement_distance: float = 0.05   # wiggle distance in meters
    force_joint_type: Optional[str] = None  # Optional override: 'prismatic' or 'revolute'
    
    # Policy model paths
    prismatic_policy_path: str = "models/prismatic"
    revolute_policy_path: str = "models/revolute"