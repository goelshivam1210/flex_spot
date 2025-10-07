import os
import numpy as np

from flex.alg import TD3 as DoorTD3
from flex.path_following_td3 import TD3 as PathTD3


class PolicyManager:
    """
    Manages TD3 policy loading and execution for different joint types.
    Handles the manipulation task by converting policy actions to robot commands.
    """
    
    def __init__(self, models_dir="models"):
        """
        Args:
            models_dir: Directory containing policy models
        """
        self.models_dir = models_dir
        self.current_policy = None
        self.current_joint_type = None
    
    def load_policy(self, joint_type):
        """
        Load TD3 policy for specified joint type (door opening).
        
        Args:
            joint_type: "prismatic" or "revolute"
            
        Returns:
            TD3 policy object
        """
        # Policy parameters
        state_dim = 6
        action_dim = 3
        max_action = 1.0
        
        # Construct model path: models/prismatic or models/revolute
        model_path = os.path.join(self.models_dir, joint_type)
        
        if not os.path.exists(f"{model_path}/final_actor.pth"):
            raise FileNotFoundError(f"Policy not found: {model_path}/final_actor.pth")
        
        # Create and load policy - Use DoorTD3 (NO max_torque)
        policy = DoorTD3(0.001, state_dim, action_dim, max_action)
        policy.load_actor(model_path, "final")  # Use model_path consistently
        
        self.current_policy = policy
        self.current_joint_type = joint_type
        print(f"→ Loaded {joint_type} policy from {model_path}")
        
        return policy
    
    def load_path_following_policy(self, policy_path="models/rotation", model_name="best_model"):
        """
        Load TD3 policy for path-following tasks (push/drag operations).
        
        Args:
            policy_path: Directory containing path-following policy models
            model_name: Name of the model to load (e.g., "best_model", "final_model")
            
        Returns:
            TD3 policy object for path-following
        """
        # Path-following policy parameters
        state_dim = 8
        action_dim = 3
        max_action = 1.0
        max_torque = 50.0
        
        if not os.path.exists(f"{policy_path}/{model_name}_actor.pth"):
            raise FileNotFoundError(f"Path-following policy not found: {policy_path}/{model_name}_actor.pth")
        
        # Create and load policy - Use PathTD3 (WITH max_torque)
        policy = PathTD3(0.0001, state_dim, action_dim, max_action, max_torque)
        policy.load_actor(policy_path, model_name)
        
        self.current_policy = policy
        self.current_joint_type = "path_following"
        print(f"→ Loaded path-following policy from {policy_path}/{model_name}")
        
        return policy