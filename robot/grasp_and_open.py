"""
Executes grasps and executes 
revolute or prismatic joint movements
based on the particular joint configuraton estimation

The code is divided into two phases:
1. Phase 1: Interactive perception to select grasp target and estimate joint type.
2. Phase 2: Policy-based manipulation using the estimated joint type.
As an additional step in phase 2, the robot will return to a clearance yaw pose
and then finally it will dock itself.

Example usage:
    python grasp_and_open.py --hostname 192.168.1.100 --max-steps 15 --action-scale 0.05

Author: Shivam Goel
Date: July 2025
"""
# Import necessary libraries
import argparse
import time
import numpy as np
import cv2
import json
import math

# Import Spot SDK modules
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.docking import DockingClient, blocking_dock_robot
from bosdyn.client.robot_command import blocking_stand


# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from interactive_perception import InteractivePerception

# Import FLEX modules
from alg import TD3
from policy_manager import PolicyManager


class InteractiveSpotController:
    """
    Main controller that integrates Spot robot with interactive perception.
    """
    
    def __init__(self, hostname, dock_id=521, image_source="hand_color_image", 
                 depth_source="hand_depth_in_hand_color_frame"):
        # Initialize robot
        self.spot = Spot(id="Interactive_Spot", hostname=hostname)
        self._docking_client = None
        self.dock_id = dock_id

        
        # Camera sources
        self.image_source = image_source
        self.depth_source = depth_source
        
        # Interactive perception module
        self.interactive_perception = InteractivePerception()
        
        # State tracking
        self.initial_pose = None
        self.grasp_position = None
        self.target_pixel = None
        
    def initialize_robot(self):
        """Start robot, power on, and stand up."""
        print("Initializing robot...")
        self.spot.start()
        self.spot.power_on()
        self.spot.stand_up()
        
        # Save initial pose for return
        self.initial_pose = self.spot.get_current_pose()
        print(f"Robot initialized at pose: {self.initial_pose}")
        
        # Initialize docking client
        try:
            self._docking_client = self.spot._client._spot.ensure_client(DockingClient.default_service_name)
            print("Docking client initialized")
        except Exception as e:
            print(f"Warning: Could not initialize docking client: {e}")
            self._docking_client = None
            if self._docking_client is None:
                print("Docking client not available, docking will be skipped.")    
    def capture_and_select_target(self):
        """Capture image and let user select grasp target."""
        print("\n Capturing image for target selection...")
        
        # Take picture
        color_img, depth_img = self.spot.take_picture(
            color_src=self.image_source,
            depth_src=self.depth_source,
            save_images=True
        )
        
        if color_img is None:
            print("Failed to capture image")
            return False
            
        print("Please click on the object to grasp...")
        
        # Get user target selection
        self.target_pixel = SpotPerception.get_target_from_user(color_img)
        
        if self.target_pixel is None:
            print(" No target selected")
            return False
            
        print(f"Target selected at pixel: {self.target_pixel}")
        
        return True
        
    def execute_grasp(self):
        """Execute grasp at selected target."""
        print("\n Executing grasp...")
        
        # Open gripper first
        self.spot.open_gripper()
        
        # Execute grasp at user-selected pixel
        success = self.spot.grasp_edge(self.target_pixel, img_src=self.image_source)
        
        if success:
            print(" Grasp executed successfully")
            
            # Get current gripper position (approximate from robot state)
            current_pose = self.spot.get_current_pose()
            self.grasp_position = np.array([current_pose[0], current_pose[1], 0.0])  # Approximate
            print(f" Estimated grasp position: {self.grasp_position}")
            
            return True
        else:
            print("Grasp failed")
            return False
            
    def print_state_values(self):
        """Print current state information."""
        print("\n Current State Values:")
        print("=" * 40)
        
        # Robot pose
        current_pose = self.spot.get_current_pose()
        print(f" Current robot pose: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, yaw={current_pose[2]:.3f}")
        print(f" Initial robot pose: x={self.initial_pose[0]:.3f}, y={self.initial_pose[1]:.3f}, yaw={self.initial_pose[2]:.3f}")
        
        # Target information
        print(f"Selected target pixel: {self.target_pixel}")
        
        if self.grasp_position is not None:
            print(f"Estimated grasp position: {self.grasp_position}")
            
            # Calculate displacement from initial position (for future use)
            initial_pos = np.array([self.initial_pose[0], self.initial_pose[1], 0.0])
            displacement = self.grasp_position - initial_pos
            print(f" Displacement from initial: {displacement}")
            
        # Interactive perception status
        print(f" Joint type estimate: {self.interactive_perception.joint_type}")
        print(f" Joint parameters: {self.interactive_perception.joint_params}")
        
        print("=" * 40)

    def perform_wiggle_movements(self):
        """Perform wiggle movements to collect trajectory data for joint estimation."""
        print("\n Starting wiggle movements for trajectory analysis...")
        
        # Get current robot state and hand position
        robot_state_client = self.spot._client._state_client
        robot_state = robot_state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        
        # Get current hand pose in vision frame as starting point
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        
        if vision_T_hand is None:
            print("Could not get hand pose")
            return None
            
        start_position = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
        print(f"Starting hand position: {start_position}")
        
        # Generate wiggle positions using InteractivePerception
        wiggle_positions = self.interactive_perception.generate_wiggle_positions(start_position)
        print(f" Generated {len(wiggle_positions)} wiggle positions")
        
        # Collect actual trajectory data by moving the arm
        trajectory = []
        
        for i, target_pos in enumerate(wiggle_positions):
            print(f" Moving to position {i+1}/{len(wiggle_positions)}: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            
            try:
                # Create SE3Pose for target position (keeping original orientation)
                target_pose = SE3Pose(
                    x=target_pos[0], 
                    y=target_pos[1], 
                    z=target_pos[2],
                    rot=vision_T_hand.rot  # Keep original orientation
                )
                
                # Build arm pose command
                arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                    target_pose.to_proto(), 
                    VISION_FRAME_NAME, 
                    seconds=2.0  # 2 second movement time
                )
                
                # Send command and wait for completion
                cmd_id = self.spot._client._command_client.robot_command(arm_cmd)
                block_until_arm_arrives(self.spot._client._command_client, cmd_id, timeout_sec=3.0)
                
                # Get actual achieved position
                time.sleep(0.5)  # Brief pause to settle
                robot_state = robot_state_client.get_robot_state()
                snapshot = robot_state.kinematic_state.transforms_snapshot
                actual_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
                
                if actual_hand_pose:
                    actual_position = np.array([actual_hand_pose.x, actual_hand_pose.y, actual_hand_pose.z])
                    trajectory.append(actual_position)
                    print(f"Achieved position: [{actual_position[0]:.3f}, {actual_position[1]:.3f}, {actual_position[2]:.3f}]")
                else:
                    print(f"Could not get actual position for movement {i+1}")
                    trajectory.append(target_pos)  # Fallback to target
                    
            except Exception as e:
                print(f" Error during movement {i+1}: {e}")
                trajectory.append(target_pos)  # Fallback to target
        
        # Convert to numpy array
        trajectory = np.array(trajectory)
        print(f"Collected trajectory with {len(trajectory)} actual positions")
        
        return trajectory
    
    def analyze_joint_and_print_state(self, trajectory):
        """Analyze trajectory to estimate joint parameters and print complete state."""
        print("\n Analyzing trajectory to estimate joint parameters...")
        
        # Analyze trajectory and estimate joint type
        joint_type, joint_params = self.interactive_perception.analyze_trajectory_and_estimate_joint(trajectory)
        
        print("\n Complete State Analysis:")
        print("=" * 50)
        
        # Robot pose information
        current_pose = self.spot.get_current_pose()
        print(f" Current robot pose: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, yaw={current_pose[2]:.3f}")
        print(f" Initial robot pose: x={self.initial_pose[0]:.3f}, y={self.initial_pose[1]:.3f}, yaw={self.initial_pose[2]:.3f}")
        
        # Grasp and target information
        print(f" Selected target pixel: {self.target_pixel}")
        print(f" Estimated grasp position: {self.grasp_position}")
        
        # Joint estimation results
        print(f" Estimated joint type: {joint_type}")
        print(f" Joint parameters: {joint_params}")
        
        # Construct state vector for policy (using initial position as reference)
        if self.grasp_position is not None:
            initial_pos = np.array([self.initial_pose[0], self.initial_pose[1], 0.0])
            try:
                state_vector = self.interactive_perception.construct_state_vector(
                    self.grasp_position, initial_pos
                )
                print(f"  State vector for policy: {state_vector}")
                print(f" State vector shape: {state_vector.shape}")
            except ValueError as e:
                print(f"  Could not construct state vector: {e}")
        
        print("=" * 50)

    def release_and_finish(self):
        """Release object and finish sequence."""
        print("\n Releasing object and finishing...")
        
        # Release gripper
        self.spot.open_gripper()
        time.sleep(1)

        print ("Docking")
        try:
            blocking_stand(self.spot._client._command_client, timeout_sec=10.0)

            dock_id = self.dock_id # hardcoded for now but should be in arg parser
            blocking_dock_robot(
                self.spot._client._spot, 
                dock_id=dock_id)
            
            print(" Robot docked successfully")

            # print ("closing gripper")
            # self.spot.close_gripper()
            time.sleep(1)

        except Exception as e:
            print(f" Error during docking: {e}")

        
        print(" Sequence complete - ready for policy execution (Phase 2)")

    def move_arm_to_position(self, target_position, timeout=3.0):
        """Move arm to target position for policy execution."""
        # Get current robot state for frame info
        robot_state = self.spot._client._state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        
        # Create target pose (keep current orientation)
        target_pose = SE3Pose(
            x=target_position[0],
            y=target_position[1], 
            z=target_position[2],
            rot=vision_T_hand.rot
        )
        
        # Send arm command
        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
            target_pose.to_proto(), VISION_FRAME_NAME, seconds=2.0
        )
        cmd_id = self.spot._client._command_client.robot_command(arm_cmd)
        block_until_arm_arrives(self.spot._client._command_client, cmd_id, timeout_sec=timeout)

    def check_success(self, initial_hand_pos, current_hand_pos, success_threshold):
        """Check if manipulation task succeeded based on joint type."""
        if self.interactive_perception.joint_type == "prismatic":
            distance = np.linalg.norm(current_hand_pos - initial_hand_pos)
            success = distance >= success_threshold['distance']
            print(f"Prismatic distance moved: {distance:.3f}m (threshold: {success_threshold['distance']:.3f}m)")
            return success
        
        elif self.interactive_perception.joint_type == "revolute":
            # Calculate angle using joint center
            joint_center = self.interactive_perception.joint_params['center']
            
            # Vectors from center to hand positions
            initial_vec = initial_hand_pos - joint_center
            current_vec = current_hand_pos - joint_center
            
            # Calculate angle between vectors
            cos_angle = np.dot(initial_vec, current_vec) / (np.linalg.norm(initial_vec) * np.linalg.norm(current_vec))
            angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
            angle_deg = np.degrees(angle_rad)
            
            success = angle_deg >= success_threshold['angle']
            print(f"Revolute angle rotated: {angle_deg:.1f}° (threshold: {success_threshold['angle']:.1f}°)")
            return success
        
        return False

    
        
    def run_phase1(self):
        """Execute Phase 1 of interactive perception."""
        print("Starting Interactive Perception Phase 1")
        print("=" * 50)
        
        try:
            # Step 1: Initialize robot
            # self.initialize_robot()
            self.spot.open_gripper()
            print (" Robot initialized and gripper opened")
            
            # Step 2: Capture image and get user target
            if not self.capture_and_select_target():
                return False
                
            # Step 3: Execute grasp
            if not self.execute_grasp():
                return False
                
            # Step 4: Print state values
            self.print_state_values()
            
            # Wait a moment to observe the grasp
            print("\n Holding position for 3 seconds...")
            time.sleep(3)

            trajectory = self.perform_wiggle_movements()
            if trajectory is None or len(trajectory) == 0:
                print(" No trajectory data collected")
                return False
            print(f"Collected trajectory with {len(trajectory)} points")    

            # Step 5: Analyze trajectory and print complete state
            self.analyze_joint_and_print_state(trajectory)

            # # Step 6: Release object and finish
            # self.release_and_finish()
            
            print("\n Phase 1 completed successfully!")
            return True
            
        except Exception as e:
            print(f" Error during Phase 1 execution: {e}")
            return False
        
    def run_phase2(self, max_steps=50, action_scale=0.02, success_threshold=None, clearance_yaw_offset=45):
        """Execute Phase 2: Policy-based manipulation."""
        print("\n Starting Phase 2: Policy Execution")
        print("=" * 50)
        
        # Initialize PolicyManager and load policy
        policy_manager = PolicyManager()
        joint_type = self.interactive_perception.joint_type
        radius = self.interactive_perception.joint_params['radius']
        
        try:
            policy = policy_manager.load_policy(joint_type)
        except FileNotFoundError as e:
            print(f" Policy loading failed: {e}")
            return False
        
        # Get initial hand position for state construction
        robot_state = self.spot._client._state_client.get_robot_state()
        snapshot = robot_state.kinematic_state.transforms_snapshot
        initial_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
        initial_hand_pos = np.array([initial_hand_pose.x, initial_hand_pose.y, initial_hand_pose.z])
        # get spots current yaw
        current_yaw = self.spot.get_current_pose()[2]
        # Execution loop with logging
        trajectory_log = []
        manipulation_success = False
        
        for step in range(max_steps):
            # Get current hand position
            robot_state = self.spot._client._state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            current_hand_pose = get_a_tform_b(snapshot, VISION_FRAME_NAME, "hand")
            current_hand_pos = np.array([current_hand_pose.x, current_hand_pose.y, current_hand_pose.z])
            
            # Construct current state vector
            state_vector = self.interactive_perception.construct_state_vector(
                current_hand_pos, initial_hand_pos
            )
            
            # Get action from policy
            action = policy.select_action(state_vector)
            # Flatten action if it's 2D
            if len(action.shape) > 1:
                action = action.flatten()
            action[0] = -action[0]  # Invert x action direction for Spot
            
            # Convert to target position
            target_position = current_hand_pos + (action * action_scale)
            
            print(f"Step {step+1}/{max_steps}: Action={action}, Target={target_position}")
            
            # Execute movement
            try:
                self.move_arm_to_position(target_position)
                
                # Log this step
                trajectory_log.append({
                    'step': step,
                    'state': state_vector.tolist(),
                    'action': action.tolist(),
                    'hand_position': current_hand_pos.tolist(),
                    'target_position': target_position.tolist()
                })
                
                # Check success
                if self.check_success(initial_hand_pos, current_hand_pos, success_threshold):
                    print(f" Task completed successfully in {step+1} steps!")
                    manipulation_success = True
                    break
                    
            except Exception as e:
                print(f"Error in step {step+1}: {e}")
                break
        
        # Print trajectory log summary
        print(f"\n Policy execution completed: {len(trajectory_log)} steps")
        print(f" Trajectory log: {trajectory_log}")

        # sleep for a moment to observe the final position
        print("\n Holding position for 2 seconds...")
        time.sleep(2)
        
        # Simple clearance for revolute joints
        if joint_type == "revolute":
            print(f" Revolute joint detected - realigning robot for clearance...")
            
            # Calculate clearance yaw (45 degrees offset from current position)
            clearance_yaw = current_yaw - math.radians(clearance_yaw_offset)
            self.spot.push_object(dx=2*radius, dy = -radius)
            
            # Use existing return_to_saved_yaw method
            # self.spot.return_to_saved_yaw(clearance_yaw)
            print(" Robot realigned for door clearance")
        
        return len(trajectory_log) > 0


def main():
    parser = argparse.ArgumentParser(description="Interactive Perception Phase 1")
    parser.add_argument("--hostname", required=True, help="Spot robot hostname or IP")
    parser.add_argument(
        "--image-source", 
        default="hand_color_image",
        help="Camera source for color images"
    )
    parser.add_argument(
        "--depth-source",
        default="hand_depth_in_hand_color_frame", 
        help="Camera source for depth images"
    )
    parser.add_argument(
    "--dock-id",
    type=int,
    default=521,
    help="Docking station ID"
)
    parser.add_argument(
    "--max-steps",
    type=int,
    default=50,
    help="Maximum policy execution steps"
)
    parser.add_argument(
        "--action-scale", 
        type=float,
        default=0.02,
        help="Scale factor for policy actions (meters)"
    )
    parser.add_argument(
        "--success-distance",
        type=float, 
        default=0.1,
        help="Success threshold distance for prismatic joints (meters)"
    )
    parser.add_argument(
        "--success-angle",
        type=float,
        default=30.0,
        help="Success threshold angle for revolute joints (degrees)"
    )
    
    args = parser.parse_args()
    
    # Create controller
    controller = InteractiveSpotController(
        hostname=args.hostname,
        dock_id=args.dock_id,
        image_source=args.image_source,
        depth_source=args.depth_source
    )
    controller.initialize_robot()
    
    with LeaseKeepAlive(controller.spot.lease_client, must_acquire=True, return_at_exit=True):
        success = controller.run_phase1()
        
        if success:
            print("Phase 1 completed successfully")
            
            # User confirmation for Phase 2
            input("\n Press Enter to start Phase 2 (Policy Execution)...")
            
            # Phase 2: Policy execution
            phase2_success = controller.run_phase2(
                max_steps=args.max_steps,
                action_scale=args.action_scale,
                success_threshold={
                    'distance': args.success_distance,
                    'angle': args.success_angle
                },
                clearance_yaw_offset=45  # Default clearance offset
            )
            
            if phase2_success:
                print(" Phase 2 completed successfully")
            else:
                print(" Phase 2 failed")

            # ask user if they want to release and finish
            input("\n Press Enter to proceed to Phase 3 (Release and Finish)...")
        
            # Phase 3: Release and finish
            controller.release_and_finish()
        else:
            print("Phase 1 failed")


if __name__ == "__main__":
    main()