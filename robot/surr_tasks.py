"""
Interactive Perception Phase 1

Integrates Spot robot control with interactive perception for object manipulation.
Grasp object based on user selection, print state values, return to rest.

Author: Shivam Goel
Date: July 2025
"""

import argparse
import time
import numpy as np
import cv2

from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.docking import DockingClient, blocking_dock_robot
from bosdyn.client.robot_command import blocking_stand


# Import existing Spot modules
from spot.spot import Spot, SpotPerception
from interactive_perception import InteractivePerception


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
        print(f"🎯 Generated {len(wiggle_positions)} wiggle positions")
        
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

            # Step 6: Release object and finish
            self.release_and_finish()
            
            # # Step 5: Release and return
            # self.release_and_return()
            
            print("\n Phase 1 completed successfully!")
            return True
            
        except Exception as e:
            print(f" Error during Phase 1 execution: {e}")
            return False


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
    
    args = parser.parse_args()
    
    # Create controller
    controller = InteractiveSpotController(
        hostname=args.hostname,
        dock_id=args.dock_id,
        image_source=args.image_source,
        depth_source=args.depth_source
    )
    controller.initialize_robot()
    
    # Run with lease management
    with LeaseKeepAlive(controller.spot.lease_client, must_acquire=True, return_at_exit=True):
        success = controller.run_phase1()
        
        if success:
            print("Interactive Perception Phase 1 completed successfully")
        else:
            print("Interactive Perception Phase 1 failed")


if __name__ == "__main__":
    main()