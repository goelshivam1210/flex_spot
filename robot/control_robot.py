import argparse
import sys
import time
import numpy as np
import cv2
import pickle
from scipy.spatial.transform import Rotation

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, robot_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME, get_vision_tform_body
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration


class PrismaticForcePolicy:
    """
    Implementation of a force-based policy for prismatic object manipulation,
    adapted from the PrismaticEnv simulation environment.
    
    This policy applies forces to move an object along a changing prismatic axis,
    converting simulation-learned policies to commands for the SPOT robot.
    """
    def __init__(self, config):
        self.config = config
        self.max_force = config.max_force if hasattr(config, 'max_force') else 200.0
        self.policy = None
        self.joint_axis = np.array([1.0, 0.0])  # Default axis along x
        self.initial_pos = None
        self.goal_pos = np.array([config.target_x, config.target_y]) if hasattr(config, 'target_x') else np.array([1.0, 0.0])
        
        # Load policy if specified
        if hasattr(config, 'policy_file') and config.policy_file:
            self.load_policy(config.policy_file)
    
    def load_policy(self, policy_file):
        """
        Load a policy from a file. The policy can be a neural network, lookup table,
        or any other form that maps state to action.
        """
        try:
            with open(policy_file, 'rb') as f:
                self.policy = pickle.load(f)
            print(f"Loaded policy from {policy_file}")
        except Exception as e:
            print(f"Error loading policy: {e}")
            print("Falling back to default (heuristic) policy")
    
    def initialize(self, initial_position):
        """Initialize the policy with the starting position of the object"""
        self.initial_pos = np.array(initial_position[:2])  # Take just x,y
        print(f"Policy initialized with starting position: {self.initial_pos}")
    
    def get_action(self, current_position):
        """
        Get the next action based on the current state.
        
        Args:
            current_position: Current (x,y,z) position of the object
            
        Returns:
            force_direction: 3D unit vector for force direction
            force_magnitude: Force magnitude (0-1, will be scaled by max_force)
        """
        # Extract 2D position and compute displacement from initial position
        current_pos_2d = np.array(current_position[:2])
        displacement = current_pos_2d - self.initial_pos
        
        # Create state vector as used in the simulation env: [joint_axis, displacement]
        state = np.concatenate([self.joint_axis, displacement])
        
        # If we have a learned policy, use it
        if self.policy is not None:
            try:
                # Get the action from the policy
                action = self.policy.predict(state.reshape(1, -1))[0]
                force_dir = np.array(action[:2])
                force_scale = action[2]
                
                # Normalize the force direction
                norm = np.linalg.norm(force_dir)
                if norm > 0:
                    force_dir = force_dir / norm
                else:
                    force_dir = self.joint_axis
                
                # Update the joint axis for next iteration
                self.joint_axis = force_dir.copy()
                
                # Return 3D force direction and magnitude
                force_direction = np.array([force_dir[0], force_dir[1], 0.0])
                force_magnitude = float(np.clip(force_scale, 0, 1))
                
                return force_direction, force_magnitude
                
            except Exception as e:
                print(f"Error executing policy: {e}")
                print("Falling back to heuristic policy")
        
        # Heuristic policy as a fallback
        return self._heuristic_policy(current_pos_2d, displacement)
    
    def _heuristic_policy(self, current_pos_2d, displacement):
        """
        Simple heuristic policy that moves the object toward the goal.
        Used as a fallback when a learned policy isn't available or fails.
        """
        # Vector from current position to goal
        to_goal = self.goal_pos - current_pos_2d
        dist_to_goal = np.linalg.norm(to_goal)
        
        # If we're very close to the goal, use a small force
        force_scale = min(1.0, dist_to_goal / 0.5)  # Scale down force when close
        
        if dist_to_goal > 0.01:  # If not already at goal
            # Normalize the direction vector
            force_dir = to_goal / dist_to_goal
        else:
            # If at goal, maintain previous direction
            force_dir = self.joint_axis
        
        # Update the joint axis for next iteration
        self.joint_axis = force_dir.copy()
        
        # Return 3D force direction and magnitude
        force_direction = np.array([force_dir[0], force_dir[1], 0.0])
        return force_direction, force_scale


def verify_estop(robot):
    """Verify the robot is not estopped"""
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    if estop_client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception("Robot is estopped. Please release E-Stop to proceed.")


def detect_object(image_data, target_dimensions=None, detection_method="color", color_range=None):
    """
    Detect objects in the image and return the centroid.
    
    Args:
        image_data: Raw image data from SPOT camera
        target_dimensions: (width, height) in meters if known
        detection_method: Method for detection ("color", "contour", "size")
        color_range: HSV color range for detection (low_hsv, high_hsv)
        
    Returns:
        pixel_x, pixel_y coordinates of the centroid
    """
    # Convert image data to OpenCV format
    decoded_image = np.frombuffer(image_data, dtype=np.uint8)
    cv_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
    
    if detection_method == "color":
        # Default color range if not specified
        if color_range is None:
            # Default to detecting reddish objects
            lower_bound = np.array([0, 100, 100])
            upper_bound = np.array([10, 255, 255])
        else:
            lower_bound, upper_bound = color_range
            
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (presumably our object)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            # Calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
    
    # Default to center if detection fails
    height, width = cv_image.shape[:2]
    return width // 2, height // 2


def force_to_pose_delta(force_direction, force_magnitude, max_force, dt=0.1, compliance=0.0005):
    """
    Convert a force vector to a position delta for the end effector.
    
    Args:
        force_direction: Unit vector (3D) indicating force direction
        force_magnitude: Force magnitude scale (0-1)
        max_force: Maximum force in Newtons
        dt: Time step in seconds
        compliance: Compliance factor (higher = more movement per unit force)
        
    Returns:
        Position delta vector (3D)
    """
    # Scale the force
    force = force_direction * force_magnitude * max_force
    
    # Convert force to position change (F = k*x --> x = F/k)
    # Higher compliance means the object moves more per unit force
    position_delta = force * compliance
    
    # Optional: limit maximum movement per step
    max_delta = 0.02  # 2cm maximum movement per step
    norm = np.linalg.norm(position_delta)
    if norm > max_delta:
        position_delta = position_delta * (max_delta / norm)
    
    return position_delta


def arm_pose_command(robot_command_client, position, orientation, frame_name=VISION_FRAME_NAME):
    """
    Create and send an arm pose command to the robot.
    
    Args:
        robot_command_client: Robot command client
        position: (x, y, z) position in meters
        orientation: Quaternion (w, x, y, z) or rotation matrix
        frame_name: Reference frame for the command
    """
    # Convert orientation to quaternion if it's a rotation matrix
    if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
        r = Rotation.from_matrix(orientation)
        quat = r.as_quat()  # Returns (x, y, z, w)
        # Reorder to (w, x, y, z) for the API
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        quat = orientation

    # Create hand pose
    hand_pose = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=position[0], y=position[1], z=position[2]),
        rotation=geometry_pb2.Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])
    )
    
    # Build the command
    arm_command = robot_command_pb2.ArmCommand.Request(
        pose_request=robot_command_pb2.ArmCommand.PoseRequest(
            pose_trajectory_in_task=robot_command_pb2.PoseTrajectory(
                points=[robot_command_pb2.PoseTrajectoryPoint(
                    pose=hand_pose,
                    time_since_reference=seconds_to_duration(0.0)
                )],
                reference_time=seconds_to_duration(time.time())
            ),
            root_frame_name=frame_name
        )
    )
    
    # Build the full robot command
    command = robot_command_pb2.RobotCommand(
        arm_command=arm_command
    )
    
    # Send the command to the robot
    cmd_id = robot_command_client.robot_command(command)
    return cmd_id


def execute_prismatic_policy(robot, command_client, robot_state_client, 
                             force_policy, duration=15.0, dt=0.2):
    """
    Execute a prismatic force-based policy.
    
    Args:
        robot: SPOT robot instance
        command_client: Robot command client
        robot_state_client: Robot state client
        force_policy: PrismaticForcePolicy instance
        duration: How long to execute the policy in seconds
        dt: Time step between commands in seconds
    """
    robot.logger.info("Executing prismatic force-based policy...")
    
    # Get initial hand pose
    robot_state = robot_state_client.get_robot_state()
    initial_hand_pos = None
    initial_hand_rot = None
    
    for frame in robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
        if frame.child_frame_name == "hand":
            initial_hand_pos = np.array([
                frame.parent_tform_child.position.x,
                frame.parent_tform_child.position.y,
                frame.parent_tform_child.position.z
            ])
            
            initial_hand_rot = np.array([
                frame.parent_tform_child.rotation.w,
                frame.parent_tform_child.rotation.x,
                frame.parent_tform_child.rotation.y,
                frame.parent_tform_child.rotation.z
            ])
            break
    
    if initial_hand_pos is None:
        robot.logger.error("Could not get hand pose, aborting policy execution")
        return
    
    # Initialize the policy with the current position
    force_policy.initialize(initial_hand_pos)
    
    # Main control loop
    start_time = time.time()
    last_position = initial_hand_pos.copy()
    
    while time.time() - start_time < duration:
        # Get current robot state
        robot_state = robot_state_client.get_robot_state()
        
        # Get current hand pose
        hand_position = None
        hand_rotation = None
        
        for frame in robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
            if frame.child_frame_name == "hand":
                # Extract position
                hand_position = np.array([
                    frame.parent_tform_child.position.x,
                    frame.parent_tform_child.position.y,
                    frame.parent_tform_child.position.z
                ])
                
                # Extract rotation (quaternion)
                hand_rotation = np.array([
                    frame.parent_tform_child.rotation.w,
                    frame.parent_tform_child.rotation.x,
                    frame.parent_tform_child.rotation.y,
                    frame.parent_tform_child.rotation.z
                ])
                break
                
        if hand_position is None:
            robot.logger.warn("Could not get hand pose, skipping policy iteration")
            time.sleep(dt)
            continue
            
        # Calculate displacement since last step
        displacement = hand_position - last_position
        last_position = hand_position.copy()
        
        # Get the next action from the policy
        force_direction, force_magnitude = force_policy.get_action(hand_position)
        
        # Convert force to position delta
        position_delta = force_to_pose_delta(
            force_direction, 
            force_magnitude, 
            force_policy.max_force, 
            dt=dt
        )
        
        # Calculate new target position
        new_position = hand_position + position_delta
        
        # Calculate distance to goal (in 2D plane)
        goal_pos_3d = np.array([force_policy.goal_pos[0], force_policy.goal_pos[1], hand_position[2]])
        distance_to_goal = np.linalg.norm(hand_position[:2] - force_policy.goal_pos)
        
        # Send the command to move the arm
        robot.logger.info(f"Applied force direction: {force_direction[:2]}, magnitude: {force_magnitude:.2f}")
        robot.logger.info(f"Position delta: {position_delta}, New position: {new_position}")
        robot.logger.info(f"Distance to goal: {distance_to_goal:.3f} meters")
        
        arm_pose_command(command_client, new_position, hand_rotation)
        
        # Check if we've reached the goal
        if distance_to_goal < 0.05:  # 5cm threshold
            robot.logger.info("Goal reached! Stopping policy execution.")
            break
        
        # Wait for the next iteration
        time.sleep(dt)


def grasp_and_manipulate_prismatic(robot, config):
    """Perform autonomous grasping and execute prismatic force-based policy"""
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client):
        robot.logger.info("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Failed to power on."

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)

        # 1. Detect and grasp the object
        robot.logger.info(f"Capturing image from: {config.image_source}")
        image_responses = image_client.get_image_from_sources([config.image_source])
        image = image_responses[0]

        # Process the image to find the object
        pixel_x, pixel_y = detect_object(
            image.shot.image.data, 
            target_dimensions=(config.object_width, config.object_height) if hasattr(config, 'object_width') else None,
            detection_method=config.detection_method,
            color_range=config.color_range if hasattr(config, 'color_range') else None
        )
        
        robot.logger.info(f"Object detected at pixel coordinates: ({pixel_x}, {pixel_y})")
        target_pixel = geometry_pb2.Vec2(x=pixel_x, y=pixel_y)

        # Set up the grasp request
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=target_pixel,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole,
        )

        # Configure grasp parameters
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        
        # Set the approach vector (top-down by default)
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            geometry_pb2.Vec3(x=1, y=0, z=0))  # gripper forward axis
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            geometry_pb2.Vec3(x=0, y=0, z=-1))  # aligned with gravity
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17  # ~10 degrees

        # Optional: grasp position offset if needed
        if hasattr(config, 'grasp_offset_z'):
            grasp.grasp_params.position_offset.CopyFrom(
                geometry_pb2.Vec3(z=config.grasp_offset_z))

        # Send the grasp command
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)
        cmd_response = manipulation_client.manipulation_api_command(grasp_request)
        cmd_id = cmd_response.manipulation_cmd_id

        # Wait for grasp to complete
        grasp_successful = False
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_id)
            feedback = manipulation_client.manipulation_api_feedback_command(feedback_request)
            state = feedback.current_state
            
            robot.logger.info("Grasp feedback state: %s",
                            manipulation_api_pb2.ManipulationFeedbackState.Name(state))
            
            # Check if we're done
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                grasp_successful = True
                break
            elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break
                
            time.sleep(0.25)

        # # 2. If grasp succeeded, execute the prismatic force policy
        # if grasp_successful:
        #     robot.logger.info("Grasp succeeded! Getting object centroid position...")
            
        #     # Get updated robot state to find end effector position
        #     robot_state = robot_state_client.get_robot_state()
            
        #     # Extract end effector pose for object centroid
        #     object_centroid = None
        #     for frame in robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
        #         if frame.child_frame_name == "hand":
        #             object_centroid = np.array([
        #                 frame.parent_tform_child.position.x,
        #                 frame.parent_tform_child.position.y,
        #                 frame.parent_tform_child.position.z
        #             ])
                    
        #             robot.logger.info(f"Object centroid (end effector position): "
        #                              f"x={object_centroid[0]}, "
        #                              f"y={object_centroid[1]}, "
        #                              f"z={object_centroid[2]}")
        #             break
            
        #     # 3. Apply the prismatic force policy
        #     if hasattr(config, 'use_force_policy') and config.use_force_policy:
        #         # Create force policy
        #         force_policy = PrismaticForcePolicy(config)
                
        #         # Execute the policy
        #         execute_prismatic_policy(
        #             robot, 
        #             command_client, 
        #             robot_state_client, 
        #             force_policy,
        #             duration=config.policy_duration
        #         )
            
        #     robot.logger.info("Manipulation sequence complete.")
        # else:
        #     robot.logger.info("Grasp failed. Aborting manipulation sequence.")

        # robot.logger.info("Powering off.")
        # robot.power_off(cut_immediately=False, timeout_sec=20)


def main():
    """Main function to parse arguments and run grasping and manipulation sequence"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', default='frontleft_fisheye_image',
                        help='Camera source for grasping')
    parser.add_argument('-w', '--object-width', type=float, 
                        help='Target object width in meters (optional)')
    parser.add_argument('-h', '--object-height', type=float, 
                        help='Target object height in meters (optional)')
    parser.add_argument('-d', '--detection-method', default='color',
                        choices=['color', 'contour', 'size'],
                        help='Object detection method')
    parser.add_argument('-o', '--grasp-offset-z', type=float, default=0.0,
                        help='Z-offset for grasping in meters')
    
    # Force policy parameters
    parser.add_argument('-f', '--use-force-policy', action='store_true',
                        help='Apply prismatic force space policy during manipulation')
    parser.add_argument('-p', '--policy-file', type=str,
                        help='Path to saved force policy parameters')
    parser.add_argument('-m', '--max-force', type=float, default=200.0,
                        help='Maximum force in Newtons')
    parser.add_argument('-t', '--policy-duration', type=float, default=15.0,
                        help='Duration to execute the force policy (seconds)')
    
    # Target position (goal)
    parser.add_argument('-x', '--target-x', type=float, default=0.5,
                        help='Target X position in meters')
    parser.add_argument('-y', '--target-y', type=float, default=0.0,
                        help='Target Y position in meters')
    
    args = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk("PrismaticForceGraspClient")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    verify_estop(robot)

    try:
        grasp_and_manipulate_prismatic(robot, args)
        return True
    except Exception as e:
        robot.logger.error(f"Error: {e}")
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)

# python spot_prismatic_policy.py \
#   --hostname SPOT_IP \
#   --image-source frontright_fisheye_image \
#   --use-force-policy \
#   --policy-file /path/to/your/policy.pkl \
#   --max-force 200.0 \
#   --policy-duration 15.0 \
#   --target-x 0.5 \
#   --target-y 0.3




the code that we have has everything we need, but I would like to do it step by step.
so lets maybe use the grasping code to let the user select a pixel in the image, and then the robot grasps it, and then we can generate the state of the object. So that we can start the policy transfer. Lets first write code do that I can tell you the dimensions of the object
length = 27 inches width = 22 inches height = 13 inches
so lets first generate the state so that we can use the policy, and then we will write and test code for the policy conversion to the end effector space.