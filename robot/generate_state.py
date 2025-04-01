import numpy as np
from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import world_object_pb2

class ObjectTracker:
    def __init__(self, target_tag_id):
        self.target_tag_id = target_tag_id
        self.initial_position = None

    def get_tag_position(self, world_objects):
        for obj in world_objects:
            if (obj.apriltag_properties.tag_id == self.target_tag_id and
                    obj.HasField("transforms_snapshot")):
                edge_map = obj.transforms_snapshot.child_to_parent_edge_map
                transform = list(edge_map.values())[0].parent_tform_child
                position = transform.position
                return np.array([position.x, position.y, position.z])
        return None

    def update_displacement(self, current_position):
        if self.initial_position is None:
            self.initial_position = current_position
            return np.array([0.0, 0.0])
        delta = current_position[:2] - self.initial_position[:2]
        return delta

def track_apriltag_displacement(hostname, tag_id):
    sdk = create_standard_sdk('AprilTagTracker')
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()

    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
    tracker = ObjectTracker(tag_id)

    for _ in range(10):
        world_objects = world_object_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]).world_objects
        pos = tracker.get_tag_position(world_objects)
        if pos is not None:
            displacement = tracker.update_displacement(pos)
            print(f"Position: {pos}, Displacement: {displacement}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', required=True)
    parser.add_argument('--tag-id', type=int, required=True)
    args = parser.parse_args()

    track_apriltag_displacement(args.hostname, args.tag_id)