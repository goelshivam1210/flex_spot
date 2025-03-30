import pybullet as p
import pybullet_data
import time
import numpy as np

# Configuration parameters
MAX_FORCE = 20.0  # Maximum force (η)
FRICTION = 0.5
SIMULATION_STEP = 1./240.

# Connect to PyBullet in GUI mode.
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(SIMULATION_STEP)
p.setGravity(0, 0, -9.8)

# Load the plane and the box
plane_id = p.loadURDF("plane.urdf")
start_pos = [0, 0, 0.1]
start_ori = p.getQuaternionFromEuler([0, 0, 0])
box_id = p.loadURDF("cube.urdf", start_pos, start_ori, globalScaling=0.5)

# Change dynamics for friction (remove angularFactor)
p.changeDynamics(box_id, -1, lateralFriction=FRICTION, angularDamping=1)

def get_box_position():
    pos, _ = p.getBasePositionAndOrientation(box_id)
    return np.array(pos[:2])

print("Simulation started. Use arrow keys to apply force. Press ESC to exit.")

while True:
    keys = p.getKeyboardEvents()
    force = np.array([0.0, 0.0, 0.0])
    
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        force[0] -= MAX_FORCE
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        force[0] += MAX_FORCE
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        force[1] += MAX_FORCE
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        force[1] -= MAX_FORCE
        
    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
        break

    if np.linalg.norm(force) > 0:
        pos, _ = p.getBasePositionAndOrientation(box_id)
        p.applyExternalForce(box_id, -1, force.tolist(), pos, p.WORLD_FRAME)
    
    p.stepSimulation()
    # Lock rotation: reset orientation to no rotation (identity quaternion)
    pos, _ = p.getBasePositionAndOrientation(box_id)
    p.resetBasePositionAndOrientation(box_id, pos, [0, 0, 0, 1])
    
    time.sleep(SIMULATION_STEP)

p.disconnect()