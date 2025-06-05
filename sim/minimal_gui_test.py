# minimal_gui_test.py
import pybullet as p
import pybullet_data
import time

print("--- Minimal PyBullet GUI Test ---")
# Attempt to connect to PyBullet GUI
try:
    client_id = p.connect(p.GUI)
    if client_id < 0:
        print("Failed to connect to PyBullet GUI. Exiting.")
        exit()
    print(f"Successfully connected to PyBullet GUI with client ID: {client_id}")
except Exception as e:
    print(f"Error connecting to PyBullet GUI: {e}")
    exit()

# Configure the visualizer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# Load basic environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")
cube_id = p.loadURDF("cube_small.urdf", basePosition=[0, 0, 0.5])

print("Environment loaded. Stepping simulation for 10 seconds...")

# Simulation loop
try:
    for _ in range(240 * 10): # Run for 10 seconds (at 240Hz)
        p.stepSimulation()
        time.sleep(1./240.)
        # Check if user closed the window (not always reliable for direct stop)
        # events = p.getKeyboardEvents()
        # if p.B3G_ESCAPE in events and events[p.B3G_ESCAPE] & p.KEY_WAS_TRIGGERED:
        #     break
except p.error as e:
    print(f"PyBullet error during simulation: {e}")
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
finally:
    if p.isConnected(client_id):
        print("Disconnecting from PyBullet.")
        p.disconnect(client_id)
    else:
        print("PyBullet was already disconnected or connection failed.")
print("--- Minimal PyBullet GUI Test Complete ---")