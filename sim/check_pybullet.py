# check_pybullet.py
import pybullet as p
import pybullet_data
import sys

print("--- Python sys.path ---")
for P_item in sys.path:
    print(P_item)
print("-----------------------")

print("\n--- PyBullet Verification Script ---")
if hasattr(p, '__file__'):
    print("File:", p.__file__)
else:
    print("p.__file__ not found")

# This is the most reliable way to get the string version in modern PyBullet
if hasattr(p, '__version__'):
    print("__version__ string:", p.__version__)
else:
    print("__version__ string attribute not found.")

# getAPIVersion() is older, but we print it for comparison
if hasattr(p, 'getAPIVersion'):
    print("getAPIVersion():", p.getAPIVersion())
else:
    print("p.getAPIVersion() not found.")

# The build time is usually printed automatically by PyBullet's C++ component on import.

print("\nAttempting p.changeDynamics with angularFactor...")
cid = -1
plane_id = -1
try:
    cid = p.connect(p.DIRECT)
    print(f"Connected to PyBullet server with client ID: {cid}")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    print(f"Loaded plane.urdf with body ID: {plane_id}")

    p.changeDynamics(plane_id, -1, angularFactor=[0,0,0])
    print("SUCCESS: p.changeDynamics with angularFactor accepted.")
except TypeError as e:
    print(f"FAILURE: TypeError with changeDynamics: {e}")
except Exception as e:
    print(f"FAILURE: Other error with PyBullet operations: {e}")
finally:
    if cid != -1: 
        try:
            # Check connection status specific to the client ID
            if p.isConnected(physicsClientId=cid): 
                p.disconnect(physicsClientId=cid) 
                print(f"Disconnected from PyBullet server (client ID {cid}).")
            else:
                # This case might occur if connection failed initially but cid got assigned.
                print(f"PyBullet server (client ID {cid}) was not connected or already disconnected prior to finally block.")
        except Exception as disconnect_e:
            print(f"Error or warning during disconnect: {disconnect_e}")
            
print("---------------------------------------------")