import sys
print("--- Python sys.path ---")
for P_item in sys.path:
    print(P_item)
print("-----------------------")

try:
    import pybullet as p
    print("\n--- PyBullet Info From This Environment ---")
    if hasattr(p, '__file__'):
        print("File:", p.__file__)
    else:
        print("p.__file__ not found")

    if hasattr(p, '__version__'):
        print("__version__ string:", p.__version__)
    else:
        print("__version__ string attribute not found.")

    if hasattr(p, 'getAPIVersion'):
        print("getAPIVersion():", p.getAPIVersion())
    else:
        print("p.getAPIVersion() not found.")

    # This is just a constant, its value doesn't indicate build time
    # but its presence confirms the module loaded somewhat.
    # The actual build time is printed by PyBullet C++ on import.
    # print("A PyBullet Constant (B3G_LOAD_TEXTURE_STUB):", p.B3G_LOAD_TEXTURE_STUB) 

except ImportError:
    print("Could not import PyBullet.")
except Exception as e:
    print(f"An error occurred: {e}")