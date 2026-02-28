# react/ Robot Execution Layer

The `react/` package bridges simulation-trained TD3 policies to the real Spot robot. It provides the state estimation, path generation, force probing, and policy inference modules that `policies/push.py` orchestrates during a push manipulation episode.

## Module Overview

```
react/
├── state_estimator.py   # 6D egocentric state (matches sim exactly)
├── path_generator.py    # Arc, straight, S-curve, meander, triple-S paths
├── force_prober.py      # Impedance-based reactive-force estimation
├── td3.py               # TD3 Actor/Critic networks and inference
└── test_probing.py      # Standalone script for force-probe testing
```

## How `push.py` Uses react/

`policies/push.py` imports four components from this package and calls them in sequence:

```
push.py pipeline
─────────────────────────────────────────────────────
1. Grasp object (Spot SDK + perception)
2. ForceProber.probe()       → norm_reactive_force
3. PathGenerator.<type>()    → path_points [N×3]
4. TD3.load_actor()          → trained policy
5. StateEstimator.compute()  → 6D state  ─┐
   TD3.select_action(state)  → [Fx, Fy]   │  policy
   spot.push_object_from_sim(dx, dy, …)    │  loop
   ← repeat until success or abort ────────┘
6. Release, stow, dock
```

## Modules

### StateEstimator (`state_estimator.py`)

Produces the 6-dimensional state vector that the TD3 actor expects, mirroring the simulation environment exactly:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `lateral_error` | Signed perpendicular distance from reference to path (m) |
| 1 | `orientation_error` | Angular error between heading and path tangent (rad) |
| 2 | `speed_fwd` | Velocity projected onto path tangent (m/s) |
| 3 | `speed_lat` | Velocity projected onto path normal (m/s) |
| 4 | `angular_vel` | Yaw rate (rad/s) |
| 5 | `norm_reactive_force` | Normalised friction force μmg / F_max (constant per episode) |

The reference position is either the EEF (hand) or the estimated box center, depending on whether `box_dimensions` are provided. Box-center mode matches the simulation's box-centric state representation.

**Key methods:**
- `StateEstimator(init_hand_pos, init_yaw, init_time, norm_reactive_force, ...)` — initialise with grasp pose
- `compute(current_hand_pos, current_yaw, current_time, path_points)` — returns `np.ndarray(6,)`
- `reset(...)` — re-initialise for a new episode without constructing a new object

### PathGenerator (`path_generator.py`)

Generates 2D waypoint paths in the robot's local frame, then transforms them into the vision frame using the saved initial pose. All returned paths are `[N × 3]` float32 arrays (x, y, z) with z held constant at EEF height.

**Supported path types:**

| Path Type | Method | Description |
|-----------|--------|-------------|
| `arc` | `gen.arc(radius, start_angle, end_angle)` | Circular arc, curvature set by angle sign |
| `straight` | `gen.straight(length)` | Straight line along robot forward |
| `s_curve` | `gen.s_curve(length, amplitude)` | Single S-shaped sinusoidal |
| `meander` | `gen.meander(length, amplitude)` | Dual-frequency meandering |
| `triple_s` | `gen.triple_s(length, amplitude)` | Three consecutive S-curves |

The standalone `__main__` block allows previewing and saving paths to CSV without a robot:

```bash
python react/path_generator.py --path-type arc --arc-radius 1.5 --arc-angle-deg 60
python react/path_generator.py --path-type s_curve --length 3.0 --amplitude 0.5
```

### ForceProber (`force_prober.py`)

Estimates the normalised reactive force `μmg / F_max` that the policy needs as its 6th state feature. Two modes:

1. **Active probing** (`probe()`): ramps a virtual spring equilibrium forward via Spot's impedance API while sampling the wrist F/T sensor at ~50 Hz. Detects the static-to-kinetic friction breakaway as a force drop and returns the peak force normalised by `F_max`.
2. **Calibrated defaults** (`_use_default()`): lookup table indexed by `(grasp_strategy, surface_type)` for when active probing is skipped.

Also provides `settle()`, which holds the current hand pose with impedance for a given duration — used between push steps to let the gripper settle against the box.

### TD3 (`td3.py`)

Twin Delayed DDPG (TD3) implementation with a direction-magnitude Actor architecture:

- **Actor**: outputs a unit-direction 2D vector (force direction) multiplied by a scalar magnitude, plus an optional torque head when `action_dim >= 3`.
- **Critic**: twin Q-networks for value estimation during training.
- **ReplayBuffer**: simple ring buffer with random sampling.

At inference time only the actor is used. Weights are loaded from `models/react/<name>_actor.pth` via `load_actor(directory, name)`.

### test_probing.py

Standalone script for testing the force probing pipeline on the real robot without running a full push episode. Useful for calibrating `ForceProber` parameters or debugging F/T sensor readings.

## Usage from push.py

```python
from react.state_estimator import StateEstimator
from react.path_generator  import PathGenerator
from react.force_prober    import ForceProber
from react.td3 import TD3
```

All modules are imported at the top of `policies/push.py`. See the [main README](../README.md) for full push.py CLI usage and examples.
