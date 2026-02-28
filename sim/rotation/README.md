# sim/rotation — Edge-Push Path-Following Policy (MuJoCo)

Trains a TD3 reinforcement learning policy to push a box along curved paths in MuJoCo simulation. The policy learns to output **2D forces (Fx, Fy)** applied from the box edge, producing both translation and rotation through the moment arm (r x F). The trained actor weights transfer directly to the real Spot robot via the [`react/`](../../robot/react/README.md) execution layer.

## Core Idea

A robot grasps one edge of a box and pushes it along a curved path. Instead of commanding force + torque independently, the policy only outputs a 2D force vector at the grasp point. The torque arises naturally from the cross product of the moment arm and the applied force — matching how the real robot physically interacts with the box through an edge grasp.

The policy receives a **6D egocentric state** that encodes path-tracking errors, velocities, and a physics feature (normalised reactive force), then outputs normalised force actions that steer the box along the desired path.

## Directory Structure

```
sim/rotation/
├── env.py                  # MuJoCo Gymnasium environment
├── td3.py                  # TD3 agent (Actor/Critic/ReplayBuffer)
├── train.py                # Training loop with TensorBoard logging
├── evaluate.py             # Systematic evaluation across conditions
├── config.yaml             # All hyperparameters (env, agent, training)
├── scene.xml               # MuJoCo scene definition (box + floor)
├── requirements.txt        # Python dependencies
│
├── resume_training.py      # Resume training from a checkpoint
├── keyboard.py             # Manual keyboard control for debugging
├── visualize.py            # Record videos of trained policies
├── record_three_videos.py  # Record short/full/random-arc videos
├── test_generalize.py      # Test on out-of-distribution path shapes (S-curve, meander)
├── test_dual_force.py      # Compare centroid vs edge force application
├── query_actions.py        # Inspect per-step action logs from evaluation
├── plot_results.py         # Plot training curves from TensorBoard logs
├── plot_trajectories.py    # Plot box trajectories vs planned paths
├── export_logs.py          # Export TensorBoard scalars to JSON
│
├── runs/                   # Training run outputs (models, logs, configs)
├── plots/                  # Generated plots
└── trajectory_plots_FINAL/ # Publication-ready trajectory figures
```

## Files in Detail

### `env.py` — SimplePathFollowingEnv

Gymnasium environment where a box follows an arc path driven by body-frame forces.

**State (6D egocentric):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `lateral_error` | Signed perpendicular distance to nearest path point (m) |
| 1 | `orientation_error` | Angular error between box heading and path tangent (rad) |
| 2 | `speed_fwd` | Velocity along path tangent (m/s) |
| 3 | `speed_lat` | Velocity along path normal (m/s) |
| 4 | `angular_vel` | Box yaw rate (rad/s) |
| 5 | `norm_reactive_force` | Physics feature: μmg / F_max, encodes mass/friction regime |

**Action:**
- **Edge mode** (`--push-from-edge`): 2D `[Fx, Fy]` applied at the edge; torque = r x F
- **Center mode** (default): 3D `[Fx, Fy, τz]` applied at the center of mass

**Domain randomization** (per episode): mass sampled from `mass_range`, friction from `friction_range`. The `norm_reactive_force` state feature explicitly encodes these parameters so the policy can condition its behaviour.

**Reward:** progress along path tangent (weighted by alignment), speed regulation toward 0.3 m/s target, lateral/orientation constraint penalties, and spin penalty. Terminal: +100 for reaching end of path, -500 for exceeding deviation tolerance.

**Paths:** circular arcs generated from `(arc_radius, arc_start, arc_end)`. Training uses short randomly-sampled segments; evaluation uses full arcs.

### `td3.py` — TD3 Agent

Twin Delayed DDPG with a **direction-magnitude Actor**:
- **Direction head**: outputs a unit 2D vector via `normalize(tanh(...))`
- **Magnitude head**: outputs a scalar in [0,1] via `sigmoid(...)`
- **Force output**: `direction * magnitude` — always a valid 2D force within the unit disk
- **Torque head** (optional): only instantiated when `action_dim >= 3`

Architecture: `Linear(state_dim, 400) → ReLU → Linear(400, 300) → ReLU → heads`

Also includes twin Critic networks, ReplayBuffer, and Polyak-averaged target networks.

### `train.py` — Training Loop

Runs episodic TD3 training with:
- Random exploration for `start_timesteps`, then policy + Gaussian noise
- Three evaluation environments run periodically (every `eval_freq` episodes):
  - **Short**: random short segments of the training arc
  - **Full**: complete arc traversal
  - **Gen**: random arc geometry (radius, sweep angle) for generalisation
- Optional **wide-range eval** with broader mass/friction ranges
- Best model saved by composite score: `0.5 * full_success + 0.5 * gen_success`
- TensorBoard logging of rewards, reward components, terminal events, timing, and eval metrics
- Full deterministic seeding (NumPy, PyTorch, CUDA, replay buffer, exploration)

```bash
python train.py --config config.yaml --seed 0
python train.py --config config.yaml --seed 0 --push-from-edge
```

### `evaluate.py` — Systematic Evaluation

Tests a trained policy across a matrix of conditions:
- **Arc types**: short segment, full arc
- **Directions**: clockwise, anticlockwise
- **In-distribution**: (5kg, 15kg) x (0.4, 0.6 friction)
- **Out-of-distribution**: 3kg, 20kg, 0.8 friction, and combinations

For each condition, runs up to `max_attempts` episodes and saves the best video (MP4) and a per-step action CSV. Produces a summary CSV and text report.

```bash
python evaluate.py --run_dir runs/run-0-2025-... --model best_model --max_attempts 5
```

### `config.yaml` — Hyperparameters

Three sections:
- **env**: physics (max_force, friction, mass ranges, damping), path (arc geometry, segment length, deviation tolerance), and mode flags (`push_from_edge`)
- **agent**: learning rate, batch size, gamma, polyak, exploration noise, replay buffer size
- **training**: episode count, eval frequency, save frequency, segment mode (short/full), optional wide eval

### `scene.xml` — MuJoCo Scene

Defines a box (0.465 x 0.61 x 0.63 m, 15 kg) with a free joint on a flat floor with checker-pattern texture. Contact parameters use `solimp` for realistic friction modelling.

### `keyboard.py` — Manual Control

Interactive keyboard interface for debugging the environment. Arrow keys apply forces, A/D apply torque, W/S and Z/X adjust force/torque scales. Useful for understanding box dynamics and tuning friction parameters.

```bash
python keyboard.py
```

### `test_generalize.py` — Path Shape Generalisation

Tests whether a policy trained on arcs can generalise to unseen path shapes:
- S-curves (single sinusoidal)
- Meander paths (dual-frequency oscillation)
- Triple-S paths

The environment's `full_path` is monkey-patched with the new geometry; everything else (physics, state/action space) stays identical.

```bash
python test_generalize.py --run_dir runs/run-0-... --model best_model
```

### `test_dual_force.py` — Force Application Comparison

Compares centroid (single wrench at COM) vs distributed contact forces on the box's rear face. Validates that training with centroid forces transfers to the physically-realistic contact force distribution.

### `visualize.py` / `record_three_videos.py` — Video Recording

Record MP4 videos of trained policies on short segments, full arcs, and random arcs with path markers rendered in the scene.

```bash
python visualize.py --run_dir runs/run-0-... --model best_model
```

### `query_actions.py` — Action Log Inspector

Query and display per-step action logs from evaluation videos. Shows forces, torques, state features, and aggregate statistics.

```bash
python query_actions.py --eval_dir runs/run-0-.../eval
python query_actions.py --eval_dir runs/run-0-.../eval --video short_low_mass_low_friction_clockwise_m5_f0.4_id --summary
```

### `resume_training.py` — Resume from Checkpoint

Resumes training from a saved checkpoint, preserving the replay buffer state and continuing TensorBoard logging.

### `plot_results.py` / `plot_trajectories.py` — Plotting

- `plot_results.py`: plots training curves (reward, success rate) from exported TensorBoard JSON
- `plot_trajectories.py`: plots box trajectories vs planned paths from evaluation runs

### `export_logs.py` — TensorBoard Export

Extracts all scalar data from a TensorBoard event file into JSON for external analysis or plotting.

```bash
python export_logs.py runs/run-0-.../logs/ > training_data.json
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (edge-push, 2D force output)
python train.py --config config.yaml --seed 0 --push-from-edge

# Evaluate
python evaluate.py --run_dir runs/run-0-... --model best_model

# Visualise
python visualize.py --run_dir runs/run-0-... --model best_model
```

## Sim-to-Real Transfer

The trained `best_model_actor.pth` is loaded by `robot/react/td3.py` and executed on the real Spot robot via `robot/policies/push.py`. The `StateEstimator` in `react/` reproduces the exact same 6D state computation from EEF (or box-center) pose, and the `ForceProber` provides the `norm_reactive_force` feature that the policy was trained to condition on.
