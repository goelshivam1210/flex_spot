## Learning force-based controllers for rigid body movement along curved paths

This module trains path-following policies in MuJoCo simulation with dual force application methods for sim-to-real transfer to multi-robot systems.

---

### Environment

#### Path Following with Force Distribution
```text
Gym environment for autonomous path following using force-based control.

Goal:
    Navigate a ~65kg hollow plywood box along curved arc paths using distributed forces.
    Validate sim-to-real transfer by testing identical policies with different force methods.

State (8D Path-Relative):
    [lateral_error, longitudinal_error, orientation_error_discretized, progress, 
     deviation, speed_along_path, box_forward_x, box_forward_y]

Action (3D Wrench):
    [force_x, force_y, torque_z] ∈ [-1,1]³ → [±400N, ±400N, ±50Nm]

Force Application Methods:
    1. Centroid: Single wrench at center of mass (simulation ideal)
    2. Contact: Distributed forces at rear contact points (robot realistic)
```

---

### Training

#### Train Path Following Policy
```bash
python train.py --config config.yaml --seed 2
```

#### Manual Testing & Debugging
```bash
python keyboard.py
```
Controls: Arrow keys (translation), A/D (rotation), W/S/Z/X (force scaling), R (reset), Q (quit)

#### Validate Sim-to-Real Transfer
```bash
python test_dual_force.py --model_path runs/run-2-2025-06-10_02-12-32/models/best_model --episodes 3
```

---

### Components

#### Core Files
- `env.py`: Main path following environment with 8D state space
- `td3.py`: Dual-headed TD3 implementation 
- `train.py`: Training pipeline
- `config.yaml`: All hyperparameters and environment settings

#### Testing & Analysis
- `keyboard.py`: Manual control interface for debugging
- `test_dual_force.py`: Dual force application validation
- `runs/`: Saved models, logs, and training data

---

### Installation & Setup

#### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages: MuJoCo, PyTorch, NumPy, Gymnasium, TensorBoard, PyYAML

---

### Usage Examples

#### 1. Train a New Policy
```bash
python train.py --seed 42 --config config.yaml
```

#### 2. Test Manual Control
```bash
python keyboard.py
```

#### 3. Validate Trained Policy
```bash
python test_dual_force.py --model_path runs/latest/models/best_model
```

#### 4. Expected Results
```
CENTROID MODE: Reward=245.3, Success=1.00, Steps=127.2
CONTACT MODE:  Reward=239.1, Success=1.00, Steps=131.5
Similar performance - sim-to-real transfer validated!
```

---

### Results & Validation

A successfully trained policy demonstrates:
- High success rate (>95%) on curved path following
- Similar performance between centroid and contact force methods
- Ready for real robot deployment with identical control interface