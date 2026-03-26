# DeepReach Training & ROS Integration Guide

Complete guide for setting up PyTorch/CUDA, training Dubins3D models, and integrating with ROS.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Training Dubins3D Model](#training-dubins3d-model)
4. [Coordinate Normalization](#coordinate-normalization)
5. [Integration with ROS](#integration-with-ros)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Verify Environment

```bash
python setup_environment.py
```

This checks:
- ✓ Python version (3.8+)
- ✓ PyTorch installation
- ✓ CUDA/GPU availability
- ✓ All dependencies
- ✓ Project structure

### 2. Train Model

```bash
# Basic: Avoid mode (reach goal while avoiding obstacles)
python train_dubins3d.py --goalR 0.25 --velocity 0.6 --omega_max 1.1

# Custom: 10,000 epochs on GPU
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --angle_alpha_factor 1.2 \
  --set_mode avoid \
  --num_epochs 10000 \
  --device cuda:0

# Reach mode: Find path to goal
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --set_mode reach \
  --num_epochs 5000
```

### 3. Use in Python

```python
from coordinate_normalizer import create_normalizer_from_bounds
import torch
import numpy as np

# Create normalizer for your workspace
normalizer = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    theta_range=(-np.pi, np.pi),
    angle_alpha_factor=1.2
)

# Convert ROS coordinates to network input
ros_state = [2.5, -1.3, 0.7]  # [x, y, theta]
normalized = normalizer.normalize_state(ros_state, time=0.0)
# Output: [0.5, 0.0, 0.5, 0.58]  (all in [-1, 1])

# Use with network
with torch.no_grad():
    network_input = torch.from_numpy(normalized).float()
    model_output = model({'coords': network_input})
```

---

## Environment Setup

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** (package manager)
- **NVIDIA GPU** (optional, but strongly recommended)
  - CUDA Compute Capability 3.5+
  - NVIDIA Driver supporting CUDA 12.1+

### Step 1: Create Virtual Environment

```bash
# Option A: Python venv
python -m venv env
source env/Scripts/activate  # Windows
# or
source env/bin/activate      # Linux/Mac

# Option B: Conda
conda create -n deepreach python=3.10
conda activate deepreach
```

### Step 2: Install PyTorch

```bash
# CUDA 12.1 (recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (slow)
pip3 install torch torchvision torchaudio
```

Check PyTorch installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected output (with GPU):
```
2.2.1+cu121
True
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Environment

```bash
python setup_environment.py
```

Example successful output:
```
======================================================================
  DeepReach Environment Verification Script
======================================================================

Platform: Windows 10
Architecture: AMD64

======================================================================
  Python Version
======================================================================

Python 3.10.12
✓ Python version is compatible (>=3.8)

======================================================================
  PyTorch Installation
======================================================================

✓ PyTorch 2.2.1+cu121 is installed

======================================================================
  CUDA/GPU Support
======================================================================

✓ CUDA is available
  CUDA Device Count: 1
  GPU 0: NVIDIA RTX 3090
  CUDA Version: 12.1
  cuDNN Version: 8900
✓ CUDA tensor operations working

...

======================================================================
  Summary
======================================================================

Python Version                 ✓ PASS
PyTorch                        ✓ PASS
CUDA/GPU                       ✓ PASS
Dependencies                   ✓ PASS
Requirements File              ✓ PASS
Project Structure              ✓ PASS
WandB (Optional)               ✓ PASS

======================================================================
✓ ENVIRONMENT READY FOR TRAINING
...
```

---

## Training Dubins3D Model

### Model Architecture

The SIREN network learns the Hamilton-Jacobi value function:

```
Input Layer:     [t, x, y, θ] ∈ [-1, 1]⁴
                 ↓
Hidden Layer 1:  512 neurons, sin(30·) activation
                 ↓
Hidden Layer 2:  512 neurons, sin(30·) activation
                 ↓
Hidden Layer 3:  512 neurons, sin(30·) activation
                 ↓
Output Layer:    1 neuron, linear (no activation)
                 ↓
Output:          V̂(t,x,y,θ) ≈ Value function
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goalR` | 0.25 | Goal region radius (meters) |
| `velocity` | 0.6 | Forward velocity (m/s) |
| `omega_max` | 1.1 | Max angular velocity (rad/s) |
| `angle_alpha_factor` | 1.2 | Angle normalization factor |
| `set_mode` | avoid | 'reach' or 'avoid' |
| `num_epochs` | 5000 | Training iterations |
| `num_hl` | 3 | Number of hidden layers |
| `num_nl` | 512 | Hidden layer width |
| `lr` | 2e-5 | Learning rate |
| `device` | cuda:0 | 'cuda:0' or 'cpu' |

### Training Examples

#### Example 1: Avoid Mode (Standard)

```bash
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --angle_alpha_factor 1.2 \
  --set_mode avoid \
  --num_epochs 5000 \
  --experiment_name dubins3d_avoid_nominal
```

**What this learns:**
- Safe states: Can reach goal while avoiding obstacles
- Unsafe states: Too close to enter zone

**Use case:** Navigation with safety constraints

#### Example 2: Reach Mode

```bash
python train_dubins3d.py \
  --goalR 0.25 \
  --set_mode reach \
  --num_epochs 10000 \
  --experiment_name dubins3d_reach_goal
```

**What this learns:**
- Time to reach goal from any state
- Optimal path to target

**Use case:** Goal-reaching behavior

#### Example 3: Fast Training (CPU)

```bash
python train_dubins3d.py \
  --num_epochs 500 \
  --device cpu \
  --experiment_name dubins3d_cpu_test
```

⚠️ **Note:** CPU training is 10-50× slower. Use GPU when possible.

### Training Output

Outputs saved in: `./runs/{experiment_name}/`

```
runs/
└── dubins3d_avoid_nominal/
    ├── config_03_19_2026_14_30.txt          # Configuration log
    └── training/
        ├── checkpoints/
        │   ├── model_current.pth            # Latest checkpoint
        │   ├── model_epoch_1000.pth         # Milestone checkpoints
        │   ├── model_epoch_2000.pth
        │   ├── ...
        │   └── model_final.pth              # Final trained model ✓
        ├── summaries/
        │   └── events.out.tfevents.*        # Tensorboard logs
        └── training_config.txt
```

### Load and Use Trained Model

```python
import torch
from utils.modules import SingleBVPNet
from dynamics.dynamics import Dubins3D

# Load dynamics
dynamics = Dubins3D(
    goalR=0.25,
    velocity=0.6,
    omega_max=1.1,
    angle_alpha_factor=1.2,
    set_mode='avoid',
    freeze_model=False
)

# Load model
model = SingleBVPNet(out_features=1, type='sine', in_features=4,
                    hidden_features=512, num_hidden_layers=3)
model.load_state_dict(torch.load('./runs/dubins3d_avoid_nominal/training/checkpoints/model_final.pth'))
model.eval()

# Use for inference
with torch.no_grad():
    # Normalized input: [t, x, y, theta] in [-1, 1]
    model_input = torch.tensor([[0.5, 0.0, 0.1, 0.3]]).float()
    result = model({'coords': model_input})
    V = result['model_out']  # Value function output
    print(f"Value function: {V.item():.4f}")
```

---

## Coordinate Normalization

### Why Normalization?

The neural network expects inputs in **normalized coordinates** $[-1, 1]$:

$$x_{norm} = \frac{x_{real} - x_{center}}{x_{scale}}$$

where
- $x_{center} = (x_{min} + x_{max})/2$
- $x_{scale} = (x_{max} - x_{min})/2$

### Quick Start

```python
from coordinate_normalizer import create_normalizer_from_bounds
import numpy as np

# Define your ROS workspace boundaries
normalizer = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),      # ROS coordinates: x ∈ [-5, 5] meters
    y_range=(-5.0, 5.0),      # ROS coordinates: y ∈ [-5, 5] meters
    theta_range=(-np.pi, np.pi),  # Heading: θ ∈ [-π, π] radians
    angle_alpha_factor=1.2     # From Dubins3D config
)

# Show configuration
normalizer.print_config()
```

### Convert ROS → Network

```python
# ROS coordinate (meters, radians)
ros_state = np.array([2.5, -1.3, 0.7])  # [x, y, θ]

# Normalize to network input
normalized = normalizer.normalize_state(ros_state, time=0.0)
# Output: array([0.5, 0.0, 0.5, 0.583333])
#                              ↑            ↑
#                            x_norm      theta_norm

print(f"Normalized: {normalized}")
# All values in [-1, 1] ✓
```

### Convert Network → ROS

```python
# Get network gradients (in normalized coordinates)
network_grads = np.array([0.5, -0.2, 0.1])  # [∂V/∂x_norm, ∂V/∂y_norm, ∂V/∂θ_norm]

# Convert to real-world gradients via chain rule
real_grads = normalizer.denormalize_gradient(network_grads)
# Output: array([0.0625, -0.0250, 0.2618])
#                ↑        ↑        ↑
#            ∂V/∂x    ∂V/∂y    ∂V/∂θ (in real world units)

print(f"Real gradients: ∂V/∂x={real_grads[0]:.4f}, ∂V/∂y={real_grads[1]:.4f}, ∂V/∂θ={real_grads[2]:.4f}")
```

### Full Workflow

```python
from coordinate_normalizer import create_normalizer_from_bounds
import torch
import numpy as np

# 1. Create normalizer for your workspace
normalizer = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    angle_alpha_factor=1.2
)

# 2. Current ROS state
ros_state = np.array([1.5, 2.3, -0.5])  # Real-world coordinates

# 3. Normalize for network
normalized_state = normalizer.normalize_state(ros_state, time=0.0)

# 4. Network inference
with torch.no_grad():
    network_input = torch.from_numpy(normalized_state).float().unsqueeze(0)
    network_output = model({'coords': network_input})
    
    # 5. Compute gradients (via autodiff or provided methods)
    # ... (network computes ∂V/∂x_norm, ∂V/∂y_norm, ∂V/∂θ_norm)

# 6. Convert gradients to real-world coordinates
real_world_gradients = normalizer.denormalize_gradient(network_gradients_normalized)

# 7. Use for control
optimal_control = compute_control_from_gradient(real_world_gradients)
```

### Handling Batch Operations

```python
# Multiple states at once
ros_states = np.array([
    [1.5, 2.3, -0.5],
    [0.0, 0.0, 0.0],
    [-2.1, 1.8, 1.2]
])  # Shape: (3, 3)

# Normalize all at once
normalized_states = normalizer.normalize_state(ros_states, time=0.0)
# Output shape: (3, 4) [time, x, y, theta for each state]

print(f"Batch normalized shape: {normalized_states.shape}")
# (3, 4)
```

---

## Integration with ROS

### ROS Node Template

Create a ROS node that uses the trained model for real-time control:

```python
#!/usr/bin/env python3
"""
DeepReach ROS Controller for Dubins3D Vehicle
Uses trained neural network for safe control synthesis.
"""

import rospy
import numpy as np
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from coordinate_normalizer import create_normalizer_from_bounds
from utils.modules import SingleBVPNet

class DeepReachController:
    """ROS node for DeepReach-based control."""
    
    def __init__(self):
        rospy.init_node('deepreach_controller')
        
        # Load configuration
        self.goal_r = rospy.get_param('~goalR', 0.25)
        self.velocity = rospy.get_param('~velocity', 0.6)
        self.omega_max = rospy.get_param('~omega_max', 1.1)
        self.model_path = rospy.get_param('~model_path', 
                                         './runs/dubins3d_avoid_nominal/training/checkpoints/model_final.pth')
        
        # Setup normalizer for your workspace
        self.normalizer = create_normalizer_from_bounds(
            x_range=(-10.0, 10.0),  # Adjust to your workspace
            y_range=(-10.0, 10.0),
            angle_alpha_factor=1.2
        )
        
        # Load neural network
        self.model = SingleBVPNet(out_features=1, type='sine', in_features=4,
                                 hidden_features=512, num_hidden_layers=3)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        # Current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # ROS subscribers and publishers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Control loop rate
        self.rate = rospy.Rate(10)  # 10 Hz
        
    def odom_callback(self, msg):
        """Update current state from odometry."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self.theta = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2 + q.z**2))
    
    def compute_control(self):
        """Compute control using trained network."""
        
        # 1. Normalize current state
        ros_state = np.array([self.x, self.y, self.theta])
        normalized = self.normalizer.normalize_state(ros_state, time=0.0)
        
        # 2. Network forward pass and gradient computation
        with torch.no_grad():
            network_input = torch.from_numpy(normalized).float().unsqueeze(0)
            result = self.model({'coords': network_input})
            
            # Get gradient via autodiff
            from torch.autograd import grad
            dV_dnorm = grad(result['model_out'].sum(), network_input, 
                           create_graph=False)[0]
            
            # Convert to real-world gradient
            dV_real = self.normalizer.denormalize_gradient(dV_dnorm[0, 1:].numpy())
        
        # 3. Compute optimal control
        # For Dubins avoid: u* = ω_max · sign(∂V/∂θ)
        optimal_omega = self.omega_max * np.sign(dV_real[2])
        
        # 4. Publish control
        cmd = Twist()
        cmd.linear.x = self.velocity
        cmd.angular.z = optimal_omega
        self.cmd_vel_pub.publish(cmd)
        
        rospy.loginfo(f"State: ({self.x:.2f}, {self.y:.2f}, {self.theta:.2f}) "
                     f"→ Control: v={self.velocity:.2f}, ω={optimal_omega:.2f}")
    
    def run(self):
        """Main control loop."""
        while not rospy.is_shutdown():
            self.compute_control()
            self.rate.sleep()

if __name__ == '__main__':
    controller = DeepReachController()
    controller.run()
```

### ROS Launch File

Create `deepreach_control.launch`:

```xml
<launch>
    <!-- DeepReach Controller Node -->
    <node name="deepreach_controller" pkg="your_robot_pkg" type="deepreach_controller.py">
        <!-- Model path -->
        <param name="model_path" 
               value="$(find your_robot_pkg)/models/dubins3d_final.pth"/>
        
        <!-- Dynamics parameters -->
        <param name="goalR" value="0.25"/>
        <param name="velocity" value="0.6"/>
        <param name="omega_max" value="1.1"/>
        
        <!-- Topics -->
        <remap from="/odom" to="/robot/odom"/>
        <remap from="/cmd_vel" to="/robot/cmd_vel"/>
    </node>
    
    <!-- Visualization (optional) -->
    <node name="rviz" pkg="rviz" type="rviz"
          args="-d $(find your_robot_pkg)/config/deepreach.rviz"/>
</launch>
```

Run with:
```bash
roslaunch your_robot_pkg deepreach_control.launch
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce model size
python train_dubins3d.py --num_nl 256 --num_hl 2

# Or use CPU
python train_dubins3d.py --device cpu
```

### Issue: "PyTorch not found"

**Solution:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Training converges slowly"

**Solutions:**
1. Increase learning rate: `--lr 5e-5`
2. Reduce model size: `--num_hl 2`
3. Enable pretraining: `--pretrain`
4. Use curriculum learning: `--counter_end 100`

### Issue: "Gradients are unstable"

**Verify:**
- Using SIREN (sine activation): `--model sine` ✓
- Network architecture matches Dubins3D requirements
- State normalization is correct

Run:
```python
normalizer.print_config()
```

### Issue: "Normalization mismatch"

**Verify bounds match your ROS coordinate system:**
```python
normalizer = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),      # ← Check these match your workspace!
    y_range=(-5.0, 5.0),
    angle_alpha_factor=1.2      # ← Must match training config
)
normalizer.print_config()
```

---

## Advanced Topics

### Custom Workspace Bounds

Adjust normalizer for your specific ROS environment:

```python
# Small workspace (indoor robot)
normalizer = create_normalizer_from_bounds(
    x_range=(-2.0, 2.0),
    y_range=(-2.0, 2.0),
)

# Large workspace (outdoor vehicle)
normalizer = create_normalizer_from_bounds(
    x_range=(-50.0, 50.0),
    y_range=(-50.0, 50.0),
)

# Asymmetric workspace
normalizer = create_normalizer_from_bounds(
    x_range=(0.0, 10.0),   # Only positive x
    y_range=(-5.0, 5.0),   # Symmetric y
)
```

### Benchmarking

```bash
# Time a training run on your hardware
time python train_dubins3d.py --num_epochs 1000 --num_nl 256
```

Compare GPU vs CPU:
```bash
# GPU
python train_dubins3d.py --device cuda:0 --num_epochs 1000

# CPU  
python train_dubins3d.py --device cpu --num_epochs 1000
```

### Visualization

Monitor training with Tensorboard:
```bash
tensorboard --logdir=./runs/dubins3d_avoid_nominal/training/summaries
```

Open browser to `http://localhost:6006`

---

## Summary

You now have:
1. ✓ **setup_environment.py** - Verify PyTorch/CUDA setup
2. ✓ **train_dubins3d.py** - Train models with single command
3. ✓ **coordinate_normalizer.py** - Convert ROS ↔ network coordinates
4. ✓ **model_final.pth** - Ready-to-use trained model
5. ✓ **ROS integration** - Use in real robotic systems

**Next steps:** Train, verify normalization, deploy to ROS!
