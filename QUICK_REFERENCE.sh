#!/bin/bash
# ============================================================================
# DeepReach Training Quick Reference
# ============================================================================
# Common commands for training Dubins3D models and generating model_final.pth
# ============================================================================

# =============================================================================
# STEP 1: VERIFY ENVIRONMENT
# =============================================================================

echo "Step 1: Verifying PyTorch/CUDA environment..."
python setup_environment.py


# =============================================================================
# STEP 2: QUICK START - AVOID MODE (Recommended for beginners)
# =============================================================================

echo "Step 2a: Quick training (avoid mode, shorter)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --angle_alpha_factor 1.2 \
  --set_mode avoid \
  --num_epochs 5000 \
  --experiment_name dubins3d_quick_start


# =============================================================================
# STEP 3: STANDARD TRAINING CONFIGURATIONS
# =============================================================================

# Option A: Avoid mode (most common)
echo "Option A: Avoid mode (reach goal safely)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --angle_alpha_factor 1.2 \
  --set_mode avoid \
  --num_epochs 10000 \
  --device cuda:0 \
  --num_hl 3 \
  --num_nl 512 \
  --experiment_name dubins3d_avoid_nominal


# Option B: Reach mode (goal finding)
echo "Option B: Reach mode (find path to goal)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --set_mode reach \
  --num_epochs 10000 \
  --device cuda:0 \
  --experiment_name dubins3d_reach_goal


# Option C: With pretraining (slower but sometimes better convergence)
echo "Option C: With pretraining"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --pretrain \
  --num_epochs 10000 \
  --device cuda:0 \
  --experiment_name dubins3d_with_pretrain


# =============================================================================
# STEP 4: ADVANCED CONFIGURATIONS
# =============================================================================

# Fast training (smaller network, fewer epochs) - GPU required
echo "Fast training (test configuration)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --num_epochs 3000 \
  --num_nl 256 \
  --num_hl 2 \
  --device cuda:0 \
  --lr 5e-5 \
  --experiment_name dubins3d_fast


# Large network training (requires more VRAM) - for production
echo "Large network training (production quality)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --num_epochs 20000 \
  --num_nl 1024 \
  --num_hl 4 \
  --device cuda:0 \
  --experiment_name dubins3d_large_network


# Different dynamics parameters
echo "Different dynamics (faster, tighter turning)"
python train_dubins3d.py \
  --goalR 0.15 \
  --velocity 1.0 \
  --omega_max 2.0 \
  --angle_alpha_factor 1.5 \
  --num_epochs 10000 \
  --device cuda:0 \
  --experiment_name dubins3d_fast_dynamics


# CPU-only training (for systems without GPU)
echo "CPU training (slow but works)"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --num_epochs 5000 \
  --device cpu \
  --num_nl 256 \
  --num_hl 2 \
  --experiment_name dubins3d_cpu


# =============================================================================
# STEP 5: LOCATE TRAINED MODEL
# =============================================================================

echo "Finding generated model_final.pth..."
find ./runs -name "model_final.pth" -type f


# Expected output:
# ./runs/dubins3d_quick_start/training/checkpoints/model_final.pth
# ./runs/dubins3d_avoid_nominal/training/checkpoints/model_final.pth
# etc.


# =============================================================================
# STEP 6: TEST COORDINATION NORMALIZATION
# =============================================================================

echo "Testing coordinate normalizer..."
python << 'EOF'
import numpy as np
from coordinate_normalizer import create_normalizer_from_bounds

# Create normalizer
normalizer = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    theta_range=(-np.pi, np.pi),
    angle_alpha_factor=1.2
)

# Print configuration
normalizer.print_config()

# Test normalization
ros_state = np.array([2.5, -1.3, 0.7])
normalized = normalizer.normalize_state(ros_state, time=0.5)

print("\nExample normalization:")
print(f"ROS state:       {ros_state}")
print(f"Normalized:      {normalized}")

# Test denormalization
denormalized = normalizer.denormalize_state(normalized)
print(f"Denormalized:    {denormalized[:3]}")
print(f"Match original?: {np.allclose(ros_state, denormalized[:3])}")
EOF


# =============================================================================
# STEP 7: MONITOR TRAINING WITH TENSORBOARD (in separate terminal)
# =============================================================================

echo "To monitor training in real-time, run in a separate terminal:"
echo "tensorboard --logdir=./runs/dubins3d_quick_start/training/summaries"
echo "Then open: http://localhost:6006"


# =============================================================================
# STEP 8: LOAD AND USE TRAINED MODEL
# =============================================================================

echo "Loading and using trained model..."
python << 'EOF'
import torch
import numpy as np
from utils.modules import SingleBVPNet
from dynamics.dynamics import Dubins3D
from coordinate_normalizer import create_normalizer_from_bounds

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

try:
    model.load_state_dict(
        torch.load('./runs/dubins3d_quick_start/training/checkpoints/model_final.pth')
    )
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Setup normalizer
    normalizer = create_normalizer_from_bounds(
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        angle_alpha_factor=1.2
    )
    
    # Test inference
    ros_state = np.array([1.0, 1.5, 0.5])
    normalized = normalizer.normalize_state(ros_state, time=0.0)
    
    with torch.no_grad():
        network_input = torch.from_numpy(normalized).float().unsqueeze(0)
        result = model({'coords': network_input})
        V = result['model_out'].item()
    
    print(f"\nTest inference:")
    print(f"  ROS state:   {ros_state}")
    print(f"  Value:       {V:.4f}")
    print(f"  Safe?        {'Yes' if V < 0 else 'No'}")
    
except FileNotFoundError:
    print("✗ Model not found. Run training first!")

EOF


# =============================================================================
# STEP 9: COPY MODEL TO ROS PACKAGE (Optional)
# =============================================================================

# Copy the trained model to your ROS package
cp ./runs/dubins3d_quick_start/training/checkpoints/model_final.pth \
   /path/to/your_ros_package/models/dubins3d_final.pth

echo "Model copied to ROS package!"


# =============================================================================
# STEP 10: TROUBLESHOOTING
# =============================================================================

# If CUDA out of memory:
echo "Fix: Reduce model size"
python train_dubins3d.py \
  --goalR 0.25 \
  --velocity 0.6 \
  --omega_max 1.1 \
  --num_hl 2 \
  --num_nl 256 \
  --experiment_name dubins3d_smaller_model


# If very slow training:
echo "Check: Are you using GPU?"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"


# If normalization issues:
echo "Debug: Print normalizer configuration"
python -c "
from coordinate_normalizer import create_normalizer_from_bounds
import numpy as np
norm = create_normalizer_from_bounds(
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0)
)
norm.print_config()
"


# =============================================================================
# USEFUL INFORMATION
# =============================================================================

echo "
╔════════════════════════════════════════════════════════════════════╗
║             DeepReach Training Quick Reference                     ║
╚════════════════════════════════════════════════════════════════════╝

Key Files:
  setup_environment.py        → Verify PyTorch/CUDA
  train_dubins3d.py          → Train models
  coordinate_normalizer.py   → Convert coordinates ROS ↔ network
  docs/TRAINING_AND_ROS_INTEGRATION_GUIDE.md → Full documentation

Generated Models:
  ./runs/{experiment_name}/training/checkpoints/model_final.pth

Neural Network Architecture:
  Input:  [t, x, y, θ] ∈ [-1, 1]⁴
  Output: V(t,x,y,θ) ∈ ℝ  (value function)
  Layers: 4 → 512 → 512 → 512 → 1
  Activation: sin(30·) [SIREN]

Training Time (approximate):
  GPU (RTX 3090, 10k epochs): ~20-40 minutes
  GPU (RTX 2080, 10k epochs): ~40-80 minutes
  CPU (no GPU, 5k epochs):   ~2-4 hours

Next Steps:
  1. python setup_environment.py
  2. python train_dubins3d.py --goalR 0.25 --velocity 0.6 --omega_max 1.1
  3. Check: ./runs/*/training/checkpoints/model_final.pth
  4. Use in ROS with coordinate_normalizer.py

Questions?
  See: docs/TRAINING_AND_ROS_INTEGRATION_GUIDE.md
  Or check code comments in train_dubins3d.py
"
