"""
10D NarrowPassage Analysis
Model: runs/narrow_passage_10d_run_lr5/training/checkpoints/model_epoch_110000.pth
No grid-based ground truth (10D grid is computationally infeasible).

State: [x1, y1, θ1, v1, φ1, x2, y2, θ2, v2, φ2]
  x, y  : position in corridor  (x ∈ [-8,8], y ∈ [-3.8,3.8])
  θ     : heading                (θ ∈ [-π, π])
  v     : speed                  (v ∈ [-1, 7])
  φ     : steering angle         (φ ∈ [-0.3π, 0.3π])

Environment:
  Corridor curbs at y = -2.8 and y = +2.8
  Stranded car at (0, -1.8)
  Car length L = 2.0

Outputs (all in baselines/plots/narrow_passage_10d/):
  value_slice_car1_pos.png          — V(x1,y1) heatmap, car 2 at (6, 1.4)
  value_slice_multi.png             — 4-panel: different car-2 positions
  training_loss.png                 — total + Dirichlet + PDE loss (log scale)
  gradient_quiver_10d.png           — ∇V quiver on x1,y1 plane
  brt_isosurface_10d.png            — 3D sub-volume isosurface (x1, y1, θ1)
  metrics_10d.txt                   — final training loss, model info

Usage (deepreach env):
  python baselines/narrow_passage_10d_analysis.py
"""

import sys
import os
import math
import pickle
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/narrow_passage_10d_run_lr5/training/checkpoints/model_epoch_160000.pth"
LOSS_CSV     = "training_benchmarks/narrow_passage_10d_run_lr5_full_benchmarks.csv"
OUT_DIR      = "baselines/plots/narrow_passage_10d"

# ── NarrowPassage physical constants (must match dynamics) ─────────────────
L            = 2.0
CURBS        = [-2.8, 2.8]
STRANDED_POS = (0.0, -1.8)
GOAL_1       = (6.0, -1.4)
GOAL_2       = (-6.0, 1.4)
AVOID_WEIGHT = 0.5

# State bounds: [x1,y1,θ1,v1,φ1, x2,y2,θ2,v2,φ2]
BOUNDS = [
    (-8.0,  8.0),               # x1
    (-3.8,  3.8),               # y1
    (-math.pi, math.pi),        # θ1
    (-1.0,  7.0),               # v1
    (-0.3*math.pi, 0.3*math.pi),# φ1
    (-8.0,  8.0),               # x2
    (-3.8,  3.8),               # y2
    (-math.pi, math.pi),        # θ2
    (-1.0,  7.0),               # v2
    (-0.3*math.pi, 0.3*math.pi),# φ2
]

T_MAX   = 1.0
N_PLOT  = 80   # resolution for 2D slices (80×80 = 6400 pts — fast)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path, device="cpu"):
    import torch
    from utils.modules import SingleBVPNet
    from dynamics.dynamics import NarrowPassage

    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')

    with open(opt_path, 'rb') as f:
        orig_opt = pickle.load(f)

    dyn = NarrowPassage(
        avoid_fn_weight=orig_opt.avoid_fn_weight,
        avoid_only=orig_opt.avoid_only,
    )
    dyn.deepreach_model = orig_opt.deepreach_model

    model = SingleBVPNet(
        in_features=dyn.input_dim, out_features=1,
        type=orig_opt.model, mode=orig_opt.model_mode,
        hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl,
    )

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, dyn, orig_opt


def eval_slice(model, dyn, fixed_state, n_pts=N_PLOT, device="cpu"):
    """
    Evaluate V on a 2D grid varying (x1, y1) with all other dims fixed.
    fixed_state: dict mapping dim index → value for dims != 0,1
    Returns (n_pts, n_pts) array.
    """
    import torch

    x1_lin = np.linspace(*BOUNDS[0], n_pts)
    y1_lin = np.linspace(*BOUNDS[1], n_pts)
    X1, Y1 = np.meshgrid(x1_lin, y1_lin, indexing='ij')

    coords = np.zeros((n_pts * n_pts, 10), dtype=np.float32)
    coords[:, 0] = X1.ravel()
    coords[:, 1] = Y1.ravel()
    for dim, val in fixed_state.items():
        coords[:, dim] = val

    t_col = np.full((len(coords), 1), T_MAX, dtype=np.float32)
    coords_t = np.concatenate([t_col, coords], axis=1)

    coords_tensor = torch.from_numpy(coords_t).to(device)
    inputs = dyn.coord_to_input(coords_tensor)

    values_list = []
    with torch.no_grad():
        for i in range(0, len(inputs), 10_000):
            batch = inputs[i:i+10_000]
            out   = model({'coords': batch})
            vals  = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(-1))
            values_list.append(vals.cpu().numpy())

    return np.concatenate(values_list).reshape(n_pts, n_pts)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_corridor(ax):
    """Draw curbs and stranded car. Goal omitted: avoid_only=True so goal
    has no effect on the BRT computation."""
    x_lims = BOUNDS[0]
    # curb shading
    ax.fill_between(x_lims, CURBS[0] - 0.6, CURBS[0], color='#888888', alpha=0.4, zorder=0)
    ax.fill_between(x_lims, CURBS[1],       CURBS[1] + 0.6, color='#888888', alpha=0.4, zorder=0)
    # curb lines
    ax.axhline(CURBS[0], color='gray', linewidth=1.5, linestyle='--')
    ax.axhline(CURBS[1], color='gray', linewidth=1.5, linestyle='--')
    # stranded car
    sc = Circle(STRANDED_POS, L / 2, color='#cc4400', alpha=0.7, zorder=5, label='Stranded car')
    ax.add_patch(sc)


def draw_car2(ax, x2, y2, theta2=0.0):
    """Mark vehicle 2 position."""
    ax.plot(x2, y2, 'b^', markersize=10, zorder=6, label=f'Car 2 ({x2:.1f},{y2:.1f})')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — single value function slice
# ══════════════════════════════════════════════════════════════════════════════

def plot_value_slice_single(model, dyn, device):
    """
    V(x1, y1) with car2 approaching from the right at (6, 1.4),
    both cars at default heading/speed/steering.
    """
    print("  Single value function slice …")
    fixed = {
        2: 0.0,            # θ1 = 0 (heading right)
        3: 3.0,            # v1 = 3 m/s
        4: 0.0,            # φ1 = 0 (straight)
        5: 6.0,            # x2 = 6
        6: 1.4,            # y2 = 1.4  (near upper lane)
        7: math.pi,        # θ2 = π (heading left — toward car 1)
        8: 3.0,            # v2 = 3
        9: 0.0,            # φ2 = 0
    }
    V = eval_slice(model, dyn, fixed, device=device)

    x1_lin = np.linspace(*BOUNDS[0], N_PLOT)
    y1_lin = np.linspace(*BOUNDS[1], N_PLOT)
    extent = [BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, origin='lower', extent=extent, cmap='coolwarm',
                   aspect='auto')
    ax.contour(x1_lin, y1_lin, V.T, levels=[0],
               colors='black', linewidths=2.0)
    plt.colorbar(im, ax=ax, label='V(x)  (negative = unsafe)')

    draw_corridor(ax)
    draw_car2(ax, fixed[5], fixed[6], fixed[7])

    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(BOUNDS[1][0] - 0.3, BOUNDS[1][1] + 0.3)
    ax.set_xlabel('$x_1$ (m)')
    ax.set_ylabel('$y_1$ (m)')
    ax.set_title('10D NarrowPassage — Learned Value Function $V(x_1, y_1)$\n'
                 'Car 1: θ₁=0°, v₁=3 m/s  |  Car 2: at (6, 1.4), heading left\n'
                 'Black contour = BRT boundary (V=0)',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "value_slice_car1_pos.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")
    return V


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — multi-panel: vary car 2 position
# ══════════════════════════════════════════════════════════════════════════════

def plot_value_slice_multi(model, dyn, device):
    """4-panel: V(x1,y1) for four different car-2 positions."""
    print("  Multi-panel value slices …")

    configs = [
        {"label": "(a) Car 2 fixed at (6,1.4)\nheading left — approaching",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5: 6.0, 6: 1.4, 7:math.pi, 8:3.0, 9:0.0}},
        {"label": "(b) Car 2 fixed at (0,1.4)\nheading left — closer",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5: 0.0, 6: 1.4, 7:math.pi, 8:3.0, 9:0.0}},
        {"label": "(c) Car 2 fixed at (3,0)\nhead-on conflict",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5: 3.0, 6: 0.0, 7:math.pi, 8:3.0, 9:0.0}},
        {"label": "(d) Car 2 fixed at (-5,1.4)\nsame direction — no conflict",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5:-5.0, 6: 1.4, 7: 0.0,     8:3.0, 9:0.0}},
    ]

    x1_lin = np.linspace(*BOUNDS[0], N_PLOT)
    y1_lin = np.linspace(*BOUNDS[1], N_PLOT)
    extent = [BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('10D NarrowPassage — $V(x_1, y_1)$ for Different Car-2 Positions\n'
                 'Car 1: θ₁=0°, v₁=3 m/s | Black contour = BRT boundary (V=0)',
                 fontsize=11)

    all_V = []
    for cfg in configs:
        V = eval_slice(model, dyn, cfg['fixed'], device=device)
        all_V.append(V)
    vmin = min(v.min() for v in all_V)
    vmax = max(v.max() for v in all_V)

    for ax, cfg, V in zip(axes, configs, all_V):
        im = ax.imshow(V.T, origin='lower', extent=extent, cmap='coolwarm',
                       vmin=vmin, vmax=vmax, aspect='auto')
        ax.contour(x1_lin, y1_lin, V.T, levels=[0],
                   colors='black', linewidths=1.5)

        draw_corridor(ax)
        draw_car2(ax, cfg['fixed'][5], cfg['fixed'][6], cfg['fixed'][7])

        ax.set_xlim(*BOUNDS[0])
        ax.set_ylim(BOUNDS[1][0] - 0.3, BOUNDS[1][1] + 0.3)
        ax.set_xlabel('$x_1$ (m)')
        ax.set_ylabel('$y_1$ (m)')
        ax.set_title(cfg['label'], fontsize=9)

    plt.colorbar(im, ax=axes[-1], shrink=0.85, label='V(x)')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "value_slice_multi.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — training loss
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_loss():
    """Log-scale training loss from CSV."""
    print("  Training loss curve …")
    if not os.path.exists(LOSS_CSV):
        print(f"  CSV not found: {LOSS_CSV} — skipping")
        return

    steps, dirichlet, pde, total = [], [], [], []
    with open(LOSS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row['step'])
            if s % 200 != 0:   # downsample for speed
                continue
            steps.append(s)
            d = row['dirichlet']
            p = row['diff_constraint_hom']
            t = row['total_train_loss']
            dirichlet.append(float(d) if d else np.nan)
            pde.append(float(p) if p else np.nan)
            total.append(float(t) if t else np.nan)

    steps    = np.array(steps)
    total    = np.array(total)
    dirichlet = np.array(dirichlet)
    pde      = np.array(pde)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(steps, total,    color='#1f77b4', linewidth=1.5, label='Total loss')
    if not np.all(np.isnan(dirichlet)):
        ax.semilogy(steps, dirichlet, color='#ff7f0e', linewidth=1.0, linestyle='--',
                    label='Dirichlet (boundary) loss', alpha=0.8)
    if not np.all(np.isnan(pde)):
        ax.semilogy(steps, pde,       color='#2ca02c', linewidth=1.0, linestyle=':',
                    label='PDE constraint loss', alpha=0.8)

    # mark checkpoint
    ax.axvline(160_000, color='purple', linestyle=':', linewidth=1.2,
               label='Checkpoint used (epoch 160k)')

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('10D NarrowPassage (lr5) — Training Loss\n'
                 'Loss ~1,329 at checkpoint. PDE constraint loss dominates after epoch ~100k.',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "training_loss.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — gradient quiver on (x1, y1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradient_quiver(model, dyn, device):
    print("  Gradient quiver …")
    fixed = {2:0.0, 3:3.0, 4:0.0, 5:6.0, 6:1.4, 7:math.pi, 8:3.0, 9:0.0}
    V = eval_slice(model, dyn, fixed, n_pts=N_PLOT, device=device)

    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / (N_PLOT - 1)
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / (N_PLOT - 1)
    gx, gy = np.gradient(V, dx, dy)

    x1_lin = np.linspace(*BOUNDS[0], N_PLOT)
    y1_lin = np.linspace(*BOUNDS[1], N_PLOT)
    XX, YY = np.meshgrid(x1_lin, y1_lin, indexing='ij')
    stride = 6
    extent = [BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, origin='lower', extent=extent,
                   cmap='coolwarm', alpha=0.85, aspect='auto')
    plt.colorbar(im, ax=ax, label='V(x)')

    mag     = np.sqrt(gx**2 + gy**2)
    mag_max = mag.max() or 1.0
    ax.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
              gx[::stride, ::stride], gy[::stride, ::stride],
              mag[::stride, ::stride] / mag_max,
              cmap='plasma', scale=15, width=0.004, alpha=0.85, clim=(0, 1))
    ax.contour(x1_lin, y1_lin, V.T, levels=[0],
               colors='black', linewidths=2.0)

    draw_corridor(ax)
    draw_car2(ax, fixed[5], fixed[6], fixed[7])

    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(BOUNDS[1][0] - 0.3, BOUNDS[1][1] + 0.3)
    ax.set_xlabel('$x_1$ (m)')
    ax.set_ylabel('$y_1$ (m)')
    ax.set_title(r'10D NarrowPassage — $\nabla V$ Quiver on $(x_1, y_1)$ plane'
                 '\nArrows point in direction of increasing V (away from danger)',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gradient_quiver_10d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — 3D isosurface: vary (x1, y1, θ1), fix rest
# ══════════════════════════════════════════════════════════════════════════════

def plot_brt_isosurface(model, dyn, device, n_pts=25):
    print("  3D isosurface (x1, y1, θ1) …")
    try:
        from skimage import measure
    except ImportError:
        print("  skimage not available — skipping isosurface")
        return

    import torch

    x1_lin  = np.linspace(*BOUNDS[0], n_pts)
    y1_lin  = np.linspace(*BOUNDS[1], n_pts)
    th1_lin = np.linspace(*BOUNDS[2], n_pts)

    # Fix: v1=3, φ1=0, x2=6, y2=1.4, θ2=π, v2=3, φ2=0
    X1, Y1, TH1 = np.meshgrid(x1_lin, y1_lin, th1_lin, indexing='ij')
    n_total = n_pts ** 3
    coords = np.zeros((n_total, 10), dtype=np.float32)
    coords[:, 0] = X1.ravel()
    coords[:, 1] = Y1.ravel()
    coords[:, 2] = TH1.ravel()
    coords[:, 3] = 3.0      # v1
    coords[:, 4] = 0.0      # φ1
    coords[:, 5] = 6.0      # x2
    coords[:, 6] = 1.4      # y2
    coords[:, 7] = math.pi  # θ2
    coords[:, 8] = 3.0      # v2
    coords[:, 9] = 0.0      # φ2

    t_col = np.full((n_total, 1), T_MAX, dtype=np.float32)
    coords_t = np.concatenate([t_col, coords], axis=1)

    coords_tensor = torch.from_numpy(coords_t).to(device)
    inputs = dyn.coord_to_input(coords_tensor)

    vals_list = []
    with torch.no_grad():
        for i in range(0, len(inputs), 10_000):
            batch = inputs[i:i+10_000]
            out   = model({'coords': batch})
            v     = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(-1))
            vals_list.append(v.cpu().numpy())
    V3d = np.concatenate(vals_list).reshape(n_pts, n_pts, n_pts)

    if V3d.min() >= 0 or V3d.max() <= 0:
        print("  No zero crossing in 3D sub-volume — skipping isosurface")
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(V3d, level=0)
    except Exception as e:
        print(f"  marching_cubes failed: {e}")
        return

    vx1  = np.interp(verts[:, 0], np.arange(n_pts), x1_lin)
    vy1  = np.interp(verts[:, 1], np.arange(n_pts), y1_lin)
    vth1 = np.interp(verts[:, 2], np.arange(n_pts), th1_lin)

    views = [
        (25, 45, "Isometric"),
        (90,  0, "Top-down  (x₁–y₁)"),
        ( 0,  0, "Side  (x₁–θ₁)"),
    ]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("10D NarrowPassage BRT Isosurface  (V=0 level set, 3D sub-volume)\n"
                 "Varying: x₁, y₁, θ₁  |  Fixed: v₁=3, φ₁=0, car2 at (6,1.4,←)\n"
                 "Blue surface = BRT boundary (states from which collision is unavoidable)",
                 fontsize=11)

    for i, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_trisurf(vx1, vy1, vth1, triangles=faces,
                        alpha=0.50, color='steelblue',
                        edgecolor='none', linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x₁ (m)',    fontsize=8, labelpad=2)
        ax.set_ylabel('y₁ (m)',    fontsize=8, labelpad=2)
        ax.set_zlabel('θ₁ (rad)', fontsize=8, labelpad=2)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "brt_isosurface_10d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics file
# ══════════════════════════════════════════════════════════════════════════════

def write_metrics(orig_opt, V_main):
    brt_pts = int(np.sum(V_main <= 0))
    total_pts = V_main.size
    brt_frac = brt_pts / total_pts * 100

    # read final loss from CSV
    final_loss = None
    if os.path.exists(LOSS_CSV):
        with open(LOSS_CSV) as f:
            for row in f:
                pass  # read to last line
        try:
            final_loss = float(row.strip().split(',')[-1])
        except Exception:
            pass

    path = os.path.join(OUT_DIR, "metrics_10d.txt")
    with open(path, 'w') as f:
        f.write("=== 10D NarrowPassage (lr5) — Analysis ===\n\n")
        f.write(f"Model checkpoint  : {MODEL_PATH}\n")
        f.write(f"Epochs trained    : {orig_opt.num_epochs}\n")
        f.write(f"Checkpoint epoch  : 160,000\n")
        f.write(f"Learning rate     : {orig_opt.lr}\n")
        f.write(f"Architecture      : {orig_opt.num_hl} hidden layers × {orig_opt.num_nl} units (SIREN)\n")
        f.write(f"avoid_fn_weight   : {orig_opt.avoid_fn_weight}\n")
        f.write(f"avoid_only        : {orig_opt.avoid_only}\n\n")
        if final_loss is not None:
            f.write(f"Final training loss (epoch 160k) : {final_loss:.2f}\n")
            f.write(f"Checkpoint loss   (epoch 110k)  : ~1360 (approx)\n\n")
        f.write("--- Value function slice stats (x1,y1 plane) ---\n")
        f.write(f"  Slice V range      : [{V_main.min():.4f}, {V_main.max():.4f}]\n")
        f.write(f"  BRT pts (V<=0)     : {brt_pts} / {total_pts} ({brt_frac:.1f}%)\n")
        f.write("\n(No ground truth available — 10D grid is computationally infeasible)\n")
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading 10D NarrowPassage model …")
    model, dyn, orig_opt = load_model(MODEL_PATH)
    print(f"  Architecture: {orig_opt.num_hl} × {orig_opt.num_nl}, lr={orig_opt.lr}")

    # Determine device
    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"  Device: {device}")
    model.to(device)

    print("\n--- Figures ---")
    V_main = plot_value_slice_single(model, dyn, device)
    plot_value_slice_multi(model, dyn, device)
    plot_training_loss()
    plot_gradient_quiver(model, dyn, device)
    plot_brt_isosurface(model, dyn, device)

    print("\n--- Metrics ---")
    write_metrics(orig_opt, V_main)

    print("\n10D NarrowPassage analysis complete.")


if __name__ == "__main__":
    main()
