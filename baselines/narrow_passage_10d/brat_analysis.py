"""
10D NarrowPassage BRAT Analysis
Model: runs/narrow_passage_10d_run_BRAT/training/checkpoints/model_epoch_160000.pth
avoid_only=False → full BRAT formulation (matches the original DeepReach paper)
No grid-based ground truth (10D grid is computationally infeasible).

State: [x1, y1, θ1, v1, φ1, x2, y2, θ2, v2, φ2]
  x, y  : position in corridor  (x ∈ [-8,8], y ∈ [-3.8,3.8])
  θ     : heading                (θ ∈ [-π, π])
  v     : speed                  (v ∈ [-1, 7])
  φ     : steering angle         (φ ∈ [-0.3π, 0.3π])

Outputs (baselines/narrow_passage_10d/plots/):
  value_slice_car1_pos.png   — V(x1,y1) heatmap, car 2 at (6, 1.4)
  value_slice_multi.png      — 4-panel: different car-2 positions
  gradient_quiver_10d.png    — ∇V quiver on x1,y1 plane
  brt_isosurface_10d.png     — 3D sub-volume isosurface (x1, y1, θ1)
  metrics_10d_brat.txt       — model info, value stats

Usage (deepreach env):
  cd ~/deepreach_CMPT419
  python baselines/narrow_passage_10d_brat_analysis.py
"""

import sys
import os
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = "runs/narrow_passage_10d_run_BRAT/training/checkpoints/model_epoch_160000.pth"
OUT_DIR    = "baselines/narrow_passage_10d/plots"
# Note: no separate CSV exists for the BRAT run — training loss plot skipped

# ── NarrowPassage physical constants ──────────────────────────────────────
L            = 2.0
CURBS        = [-2.8, 2.8]
STRANDED_POS = (0.0, -1.8)
GOAL_1       = (6.0, -1.4)
GOAL_2       = (-6.0, 1.4)

BOUNDS = [
    (-8.0,  8.0),
    (-3.8,  3.8),
    (-math.pi, math.pi),
    (-1.0,  7.0),
    (-0.3*math.pi, 0.3*math.pi),
    (-8.0,  8.0),
    (-3.8,  3.8),
    (-math.pi, math.pi),
    (-1.0,  7.0),
    (-0.3*math.pi, 0.3*math.pi),
]

T_MAX  = 1.0
N_PLOT = 80


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

def draw_corridor(ax, show_goals=True):
    """Draw curbs, stranded car, and goal markers.
    For BRAT (avoid_only=False) goals are relevant — show them."""
    x_lims = BOUNDS[0]
    ax.fill_between(x_lims, CURBS[0] - 0.6, CURBS[0], color='#888888', alpha=0.4, zorder=0)
    ax.fill_between(x_lims, CURBS[1],       CURBS[1] + 0.6, color='#888888', alpha=0.4, zorder=0)
    ax.axhline(CURBS[0], color='gray', linewidth=1.5, linestyle='--')
    ax.axhline(CURBS[1], color='gray', linewidth=1.5, linestyle='--')
    sc = Circle(STRANDED_POS, L / 2, color='#cc4400', alpha=0.7, zorder=5, label='Stranded car')
    ax.add_patch(sc)
    if show_goals:
        ax.plot(*GOAL_1, 'g*', markersize=12, zorder=6, label=f'Goal 1 {GOAL_1}')
        ax.plot(*GOAL_2, 'm*', markersize=12, zorder=6, label=f'Goal 2 {GOAL_2}')


def draw_car2(ax, x2, y2):
    ax.plot(x2, y2, 'b^', markersize=10, zorder=6, label=f'Car 2 ({x2:.1f},{y2:.1f})')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — single value slice
# ══════════════════════════════════════════════════════════════════════════════

def plot_value_slice_single(model, dyn, device):
    print("  Single value slice …")
    fixed = {2: 0.0, 3: 3.0, 4: 0.0,
             5: 6.0, 6: 1.4, 7: math.pi, 8: 3.0, 9: 0.0}
    V = eval_slice(model, dyn, fixed, device=device)

    x1_lin = np.linspace(*BOUNDS[0], N_PLOT)
    y1_lin = np.linspace(*BOUNDS[1], N_PLOT)
    extent = [BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(V.T, origin='lower', extent=extent, cmap='coolwarm', aspect='auto')
    ax.contour(x1_lin, y1_lin, V.T, levels=[0], colors='black', linewidths=2.0)
    plt.colorbar(im, ax=ax, label='V(x)  (negative = unsafe)')

    draw_corridor(ax, show_goals=True)
    draw_car2(ax, fixed[5], fixed[6])

    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(BOUNDS[1][0] - 0.3, BOUNDS[1][1] + 0.3)
    ax.set_xlabel('$x_1$ (m)')
    ax.set_ylabel('$y_1$ (m)')
    ax.set_title('10D NarrowPassage BRAT — Learned Value Function $V(x_1, y_1)$\n'
                 'Car 1: θ₁=0°, v₁=3 m/s  |  Car 2: at (6, 1.4), heading left\n'
                 'Black contour = BRAT boundary (V=0): states from which goal is unreachable\n'
                 'Stars = goal positions (relevant for BRAT)',
                 fontsize=9)
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "value_slice_car1_pos.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")
    return V


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — multi-panel: four car-2 positions
# ══════════════════════════════════════════════════════════════════════════════

def plot_value_slice_multi(model, dyn, device):
    print("  Multi-panel value slices …")
    # Panel (a): Car 2 near its demo start position (upper lane, far right)
    # Panel (b): Car 2 mid-corridor in upper lane — conflict zone as Car 1 swerves up
    configs = [
        {"label": "(a) Car 2 at (6,1.4)\nheading left — demo start position",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5: 6.0, 6: 1.4, 7:math.pi, 8:3.0, 9:0.0}},
        {"label": "(b) Car 2 at (3,1.4)\nupper lane — corridor conflict zone",
         "fixed": {2:0.0, 3:3.0, 4:0.0, 5: 3.0, 6: 1.4, 7:math.pi, 8:3.0, 9:0.0}},
    ]

    x1_lin = np.linspace(*BOUNDS[0], N_PLOT)
    y1_lin = np.linspace(*BOUNDS[1], N_PLOT)
    extent = [BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('10D NarrowPassage BRAT — $V(x_1, y_1)$ for Different Car-2 Positions\n'
                 'Car 1: θ₁=0°, v₁=3 m/s | Black contour = BRAT boundary (V=0) | '
                 'Stars = goal positions',
                 fontsize=11)

    all_V = [eval_slice(model, dyn, cfg['fixed'], device=device) for cfg in configs]
    vmin = min(v.min() for v in all_V)
    vmax = max(v.max() for v in all_V)

    for ax, cfg, V in zip(axes, configs, all_V):
        im = ax.imshow(V.T, origin='lower', extent=extent,
                       cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
        ax.contour(x1_lin, y1_lin, V.T, levels=[0], colors='black', linewidths=1.5)
        draw_corridor(ax, show_goals=True)
        draw_car2(ax, cfg['fixed'][5], cfg['fixed'][6])
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
# Figure 2b — 1D corridor sweep: V(x1) along lower lane, two car-2 configs
# ══════════════════════════════════════════════════════════════════════════════

def plot_1d_corridor_sweep(model, dyn, device, n_pts=200):
    """
    Sweep x1 from -8 to +8 along y1 = -1.4 (goal-1 lane) with two car-2
    positions: far-approaching vs. head-on threat. Shade the stranded obstacle
    footprint and mark the goal. Shows BRAT value rises toward the goal and
    drops when car 2 is in direct conflict — demonstrates spatial goal structure.
    Outputs: narrow_passage_10d_brat/value_slice_1d_x1.png
    """
    import torch
    print("  1D corridor sweep (x1 sweep) …")

    x1_lin = np.linspace(*BOUNDS[0], n_pts).astype(np.float32)

    configs = [
        {"label": "Car 2 far  $(6.0,\\ 1.4)$, heading left",
         "x2": 6.0, "y2": 1.4, "color": "steelblue"},
        {"label": "Car 2 head-on $(1.0,\\ 0.0)$, heading left",
         "x2": 1.0, "y2": 0.0, "color": "darkorange"},
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    all_V = []

    for cfg in configs:
        coords = np.zeros((n_pts, 10), dtype=np.float32)
        coords[:, 0] = x1_lin      # x1 varies
        coords[:, 1] = -1.4        # y1 = lower lane (goal-1 y)
        coords[:, 2] = 0.0         # θ1 = 0° (heading right)
        coords[:, 3] = 3.0         # v1
        coords[:, 4] = 0.0         # φ1
        coords[:, 5] = cfg["x2"]   # x2
        coords[:, 6] = cfg["y2"]   # y2
        coords[:, 7] = math.pi     # θ2 (heading left)
        coords[:, 8] = 3.0         # v2
        coords[:, 9] = 0.0         # φ2

        t_col = np.full((n_pts, 1), T_MAX, dtype=np.float32)
        coords_t = np.concatenate([t_col, coords], axis=1)
        coords_tensor = torch.from_numpy(coords_t).to(device)
        inputs = dyn.coord_to_input(coords_tensor)

        vals_list = []
        with torch.no_grad():
            for i in range(0, len(inputs), 10_000):
                batch = inputs[i:i+10_000]
                out   = model({'coords': batch})
                vals  = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(-1))
                vals_list.append(vals.cpu().numpy())
        V = np.concatenate(vals_list)
        all_V.append(V)
        ax.plot(x1_lin, V, color=cfg["color"], linewidth=2.0, label=cfg["label"])

    y_lo = min(v.min() for v in all_V) - 0.3
    y_hi = max(v.max() for v in all_V) + 0.3

    # Stranded obstacle footprint projected onto y1=-1.4 lane
    # Obstacle at (0, -1.8), radius = L/2 = 1.0 m
    # Collision when sqrt(x1^2 + (y1 - (-1.8))^2) <= 1.0
    # With y1=-1.4: sqrt(x1^2 + 0.16) <= 1.0  →  |x1| <= sqrt(0.84) ≈ 0.917
    dy_obs     = (-1.4) - STRANDED_POS[1]   # = 0.4
    obs_radius = L / 2                       # = 1.0 m
    x_obs_half = math.sqrt(max(obs_radius**2 - dy_obs**2, 0.0))
    in_obstacle = np.abs(x1_lin - STRANDED_POS[0]) <= x_obs_half
    if np.any(in_obstacle):
        ax.fill_between(x1_lin, y_lo, y_hi, where=in_obstacle,
                        color='#FF6600', alpha=0.30, label='Stranded obstacle footprint')

    # Goal and V=0 reference
    ax.axvline(GOAL_1[0], color='green', linewidth=1.5, linestyle='--',
               alpha=0.8, label=f'Goal $x_1 = {GOAL_1[0]}$ m')
    ax.axhline(0, color='black', linewidth=1.0, linestyle=':', alpha=0.5, label='$V = 0$')

    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('$x_1$ (m) — car 1 position along corridor', fontsize=11)
    ax.set_ylabel('$V(x_1)$', fontsize=11)
    ax.set_title('10D NarrowPassage BRAT — 1D Value Sweep Along Goal Lane ($y_1 = -1.4$ m)\n'
                 '$\\theta_1=0°$, $v_1=3$ m/s | BRAT: $V > 0$ means goal is reachable\n'
                 'Value rises toward goal; drops when Car 2 is in direct conflict (orange curve)',
                 fontsize=9)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "value_slice_1d_x1.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — gradient quiver
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
    ax.contour(x1_lin, y1_lin, V.T, levels=[0], colors='black', linewidths=2.0)
    draw_corridor(ax, show_goals=True)
    draw_car2(ax, fixed[5], fixed[6])
    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(BOUNDS[1][0] - 0.3, BOUNDS[1][1] + 0.3)
    ax.set_xlabel('$x_1$ (m)')
    ax.set_ylabel('$y_1$ (m)')
    ax.set_title(r'10D NarrowPassage BRAT — $\nabla V$ on $(x_1, y_1)$'
                 '\nArrows point toward increasing V (away from danger / toward goal)',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gradient_quiver_10d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — 3D isosurface: (x1, y1, θ1)
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

    X1, Y1, TH1 = np.meshgrid(x1_lin, y1_lin, th1_lin, indexing='ij')
    n_total = n_pts ** 3
    coords = np.zeros((n_total, 10), dtype=np.float32)
    coords[:, 0] = X1.ravel()
    coords[:, 1] = Y1.ravel()
    coords[:, 2] = TH1.ravel()
    coords[:, 3] = 3.0
    coords[:, 4] = 0.0
    coords[:, 5] = 6.0
    coords[:, 6] = 1.4
    coords[:, 7] = math.pi
    coords[:, 8] = 3.0
    coords[:, 9] = 0.0

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

    views = [(25, 45, "Isometric"), (90, 0, "Top-down (x₁–y₁)"), (0, 0, "Side (x₁–θ₁)")]
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("10D NarrowPassage BRAT Isosurface (V=0, 3D sub-volume)\n"
                 "Varying: x₁, y₁, θ₁  |  Fixed: v₁=3, φ₁=0, car2 at (6,1.4,←)\n"
                 "Surface = boundary of states where goal is unreachable (BRAT)",
                 fontsize=11)
    for i, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_trisurf(vx1, vy1, vth1, triangles=faces,
                        alpha=0.50, color='steelblue', edgecolor='none', linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x₁ (m)', fontsize=8, labelpad=2)
        ax.set_ylabel('y₁ (m)', fontsize=8, labelpad=2)
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
    brt_pts   = int(np.sum(V_main <= 0))
    total_pts = V_main.size
    brt_frac  = brt_pts / total_pts * 100

    path = os.path.join(OUT_DIR, "metrics_10d_brat.txt")
    with open(path, 'w') as f:
        f.write("=== 10D NarrowPassage BRAT — Analysis ===\n\n")
        f.write(f"Model checkpoint  : {MODEL_PATH}\n")
        f.write(f"Checkpoint epoch  : 160,000\n")
        f.write(f"Learning rate     : {orig_opt.lr}\n")
        f.write(f"Architecture      : {orig_opt.num_hl} × {orig_opt.num_nl} (SIREN)\n")
        f.write(f"avoid_fn_weight   : {orig_opt.avoid_fn_weight}\n")
        f.write(f"avoid_only        : {orig_opt.avoid_only}  (False = BRAT, True = BRT)\n\n")
        f.write("NOTE: This is the BRAT formulation (avoid_only=False), matching the\n")
        f.write("      original DeepReach paper. Previous run used BRT (avoid_only=True).\n\n")
        f.write("--- Value function slice stats (x1,y1 plane) ---\n")
        f.write(f"  V range      : [{V_main.min():.4f}, {V_main.max():.4f}]\n")
        f.write(f"  BRAT pts (V<=0) : {brt_pts} / {total_pts} ({brt_frac:.1f}%)\n")
        f.write("\n(No ground truth — 10D grid computationally infeasible)\n")
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading 10D NarrowPassage BRAT model …")
    model, dyn, orig_opt = load_model(MODEL_PATH, device=device)
    print(f"  Architecture: {orig_opt.num_hl} × {orig_opt.num_nl}, "
          f"lr={orig_opt.lr}, avoid_only={orig_opt.avoid_only}")
    model.to(device)

    V_main = plot_value_slice_single(model, dyn, device)
    plot_value_slice_multi(model, dyn, device)
    plot_1d_corridor_sweep(model, dyn, device)
    plot_gradient_quiver(model, dyn, device)
    plot_brt_isosurface(model, dyn, device)
    write_metrics(orig_opt, V_main)

    print("\n10D NarrowPassage BRAT analysis complete.")


if __name__ == "__main__":
    main()
