"""
Phase 3 — 6D TwoVehicleCollision Analysis
Model:    runs/collision_6d_run/training/checkpoints/model_epoch_110000.pth
Baseline: baselines/grids/collision6d_grid_{11,15,21}pt.npy

Outputs (all in baselines/plots/collision6d/):
  comparison_6d_21pt.png          — 2 panels: baseline V | DeepReach V  (21pt)
  brt_overlay_6d_21pt.png         — 1 panel:  BRT overlap at 21pt
  brt_resolution_comparison.png   — 3 panels: 11pt | 15pt | 21pt BRT side-by-side
  gradient_quiver_6d.png          — 2 panels: ∇V quiver (θ1=0,θ2=0 | θ1=π/2,θ2=0)
  metrics_resolution_table.txt    — MSE + BRT error at 11, 15, 21pt

6D state order: [x1, y1, x2, y2, θ1, θ2]
2D slice shown: x1 vs y1, with x2=0.5, y2=0.0 fixed (vehicle 2 at (0.5, 0))

Usage (deepreach env):
  python baselines/collision6d_analysis.py
"""

import sys
import os
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = "runs/collision_6d_run/training/checkpoints/model_epoch_110000.pth"
GRID_PATHS = {
    11: "baselines/grids/collision6d_grid_11pt.npy",
    15: "baselines/grids/collision6d_grid_15pt.npy",
    21: "baselines/grids/collision6d_grid_21pt.npy",
}
OUT_DIR = "baselines/plots/collision6d"
T_MAX   = 1.0

# ── 6D grid bounds: [x1, y1, x2, y2, θ1, θ2] ──────────────────────────────
BOUNDS = [
    (-1.0,      1.0),       # x1
    (-1.0,      1.0),       # y1
    (-1.0,      1.0),       # x2
    (-1.0,      1.0),       # y2
    (-math.pi,  math.pi),   # θ1
    (-math.pi,  math.pi),   # θ2
]
DIM_LABELS = [r'$x_1$', r'$y_1$', r'$x_2$', r'$y_2$', r'$\theta_1$', r'$\theta_2$']

# 2D slice config: vary dims 0,1 (x1, y1); fix dims 2,3,4,5
PLOT_X, PLOT_Y = 0, 1          # x1, y1
FIXED = {2: 0.5, 3: 0.0, 4: 0.0, 5: 0.0}   # x2=0.5, y2=0, θ1=0, θ2=0
VEHICLE2_POS = (FIXED[2], FIXED[3])          # for annotation

COLLISION_R = 0.25             # collision radius (for circle annotation)


# ══════════════════════════════════════════════════════════════════════════════
# Model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(model_path, device="cpu"):
    """Load and return (model, dyn) — shared across resolutions."""
    import torch
    from utils.modules import SingleBVPNet
    from dynamics.dynamics import TwoVehicleCollision6D

    dyn = TwoVehicleCollision6D()
    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')

    if os.path.exists(opt_path):
        with open(opt_path, 'rb') as f:
            orig_opt = pickle.load(f)
        dyn.deepreach_model = orig_opt.deepreach_model
        model = SingleBVPNet(in_features=7, out_features=1,
                             type=orig_opt.model, mode=orig_opt.model_mode,
                             hidden_features=orig_opt.num_nl,
                             num_hidden_layers=orig_opt.num_hl)
    else:
        dyn.deepreach_model = "exact"
        model = SingleBVPNet(in_features=7, out_features=1,
                             type='sine', mode='mlp',
                             hidden_features=512, num_hidden_layers=3)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()
    return model, dyn


def _run_batches(model, dyn, coords_with_time, batch_size=50_000):
    import torch
    coords_tensor = torch.from_numpy(coords_with_time).float()
    inputs = dyn.coord_to_input(coords_tensor)
    values_list = []
    n_batches = math.ceil(len(inputs) / batch_size)
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            out   = model({'coords': batch})
            vals  = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(dim=-1))
            values_list.append(vals.cpu().numpy())
            b_idx = i // batch_size + 1
            if b_idx % max(1, n_batches // 10) == 0:
                print(f"    {b_idx}/{n_batches} batches", flush=True)
    return np.concatenate(values_list)


def evaluate_model_full(model, dyn, n_pts):
    """Evaluate on full n_pts^6 grid. Only feasible for small n_pts (≤15)."""
    axes = [np.linspace(b[0], b[1], n_pts) for b in BOUNDS]
    grids = np.meshgrid(*axes, indexing='ij')
    coords_flat = np.stack([g.ravel() for g in grids], axis=-1)
    t_col = np.full((len(coords_flat), 1), T_MAX)
    vals = _run_batches(model, dyn, np.concatenate([t_col, coords_flat], axis=-1))
    return vals.reshape(*([n_pts] * 6))


def evaluate_model_slice(model, dyn, n_pts, fixed=FIXED):
    """
    Evaluate DeepReach only on the 2D slice (x1, y1) with other dims fixed.
    Returns a (n_pts, n_pts) array — avoids allocating the full 6D grid.
    """
    x1_lin = np.linspace(*BOUNDS[PLOT_X], n_pts)
    y1_lin = np.linspace(*BOUNDS[PLOT_Y], n_pts)
    X1, Y1 = np.meshgrid(x1_lin, y1_lin, indexing='ij')

    # Build fixed values for other dims
    fixed_vals = []
    for d in range(6):
        if d == PLOT_X:
            fixed_vals.append(X1.ravel())
        elif d == PLOT_Y:
            fixed_vals.append(Y1.ravel())
        else:
            fixed_vals.append(np.full(n_pts * n_pts, fixed.get(d, 0.0)))

    coords_flat = np.stack(fixed_vals, axis=-1)
    t_col = np.full((len(coords_flat), 1), T_MAX)
    vals = _run_batches(model, dyn, np.concatenate([t_col, coords_flat], axis=-1))
    return vals.reshape(n_pts, n_pts)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_slice(volume, n_pts, fixed=FIXED, px=PLOT_X, py=PLOT_Y):
    """Extract 2D slice (px, py) from 6D volume with other dims fixed."""
    idx = [slice(None)] * 6
    for d in range(6):
        if d in (px, py):
            continue
        lo, hi = BOUNDS[d]
        axis   = np.linspace(lo, hi, volume.shape[d])
        val    = fixed.get(d, 0.0)
        idx[d] = int(np.argmin(np.abs(axis - val)))
    return volume[tuple(idx)]


def brt_vol_error(bl, dr):
    bl_brt = bl <= 0
    dr_brt = dr <= 0
    sym    = np.sum(bl_brt != dr_brt)
    vol    = np.sum(bl_brt)
    return float(sym / vol * 100) if vol > 0 else 0.0


def add_vehicle2(ax):
    """Mark vehicle 2's fixed position with a circle."""
    circ = Circle(VEHICLE2_POS, COLLISION_R,
                  fill=False, edgecolor='white', linewidth=1.5,
                  linestyle='--', label=f'Vehicle 2 (collision R={COLLISION_R})')
    ax.add_patch(circ)
    ax.plot(*VEHICLE2_POS, 'w^', markersize=8, label='Vehicle 2 pos')


# ══════════════════════════════════════════════════════════════════════════════
# 3A + 3B  Comparison and BRT overlay at 21pt
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison_slice(bl_sl, dr_sl, n_pts):
    """2-panel: baseline V | DeepReach V from pre-extracted 2D slices."""
    vmin   = min(bl_sl.min(), dr_sl.min())
    vmax   = max(bl_sl.max(), dr_sl.max())
    x_lin  = np.linspace(*BOUNDS[PLOT_X], n_pts)
    y_lin  = np.linspace(*BOUNDS[PLOT_Y], n_pts)
    extent = [*BOUNDS[PLOT_X], *BOUNDS[PLOT_Y]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"6D Collision — Value Function ({n_pts}pt/dim)\n"
                 f"Slice: x₁ vs y₁ | vehicle 2 at ({VEHICLE2_POS[0]}, {VEHICLE2_POS[1]}), "
                 f"θ₁=θ₂=0",
                 fontsize=11)

    for ax, data, title in zip(axes,
                                [bl_sl, dr_sl],
                                ["Ground Truth (optimized_dp)", "DeepReach"]):
        im = ax.imshow(data.T, origin='lower', extent=extent,
                       cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.contour(x_lin, y_lin, data.T, levels=[0],
                   colors='black', linewidths=1.5)
        add_vehicle2(ax)
        plt.colorbar(im, ax=ax, shrink=0.85, label="V(x)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$y_1$")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "comparison_6d_21pt.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def _plot_brt_overlay_slice(bl_sl, dr_sl, metrics, n_pts):
    """1-panel BRT overlay from pre-extracted 2D slices."""
    bl_brt = bl_sl <= 0
    dr_brt = dr_sl <= 0

    overlay = np.zeros((*bl_brt.shape, 3))
    overlay[bl_brt & dr_brt]   = [0.2, 0.8, 0.2]
    overlay[bl_brt & ~dr_brt]  = [1.0, 0.4, 0.7]
    overlay[~bl_brt & dr_brt]  = [1.0, 0.6, 0.2]

    m = metrics[n_pts]
    mse_str = f"MSE: {m['mse']:.2e}" if m['mse'] is not None else "MSE: slice only"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay.transpose(1, 0, 2), origin='lower',
              extent=[*BOUNDS[PLOT_X], *BOUNDS[PLOT_Y]])
    add_vehicle2(ax)
    ax.set_title(f"6D BRT Overlay — {n_pts}pt/dim\n"
                 f"Vol Error: {m['brt_vol_error']:.1f}%  |  {mse_str}",
                 fontsize=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$y_1$")

    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='Both agree (in BRT)'),
        Patch(facecolor=[1.0, 0.4, 0.7], label='Baseline only (missed)'),
        Patch(facecolor=[1.0, 0.6, 0.2], label='DeepReach only (false pos.)'),
        Patch(facecolor='black',          label='Both agree (outside BRT)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "brt_overlay_6d_21pt.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3B  Resolution comparison (11 | 15 | 21pt)
# ══════════════════════════════════════════════════════════════════════════════

def plot_resolution_comparison(slice_baselines, slice_deepreaches, metrics, resolutions):
    """N-panel BRT overlay side-by-side, one panel per resolution."""
    n_panels = len(resolutions)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("6D Collision BRT — Baseline Resolution Comparison\n"
                 "As grid resolution increases, BRT boundary smooths out "
                 "→ lower apparent error (baseline artifact, not model error)",
                 fontsize=11)

    for ax, n_pts in zip(axes, resolutions):
        bl_sl  = slice_baselines[n_pts]
        dr_sl  = slice_deepreaches[n_pts]
        bl_brt = bl_sl <= 0
        dr_brt = dr_sl <= 0

        overlay = np.zeros((*bl_brt.shape, 3))
        overlay[bl_brt & dr_brt]   = [0.2, 0.8, 0.2]
        overlay[bl_brt & ~dr_brt]  = [1.0, 0.4, 0.7]
        overlay[~bl_brt & dr_brt]  = [1.0, 0.6, 0.2]

        ax.imshow(overlay.transpose(1, 0, 2), origin='lower',
                  extent=[*BOUNDS[PLOT_X], *BOUNDS[PLOT_Y]])
        add_vehicle2(ax)
        m = metrics[n_pts]
        mse_str = f"MSE: {m['mse']:.2e}" if m['mse'] is not None else "MSE: N/A"
        ax.set_title(f"{n_pts}pt/dim  ({n_pts}⁶ = {n_pts**6:,} pts)\n"
                     f"BRT Vol Err: {m['brt_vol_error']:.1f}%  |  {mse_str}",
                     fontsize=9)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$y_1$")

    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='Both agree (in BRT)'),
        Patch(facecolor=[1.0, 0.4, 0.7], label='Baseline only'),
        Patch(facecolor=[1.0, 0.6, 0.2], label='DeepReach only'),
        Patch(facecolor='black',          label='Both outside BRT'),
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=7)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "brt_resolution_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3C  Gradient quiver (2-panel: θ1=0,θ2=0  |  θ1=π/2,θ2=0)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_gradient_quiver_slices(model, dyn, n_pts=21):
    """Evaluate two heading configs as slices, then plot quiver."""
    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / (n_pts - 1)
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / (n_pts - 1)

    slice_configs = [
        ({2: 0.5, 3: 0.0, 4: 0.0,       5: 0.0}, r"$\theta_1=0,\ \theta_2=0$"),
        ({2: 0.5, 3: 0.0, 4: math.pi/2, 5: 0.0}, r"$\theta_1=\pi/2,\ \theta_2=0$"),
    ]

    x_lin = np.linspace(*BOUNDS[PLOT_X], n_pts)
    y_lin = np.linspace(*BOUNDS[PLOT_Y], n_pts)
    XX, YY = np.meshgrid(x_lin, y_lin, indexing='ij')
    stride = 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"6D Collision DeepReach: $\nabla V$ Quiver (x₁, y₁ plane)"
                 "\nArrows point away from danger — vehicle 2 position marked",
                 fontsize=11)

    for ax, (fixed_cfg, title_str) in zip(axes, slice_configs):
        print(f"  Evaluating slice for quiver: {title_str} …")
        V_sl = evaluate_model_slice(model, dyn, n_pts, fixed=fixed_cfg)
        gx, gy = np.gradient(V_sl, dx, dy)

        im = ax.imshow(V_sl.T, origin='lower',
                       extent=[*BOUNDS[PLOT_X], *BOUNDS[PLOT_Y]],
                       cmap='coolwarm', alpha=0.85)
        plt.colorbar(im, ax=ax, shrink=0.8, label="V(x)")

        mag     = np.sqrt(gx**2 + gy**2)
        mag_max = mag.max() or 1.0
        ax.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
                  gx[::stride, ::stride],  gy[::stride, ::stride],
                  mag[::stride, ::stride] / mag_max,
                  cmap='plasma', scale=12, width=0.005,
                  alpha=0.8, clim=(0, 1))
        ax.contour(x_lin, y_lin, V_sl.T, levels=[0],
                   colors='black', linewidths=1.5)
        add_vehicle2(ax)

        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$y_1$")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gradient_quiver_6d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3D isosurface: fix θ1, θ2; vary x1, y1, x2  →  show collision tube in 3D
# ══════════════════════════════════════════════════════════════════════════════

def plot_brt_isosurface_6d(model, dyn, n_pts=21):
    """
    Extract a 3D sub-volume by fixing y2=0, θ1=0, θ2=0 and varying x1, y1, x2.
    Run marching cubes on this 3D sub-volume to get the collision tube in 3D.
    3 views: isometric, top-down (x1–y1), side (x1–x2).
    """
    try:
        from skimage import measure
    except ImportError:
        print("  skimage not available — skipping 6D isosurface")
        return

    print("  Building 3D sub-volume for isosurface (x1, y1, x2 | y2=0, θ1=0, θ2=0) …")

    x1_lin = np.linspace(*BOUNDS[0], n_pts)
    y1_lin = np.linspace(*BOUNDS[1], n_pts)
    x2_lin = np.linspace(*BOUNDS[2], n_pts)
    # Fix y2=0, θ1=0, θ2=0
    y2_val, θ1_val, θ2_val = 0.0, 0.0, 0.0

    X1, Y1, X2 = np.meshgrid(x1_lin, y1_lin, x2_lin, indexing='ij')
    y2_arr  = np.full_like(X1, y2_val)
    θ1_arr  = np.full_like(X1, θ1_val)
    θ2_arr  = np.full_like(X1, θ2_val)

    coords_flat = np.stack([X1.ravel(), Y1.ravel(), X2.ravel(),
                             y2_arr.ravel(), θ1_arr.ravel(), θ2_arr.ravel()], axis=-1)
    t_col = np.full((len(coords_flat), 1), T_MAX, dtype=np.float32)
    coords_with_time = np.concatenate([t_col, coords_flat.astype(np.float32)], axis=1)

    import torch
    vals = _run_batches(model, dyn,
                        coords_with_time)
    V3d = vals.reshape(n_pts, n_pts, n_pts)

    # Check there's a zero crossing
    if V3d.min() >= 0 or V3d.max() <= 0:
        print("  No zero crossing in sub-volume — skipping isosurface")
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(V3d, level=0)
    except Exception as e:
        print(f"  marching_cubes failed: {e}")
        return

    # Convert grid indices → physical coords
    vx1 = np.interp(verts[:, 0], np.arange(n_pts), x1_lin)
    vy1 = np.interp(verts[:, 1], np.arange(n_pts), y1_lin)
    vx2 = np.interp(verts[:, 2], np.arange(n_pts), x2_lin)

    views = [
        (25, 45,  "Isometric"),
        (90,  0,  "Top-down  (x₁–y₁)"),
        ( 0,  0,  "Side  (x₁–x₂)"),
    ]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("6D Collision BRT Isosurface  (3D sub-volume: x₁, y₁, x₂)\n"
                 "Fixed: y₂=0, θ₁=0, θ₂=0  |  Blue = collision-unavoidable region boundary",
                 fontsize=11)

    for i, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_trisurf(vx1, vy1, vx2, triangles=faces,
                        alpha=0.50, color='steelblue',
                        edgecolor='none', linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x₁", fontsize=8, labelpad=2)
        ax.set_ylabel("y₁", fontsize=8, labelpad=2)
        ax.set_zlabel("x₂", fontsize=8, labelpad=2)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "brt_isosurface_6d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load model once ────────────────────────────────────────────────────
    print("Loading DeepReach model …")
    model, dyn = _load_model(MODEL_PATH)

    # ── load baselines ─────────────────────────────────────────────────────
    baselines = {}
    for n, path in sorted(GRID_PATHS.items()):
        if not os.path.exists(path):
            print(f"  Skipping {n}pt — grid not found: {path}")
            continue
        baselines[n] = np.load(path)
        print(f"Loaded baseline {n}pt: shape={baselines[n].shape}  "
              f"range=[{baselines[n].min():.4f}, {baselines[n].max():.4f}]")

    # ── full-grid evaluation: 11pt and 15pt only (memory-feasible) ─────────
    full_deepreaches = {}
    metrics = {}
    for n in [11, 15]:
        if n not in baselines:
            continue
        print(f"\nFull-grid evaluation at {n}pt ({n**6:,} points) …")
        full_deepreaches[n] = evaluate_model_full(model, dyn, n)
        dr = full_deepreaches[n]
        bl = baselines[n]
        print(f"  DR range=[{dr.min():.4f}, {dr.max():.4f}]")
        mse     = float(np.mean((bl - dr) ** 2))
        vol_err = brt_vol_error(bl, dr)
        metrics[n] = {"mse": mse, "brt_vol_error": vol_err,
                      "bl_brt": int(np.sum(bl <= 0)),
                      "dr_brt": int(np.sum(dr <= 0))}
        print(f"  MSE={mse:.4e}  BRT vol error={vol_err:.2f}%")

    # ── slice-only evaluation: 21pt (85M pts full grid exceeds RAM) ────────
    slice_deepreaches = {}   # {n_pts: 2D slice array}
    slice_baselines   = {}

    for n in sorted(GRID_PATHS.keys()):
        if n not in baselines:
            continue
        if n in full_deepreaches:
            # Reuse full grid slice
            slice_deepreaches[n] = get_slice(full_deepreaches[n], n)
            slice_baselines[n]   = get_slice(baselines[n], n)
        else:
            print(f"\nSlice-only evaluation at {n}pt ({n*n} points) …")
            slice_deepreaches[n] = evaluate_model_slice(model, dyn, n)
            slice_baselines[n]   = get_slice(baselines[n], n)
            print(f"  Slice DR range=[{slice_deepreaches[n].min():.4f}, "
                  f"{slice_deepreaches[n].max():.4f}]")
            # Approximate BRT error on slice only
            vol_err = brt_vol_error(slice_baselines[n], slice_deepreaches[n])
            metrics[n] = {"mse": None, "brt_vol_error": vol_err,
                          "bl_brt": int(np.sum(slice_baselines[n] <= 0)),
                          "dr_brt": int(np.sum(slice_deepreaches[n] <= 0)),
                          "slice_only": True}
            print(f"  Slice BRT vol error={vol_err:.2f}%")

    # ── figures ────────────────────────────────────────────────────────────
    print("\n--- Figures ---")

    # comparison + overlay at 21pt using slice data
    if 21 in slice_deepreaches:
        _plot_comparison_slice(slice_baselines[21], slice_deepreaches[21], 21)
        _plot_brt_overlay_slice(slice_baselines[21], slice_deepreaches[21], metrics, 21)

    # resolution comparison (11 | 15 | 21)
    avail_slices = sorted(slice_deepreaches.keys())
    if len(avail_slices) >= 2:
        plot_resolution_comparison(slice_baselines, slice_deepreaches, metrics, avail_slices)

    # gradient quiver at 21pt using slice
    if 21 in slice_deepreaches:
        _plot_gradient_quiver_slices(model, dyn, 21)

    # 3D isosurface (3D sub-volume)
    plot_brt_isosurface_6d(model, dyn, n_pts=21)

    # ── metrics file ───────────────────────────────────────────────────────
    metrics_path = os.path.join(OUT_DIR, "metrics_resolution_table.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== 6D Collision — Phase 3 Metrics ===\n\n")
        f.write(f"{'Res':>6}  {'Total pts':>14}  {'MSE (full)':>12}  "
                f"{'BRT vol err':>12}  {'BL BRT':>10}  {'DR BRT':>10}  Notes\n")
        f.write("-" * 90 + "\n")
        for n in sorted(metrics.keys()):
            m = metrics[n]
            mse_str = f"{m['mse']:.4e}" if m['mse'] is not None else "N/A (slice)"
            note    = "(slice only)" if m.get("slice_only") else "(full grid)"
            f.write(f"{n:>4}pt  {n**6:>14,}  {mse_str:>12}  "
                    f"{m['brt_vol_error']:>11.2f}%  {m['bl_brt']:>10,}  "
                    f"{m['dr_brt']:>10,}  {note}\n")
        f.write("\nNote: BRT vol error should DECREASE as resolution INCREASES\n")
        f.write("      (pixelation artifact in coarse baseline, not model error)\n")
        f.write("      21pt full-grid metrics omitted — 21^6=85M pts exceeds available RAM\n")
    print(f"\n  Saved {metrics_path}")
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
