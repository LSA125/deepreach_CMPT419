"""
Phase 2 — Dubins3D Full Analysis
Model:    runs/dubins3d_tutorial_run/training/checkpoints/model_final.pth
Baseline: baselines/grids/dubins3d_grid.npy (101pt)

Outputs (all in baselines/dubins3d/plots/):
  Per angle  (θ = 0, 45, 90, 180, -90):
    comparison_theta_{label}.png   — 2 panels: baseline V | DeepReach V
    brt_overlay_theta_{label}.png  — 1 panel:  BRT overlap (green/pink/orange)

  Summary:
    gradient_quiver_dubins3d.png   — 2 panels: θ=0 | θ=π/2
    control_field_dubins3d.png     — 2 panels: θ=0 | θ=π/2
    metrics_dubins3d.txt           — raw + norm MSE, BRT vol error,
                                     per-angle BRT vol error, control agreement %

Usage (deepreach env):
  python baselines/dubins3d_analysis.py
"""

import sys
import os
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "runs/dubins3d_tutorial_run/training/checkpoints/model_final.pth"
BASELINE_PATH = "baselines/grids/dubins3d_grid.npy"
OUT_DIR       = "baselines/dubins3d/plots"

# ── Dubins3D grid params ────────────────────────────────────────────────────
GRID_POINTS = 101
X_BOUNDS    = (-1.0,  1.0)
Y_BOUNDS    = (-1.0,  1.0)
THETA_BOUNDS = (-math.pi, math.pi)
OMEGA_MAX   = 1.1
T_MAX       = 1.0

THETA_CASES = [
    (0.0,           "0"),
    (math.pi/4,     "45"),
    (math.pi/2,     "90"),
    (math.pi,       "180"),
    (-math.pi/2,    "neg90"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_deepreach_values(model_path, device="cpu"):
    import torch
    from utils.modules import SingleBVPNet
    from dynamics.dynamics import Dubins3D

    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')

    if os.path.exists(opt_path):
        with open(opt_path, 'rb') as f:
            orig_opt = pickle.load(f)
        dyn = Dubins3D(goalR=orig_opt.goalR, velocity=orig_opt.velocity,
                       omega_max=orig_opt.omega_max,
                       angle_alpha_factor=orig_opt.angle_alpha_factor,
                       set_mode=orig_opt.set_mode, freeze_model=False)
        dyn.deepreach_model = orig_opt.deepreach_model
        model = SingleBVPNet(in_features=4, out_features=1,
                             type=orig_opt.model, mode=orig_opt.model_mode,
                             hidden_features=orig_opt.num_nl,
                             num_hidden_layers=orig_opt.num_hl)
    else:
        dyn = Dubins3D(goalR=0.25, velocity=0.6, omega_max=OMEGA_MAX,
                       angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False)
        dyn.deepreach_model = "exact"
        model = SingleBVPNet(in_features=4, out_features=1,
                             type='sine', mode='mlp',
                             hidden_features=512, num_hidden_layers=3)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()

    axes = [
        np.linspace(X_BOUNDS[0],     X_BOUNDS[1],     GRID_POINTS),
        np.linspace(Y_BOUNDS[0],     Y_BOUNDS[1],     GRID_POINTS),
        np.linspace(THETA_BOUNDS[0], THETA_BOUNDS[1], GRID_POINTS),
    ]
    X0, X1, X2 = np.meshgrid(*axes, indexing='ij')
    coords_flat = np.stack([X0.ravel(), X1.ravel(), X2.ravel()], axis=-1)
    t_col = np.full((len(coords_flat), 1), T_MAX)
    coords_with_time = np.concatenate([t_col, coords_flat], axis=-1)

    coords_tensor = torch.from_numpy(coords_with_time).float().to(device)
    inputs = dyn.coord_to_input(coords_tensor)

    values_list = []
    with torch.no_grad():
        for i in range(0, len(inputs), 50_000):
            batch = inputs[i:i+50_000]
            out   = model({'coords': batch})
            vals  = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(dim=-1))
            values_list.append(vals.cpu().numpy())

    return np.concatenate(values_list).reshape(GRID_POINTS, GRID_POINTS, GRID_POINTS)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def theta_to_idx(theta_val):
    grid = np.linspace(THETA_BOUNDS[0], THETA_BOUNDS[1], GRID_POINTS)
    return int(np.argmin(np.abs(grid - theta_val)))


def minmax_norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


def brt_vol_error(bl, dr):
    bl_brt = bl <= 0
    dr_brt = dr <= 0
    sym_diff = np.sum(bl_brt != dr_brt)
    bl_vol   = np.sum(bl_brt)
    return float(sym_diff / bl_vol * 100) if bl_vol > 0 else 0.0


def draw_dubins_car_icon(ax, cx, cy, theta, color='black', size=0.10, label=None):
    """Draw a small top-down Dubins car icon at (cx, cy) pointing in direction theta.
    Shows the heading of the vehicle for the current slice."""
    from matplotlib.transforms import Affine2D
    import matplotlib.patches as mpatches

    t = Affine2D().rotate(theta).translate(cx, cy) + ax.transData
    length, width = size, size * 0.55
    body = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor='white', linewidth=0.8,
        alpha=0.85, zorder=10, transform=t
    )
    ax.add_patch(body)
    # Direction arrow nub
    nub = plt.Polygon(
        [[length * 0.45, 0],
         [length * 0.22, width * 0.42],
         [length * 0.22, -width * 0.42]],
        closed=True, facecolor='white', alpha=0.9,
        zorder=11, transform=t
    )
    ax.add_patch(nub)
    if label:
        ax.text(cx, cy + size * 1.05, label,
                ha='center', va='top', fontsize=7,
                color=color, zorder=12,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.75))


# ══════════════════════════════════════════════════════════════════════════════
# Per-angle figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(baseline, deepreach, theta_val, label):
    idx    = theta_to_idx(theta_val)
    bl_sl  = baseline[:, :, idx]
    dr_sl  = deepreach[:, :, idx]
    vmin   = min(bl_sl.min(), dr_sl.min())
    vmax   = max(bl_sl.max(), dr_sl.max())

    x_lin  = np.linspace(*X_BOUNDS, GRID_POINTS)
    y_lin  = np.linspace(*Y_BOUNDS, GRID_POINTS)
    extent = [*X_BOUNDS, *Y_BOUNDS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    theta_str = f"θ = {math.degrees(theta_val):.0f}°"
    fig.suptitle(f"Dubins3D Value Function — {theta_str}", fontsize=12)

    for ax, data, title in zip(axes,
                                [bl_sl, dr_sl],
                                ["Ground Truth (optimized_dp)", "DeepReach"]):
        im = ax.imshow(data.T, origin='lower', extent=extent,
                       cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.contour(x_lin, y_lin, data.T, levels=[0],
                   colors='black', linewidths=1.5)
        plt.colorbar(im, ax=ax, shrink=0.85, label="V(x)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        # Vehicle icon showing heading at this slice
        draw_dubins_car_icon(ax, 0.72, 0.72, theta_val, color='#333333',
                             label=f"θ={math.degrees(theta_val):.0f}°")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"comparison_theta_{label}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_brt_overlay(baseline, deepreach, theta_val, label):
    idx   = theta_to_idx(theta_val)
    bl_sl = baseline[:, :, idx]
    dr_sl = deepreach[:, :, idx]

    bl_brt = bl_sl <= 0
    dr_brt = dr_sl <= 0

    sym_diff = np.sum(bl_brt != dr_brt)
    bl_vol   = np.sum(bl_brt)
    vol_err  = float(sym_diff / bl_vol * 100) if bl_vol > 0 else 0.0

    overlay = np.zeros((*bl_brt.shape, 3))
    overlay[bl_brt & dr_brt]   = [0.2, 0.8, 0.2]
    overlay[bl_brt & ~dr_brt]  = [1.0, 0.4, 0.7]
    overlay[~bl_brt & dr_brt]  = [1.0, 0.6, 0.2]

    theta_str = f"θ = {math.degrees(theta_val):.0f}°"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay.transpose(1, 0, 2), origin='lower', extent=[*X_BOUNDS, *Y_BOUNDS])
    ax.set_title(f"Dubins3D BRT Overlay — {theta_str}\n"
                 f"Vol Error: {vol_err:.1f}%  |  GT: {bl_vol} pts, DR: {int(np.sum(dr_brt))} pts",
                 fontsize=10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='Both agree (in BRT)'),
        Patch(facecolor=[1.0, 0.4, 0.7], label='Baseline only (missed)'),
        Patch(facecolor=[1.0, 0.6, 0.2], label='DeepReach only (false pos.)'),
        Patch(facecolor='black',          label='Both agree (outside BRT)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
    draw_dubins_car_icon(ax, 0.72, 0.72, theta_val, color='#333333',
                         label=f"θ={math.degrees(theta_val):.0f}°")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"brt_overlay_theta_{label}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

    return vol_err, int(bl_vol), int(np.sum(dr_brt))


# ══════════════════════════════════════════════════════════════════════════════
# Gradient quiver  (2-panel: θ=0 | θ=π/2)
# ══════════════════════════════════════════════════════════════════════════════

def plot_brt_isosurface(deepreach):
    """3-panel 3D isosurface of the BRT V=0 level set in (x, y, θ) state space."""
    try:
        from skimage import measure
    except ImportError:
        print("  skimage not available — skipping isosurface (pip install scikit-image)")
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(deepreach, level=0)
    except Exception as e:
        print(f"  marching_cubes failed: {e}")
        return

    x_lin = np.linspace(*X_BOUNDS,     GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS,     GRID_POINTS)
    t_lin = np.linspace(*THETA_BOUNDS, GRID_POINTS)

    vx = np.interp(verts[:, 0], np.arange(GRID_POINTS), x_lin)
    vy = np.interp(verts[:, 1], np.arange(GRID_POINTS), y_lin)
    vt = np.interp(verts[:, 2], np.arange(GRID_POINTS), t_lin)

    views = [
        (25,  45, "Isometric"),
        (90,   0, "Top-down  (x–y plane)"),
        ( 0,   0, "Side  (x–θ plane)"),
    ]

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Dubins3D BRT Isosurface  (V = 0 level set in state space)\n"
                 "Blue surface = boundary of unsafe set — BRT rotates with heading θ",
                 fontsize=11)

    for i, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_trisurf(vx, vy, vt, triangles=faces,
                        alpha=0.50, color='steelblue',
                        edgecolor='none', linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x",       fontsize=8, labelpad=2)
        ax.set_ylabel("y",       fontsize=8, labelpad=2)
        ax.set_zlabel("θ (rad)", fontsize=8, labelpad=2)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "brt_isosurface_dubins3d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_gradient_quiver(deepreach):
    dx = (X_BOUNDS[1]     - X_BOUNDS[0])     / (GRID_POINTS - 1)
    dy = (Y_BOUNDS[1]     - Y_BOUNDS[0])     / (GRID_POINTS - 1)
    dt = (THETA_BOUNDS[1] - THETA_BOUNDS[0]) / (GRID_POINTS - 1)
    dVdx, dVdy, _ = np.gradient(deepreach, dx, dy, dt)

    x_lin = np.linspace(*X_BOUNDS, GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS, GRID_POINTS)
    XX, YY = np.meshgrid(x_lin, y_lin, indexing='ij')
    stride = 5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Dubins3D DeepReach: $\nabla V$ Quiver  (arrows = gradient direction, colour = magnitude)",
                 fontsize=11)

    for ax, (theta_val, lbl) in zip(axes, [(0.0, r"$\theta=0$"), (math.pi/2, r"$\theta=\pi/2$")]):
        idx   = theta_to_idx(theta_val)
        V_sl  = deepreach[:, :, idx]
        gx_sl = dVdx[:, :, idx]
        gy_sl = dVdy[:, :, idx]

        im = ax.imshow(V_sl.T, origin='lower', extent=[*X_BOUNDS, *Y_BOUNDS],
                       cmap='coolwarm', alpha=0.85)
        plt.colorbar(im, ax=ax, shrink=0.8, label="V(x)")

        mag     = np.sqrt(gx_sl**2 + gy_sl**2)
        mag_max = mag.max() or 1.0
        ax.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
                  gx_sl[::stride, ::stride], gy_sl[::stride, ::stride],
                  mag[::stride, ::stride] / mag_max,
                  cmap='plasma', scale=20, width=0.004, alpha=0.8, clim=(0, 1))
        ax.contour(x_lin, y_lin, V_sl.T, levels=[0],
                   colors='black', linewidths=1.5)

        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gradient_quiver_dubins3d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Control field  (2-panel: θ=0 | θ=π/2)
# ══════════════════════════════════════════════════════════════════════════════

def plot_control_field(deepreach):
    """
    Dubins3D avoid mode: u* = +ω_max · sign(∂V/∂θ)  (matches dynamics.py optimal_control)
    +ω = turn left = blue, -ω = turn right = red
    """
    dx = (X_BOUNDS[1]     - X_BOUNDS[0])     / (GRID_POINTS - 1)
    dy = (Y_BOUNDS[1]     - Y_BOUNDS[0])     / (GRID_POINTS - 1)
    dt = (THETA_BOUNDS[1] - THETA_BOUNDS[0]) / (GRID_POINTS - 1)
    _, _, dVdt = np.gradient(deepreach, dx, dy, dt)

    x_lin = np.linspace(*X_BOUNDS, GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS, GRID_POINTS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Dubins3D Optimal Control $u^*$  (blue = turn left, red = turn right)",
                 fontsize=11)

    for ax, (theta_val, lbl) in zip(axes, [(0.0, r"$\theta=0$"), (math.pi/2, r"$\theta=\pi/2$")]):
        idx   = theta_to_idx(theta_val)
        V_sl  = deepreach[:, :, idx]
        gp_sl = dVdt[:, :, idx]

        u_opt = +OMEGA_MAX * np.sign(gp_sl)

        im = ax.imshow(u_opt.T, origin='lower', extent=[*X_BOUNDS, *Y_BOUNDS],
                       cmap='RdBu', vmin=-OMEGA_MAX, vmax=OMEGA_MAX, alpha=0.9)
        plt.colorbar(im, ax=ax, shrink=0.8, label=r"$u^*$ (rad/s)")
        ax.contour(x_lin, y_lin, V_sl.T, levels=[0],
                   colors='black', linewidths=1.5)

        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "control_field_dubins3d.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Control agreement metric
# ══════════════════════════════════════════════════════════════════════════════

def control_agreement(baseline, deepreach):
    dx = (X_BOUNDS[1]     - X_BOUNDS[0])     / (GRID_POINTS - 1)
    dy = (Y_BOUNDS[1]     - Y_BOUNDS[0])     / (GRID_POINTS - 1)
    dt = (THETA_BOUNDS[1] - THETA_BOUNDS[0]) / (GRID_POINTS - 1)

    _, _, bl_gp = np.gradient(baseline,  dx, dy, dt)
    _, _, dr_gp = np.gradient(deepreach, dx, dy, dt)

    ctrl_bl = -OMEGA_MAX * np.sign(bl_gp)
    ctrl_dr = -OMEGA_MAX * np.sign(dr_gp)

    valid = (ctrl_bl != 0) & (ctrl_dr != 0)
    return float(np.mean(ctrl_bl[valid] == ctrl_dr[valid]) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading baseline …")
    baseline = np.load(BASELINE_PATH)
    print(f"  shape={baseline.shape}  range=[{baseline.min():.4f}, {baseline.max():.4f}]")

    print("Evaluating DeepReach model …")
    deepreach = load_deepreach_values(MODEL_PATH)
    print(f"  shape={deepreach.shape}  range=[{deepreach.min():.4f}, {deepreach.max():.4f}]")

    # ── global metrics ─────────────────────────────────────────────────────
    raw_mse  = float(np.mean((baseline - deepreach) ** 2))
    raw_vol  = brt_vol_error(baseline, deepreach)
    bl_n     = minmax_norm(baseline)
    dr_n     = minmax_norm(deepreach)
    norm_mse = float(np.mean((bl_n - dr_n) ** 2))
    norm_vol = brt_vol_error(bl_n, dr_n)
    ctrl_agr = control_agreement(baseline, deepreach)

    print("\n=== Metrics ===")
    print(f"  Raw  MSE           : {raw_mse:.6e}")
    print(f"  Raw  BRT vol error : {raw_vol:.2f}%")
    print(f"  Norm MSE           : {norm_mse:.6e}")
    print(f"  Norm BRT vol error : {norm_vol:.2f}%")
    print(f"  Control agreement  : {ctrl_agr:.2f}%")

    # ── per-angle figures ──────────────────────────────────────────────────
    print("\n--- Per-angle figures ---")
    per_angle_rows = []
    for theta_val, label in THETA_CASES:
        plot_comparison(baseline, deepreach, theta_val, label)
        vol_err, bl_vol, dr_vol = plot_brt_overlay(baseline, deepreach, theta_val, label)
        per_angle_rows.append((label, theta_val, vol_err, bl_vol, dr_vol))

    # ── summary figures ────────────────────────────────────────────────────
    print("\n--- Summary figures ---")
    plot_brt_isosurface(deepreach)
    plot_gradient_quiver(deepreach)
    plot_control_field(deepreach)

    # ── metrics file ───────────────────────────────────────────────────────
    metrics_path = os.path.join(OUT_DIR, "metrics_dubins3d.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== Dubins3D — Phase 2 Metrics ===\n\n")
        f.write(f"raw_mse                 = {raw_mse:.6e}\n")
        f.write(f"raw_brt_vol_error_pct   = {raw_vol:.4f}\n")
        f.write(f"norm_mse                = {norm_mse:.6e}\n")
        f.write(f"norm_brt_vol_error_pct  = {norm_vol:.4f}\n")
        f.write(f"control_agreement_pct   = {ctrl_agr:.4f}\n")
        f.write("\nPer-angle BRT volume error:\n")
        for label, theta_val, vol_err, bl_vol, dr_vol in per_angle_rows:
            f.write(f"  theta={label:>6}  ({math.degrees(theta_val):+.1f} deg)  "
                    f"vol_err={vol_err:.2f}%  bl={bl_vol}  dr={dr_vol}\n")
    print(f"\n  Saved {metrics_path}")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
