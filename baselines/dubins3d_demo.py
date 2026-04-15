"""
Phase 2D — Dubins3D Safety Simulation Demo

Single Dubins car heading into an unsafe obstacle zone.
Compares two controllers:
  • Crash  — nominal control (u = 0, straight ahead), no safety
  • Safe   — DeepReach safety override: when V(x) < threshold use
             u* = -ω_max · sign(∂V/∂θ)  to steer away from the BRT

Outputs (baselines/plots/demo/):
  dubins3d_demo_snapshots.png    — 2 rows × 5 time-step panels (crash | safe)
  dubins3d_demo_3scenarios.png   — 3 approach scenarios, crash vs safe
  dubins3d_demo.gif              — animated side-by-side (requires imageio/pillow)

Usage (deepreach env):
  cd ~/deepreach_CMPT419
  python baselines/dubins3d_demo.py
"""

import sys
import os
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "runs/dubins3d_tutorial_run/training/checkpoints/model_final.pth"
OUT_DIR    = "baselines/plots/demo"

# ── Dubins3D physical params ─────────────────────────────────────────────────
GRID_POINTS  = 101
X_BOUNDS     = (-1.0,  1.0)
Y_BOUNDS     = (-1.0,  1.0)
THETA_BOUNDS = (-math.pi, math.pi)
VELOCITY     = 0.6
OMEGA_MAX    = 1.1
GOAL_R       = 0.25
T_MAX        = 1.0

# ── Simulation params ─────────────────────────────────────────────────────────
DT             = 0.02
SIM_TIME       = 4.0              # seconds
SAFETY_THRESH  = 0.15             # safety override when V < this

# ── 3 crash-course scenarios ──────────────────────────────────────────────────
# (x0, y0, theta0, label)
SCENARIOS = [
    (0.80,  0.00,  math.pi,         "Head-on"),
    (0.00, -0.80,  math.pi / 2,     "From Below"),
    (0.60, -0.60,  3 * math.pi / 4, "Diagonal"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading — identical to dubins3d_analysis.py
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
        dyn = Dubins3D(goalR=GOAL_R, velocity=VELOCITY, omega_max=OMEGA_MAX,
                       angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False)
        dyn.deepreach_model = "exact"
        model = SingleBVPNet(in_features=4, out_features=1,
                             type='sine', mode='mlp',
                             hidden_features=512, num_hidden_layers=3)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
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

    import torch as th
    coords_tensor = th.from_numpy(coords_with_time).float().to(device)
    inputs = dyn.coord_to_input(coords_tensor)

    values_list = []
    with th.no_grad():
        for i in range(0, len(inputs), 50_000):
            batch = inputs[i:i + 50_000]
            out   = model({'coords': batch})
            vals  = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(-1))
            values_list.append(vals.cpu().numpy())

    return np.concatenate(values_list).reshape(GRID_POINTS, GRID_POINTS, GRID_POINTS)


# ══════════════════════════════════════════════════════════════════════════════
# Interpolators
# ══════════════════════════════════════════════════════════════════════════════

def build_interpolators(V_grid):
    x_lin = np.linspace(*X_BOUNDS,     GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS,     GRID_POINTS)
    t_lin = np.linspace(*THETA_BOUNDS, GRID_POINTS)

    dx = (X_BOUNDS[1]     - X_BOUNDS[0])     / (GRID_POINTS - 1)
    dy = (Y_BOUNDS[1]     - Y_BOUNDS[0])     / (GRID_POINTS - 1)
    dt = (THETA_BOUNDS[1] - THETA_BOUNDS[0]) / (GRID_POINTS - 1)

    _, _, dVdtheta_grid = np.gradient(V_grid, dx, dy, dt)

    kw = dict(bounds_error=False, fill_value=None)
    V_itp       = RegularGridInterpolator((x_lin, y_lin, t_lin), V_grid,       **kw)
    dVdtheta_itp = RegularGridInterpolator((x_lin, y_lin, t_lin), dVdtheta_grid, **kw)

    return V_itp, dVdtheta_itp, V_grid, dVdtheta_grid


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

def wrap_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def query_pt(itp, x, y, theta):
    pt = np.array([[
        float(np.clip(x,     X_BOUNDS[0],     X_BOUNDS[1])),
        float(np.clip(y,     Y_BOUNDS[0],     Y_BOUNDS[1])),
        float(np.clip(theta, THETA_BOUNDS[0], THETA_BOUNDS[1])),
    ]])
    return float(itp(pt)[0])


def simulate(x0, y0, theta0, use_safety, V_itp, dVdtheta_itp,
             nominal_u=0.0, dt=DT, sim_time=SIM_TIME):
    """
    Simulate a Dubins car from (x0, y0, theta0).
    Returns arrays of states and value function along trajectory.
    """
    steps = int(sim_time / dt)
    xs     = [x0]
    ys     = [y0]
    thetas = [theta0]
    vs     = []
    crash_step = None

    x, y, theta = x0, y0, theta0

    for step in range(steps):
        V_val = query_pt(V_itp, x, y, theta)
        vs.append(V_val)

        # Detect crash (enter obstacle)
        if crash_step is None and math.hypot(x, y) < GOAL_R:
            crash_step = step

        if use_safety and V_val < SAFETY_THRESH:
            dVdt = query_pt(dVdtheta_itp, x, y, theta)
            # set_mode='avoid': optimal safety control is +ω_max * sign(∂V/∂θ)
            # (matches dynamics.py Dubins3D.optimal_control for avoid mode)
            u = +OMEGA_MAX * float(np.sign(dVdt)) if abs(dVdt) > 1e-9 else nominal_u
        else:
            u = nominal_u

        x     = x     + dt * VELOCITY * math.cos(theta)
        y     = y     + dt * VELOCITY * math.sin(theta)
        theta = wrap_angle(theta + dt * u)

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

        # Stop simulation if clearly out of grid
        if abs(x) > 1.3 or abs(y) > 1.3:
            break

    return (np.array(xs), np.array(ys), np.array(thetas),
            np.array(vs), crash_step)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_car(ax, x, y, theta, color, size=0.07, alpha=1.0, zorder=12):
    """Top-down directional rectangle with heading arrow."""
    length, width = size, size * 0.55
    t = Affine2D().rotate(theta).translate(x, y) + ax.transData

    body = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.006",
        facecolor=color, edgecolor='white', linewidth=0.8,
        alpha=alpha, zorder=zorder, transform=t
    )
    ax.add_patch(body)

    nub = mpatches.Polygon(
        [[length * 0.45, 0],
         [length * 0.22, width * 0.42],
         [length * 0.22, -width * 0.42]],
        closed=True, facecolor='white', alpha=0.9 * alpha,
        zorder=zorder + 1, transform=t
    )
    ax.add_patch(nub)


def draw_obstacle(ax):
    ax.add_patch(plt.Circle((0, 0), GOAL_R, color='red', alpha=0.20, zorder=4))
    ax.add_patch(plt.Circle((0, 0), GOAL_R, fill=False, color='red',
                             linewidth=1.8, zorder=5))
    ax.text(0, 0, "unsafe", ha='center', va='center',
            fontsize=6.5, color='darkred', zorder=6, style='italic')


def setup_panel(ax, title, V_sl=None, x_lin=None, y_lin=None, show_vf=True):
    ax.set_xlim(*X_BOUNDS)
    ax.set_ylim(*Y_BOUNDS)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)

    if show_vf and V_sl is not None:
        ax.imshow(V_sl.T, origin='lower', extent=[*X_BOUNDS, *Y_BOUNDS],
                  cmap='coolwarm', alpha=0.45, vmin=-0.4, vmax=0.4,
                  aspect='auto', zorder=1)
        ax.contour(x_lin, y_lin, V_sl.T, levels=[0],
                   colors='black', linewidths=1.2, zorder=3)

    draw_obstacle(ax)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Snapshot panels (head-on scenario, 2 rows × 5 time steps)
# ══════════════════════════════════════════════════════════════════════════════

def plot_snapshots(V_grid, V_itp, dVdtheta_itp):
    x0, y0, theta0, sc_label = SCENARIOS[0]

    xs_c, ys_c, thetas_c, vs_c, crash_c = simulate(
        x0, y0, theta0, False, V_itp, dVdtheta_itp)
    xs_s, ys_s, thetas_s, vs_s, crash_s = simulate(
        x0, y0, theta0, True,  V_itp, dVdtheta_itp)

    x_lin = np.linspace(*X_BOUNDS,     GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS,     GRID_POINTS)
    t_lin = np.linspace(*THETA_BOUNDS, GRID_POINTS)
    idx0  = int(np.argmin(np.abs(t_lin - theta0)))
    V_sl  = V_grid[:, :, idx0]

    n = min(len(xs_c), len(xs_s)) - 1
    snap_steps = [0, n // 4, n // 2, 3 * n // 4, n]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    fig.suptitle(
        f"Dubins3D Safety Demo — {sc_label}\n"
        r"Top: No Safety (u = 0)  |  Bottom: DeepReach Safety Active"
        f"   (override when V < {SAFETY_THRESH})",
        fontsize=11
    )

    for col, step in enumerate(snap_steps):
        t_sim = step * DT

        # ── Crash row (top) ──────────────────────────────────────────────
        ax = axes[0, col]
        is_crash_here = crash_c is not None and step >= crash_c
        title = f"t = {t_sim:.2f}s" + (" ✖ CRASH" if is_crash_here else "")
        setup_panel(ax, title, show_vf=False)

        end = min(step + 1, len(xs_c))
        if end > 1:
            ax.plot(xs_c[:end], ys_c[:end], '-', color='tomato',
                    linewidth=1.8, alpha=0.75, zorder=7)
        s = min(step, len(xs_c) - 1)
        draw_car(ax, xs_c[s], ys_c[s], thetas_c[s], color='tomato')
        if is_crash_here and crash_c is not None and crash_c < len(xs_c):
            ax.plot(xs_c[crash_c], ys_c[crash_c], 'X', color='darkred',
                    markersize=10, zorder=13)

        # ── Safe row (bottom) ────────────────────────────────────────────
        ax = axes[1, col]
        V_now = vs_s[min(step, len(vs_s) - 1)] if len(vs_s) > 0 else 0.0
        override = V_now < SAFETY_THRESH
        title2 = f"t = {t_sim:.2f}s" + (" [override]" if override else "")
        setup_panel(ax, title2, V_sl, x_lin, y_lin, show_vf=True)

        end = min(step + 1, len(xs_s))
        if end > 1:
            ax.plot(xs_s[:end], ys_s[:end], '-', color='steelblue',
                    linewidth=1.8, alpha=0.75, zorder=7)
        s = min(step, len(xs_s) - 1)
        draw_car(ax, xs_s[s], ys_s[s], thetas_s[s], color='steelblue')

        # V(x) annotation
        ax.text(0.03, 0.97,
                f"V = {V_now:.3f}" + (" ← override" if override else ""),
                transform=ax.transAxes, fontsize=7, va='top',
                color='darkred' if override else 'black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75))

    axes[0, 0].set_ylabel("No Safety\ny", fontsize=8.5, color='tomato', fontweight='bold')
    axes[1, 0].set_ylabel("DeepReach Safety\ny", fontsize=8.5, color='steelblue', fontweight='bold')

    legend_elems = [
        mpatches.Patch(facecolor='tomato',    label='Crash trajectory'),
        mpatches.Patch(facecolor='steelblue', label='Safe trajectory'),
        mpatches.Patch(facecolor='red', alpha=0.25, label='Unsafe set'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='BRT boundary (V=0)'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT_DIR, "dubins3d_demo_snapshots.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — 3 scenarios: crash vs safe (full trajectory, side-by-side rows)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3scenarios(V_grid, V_itp, dVdtheta_itp):
    x_lin = np.linspace(*X_BOUNDS,     GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS,     GRID_POINTS)
    t_lin = np.linspace(*THETA_BOUNDS, GRID_POINTS)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Dubins3D Safety Demo — Three Approach Scenarios\n"
        r"Top: No Safety (u = 0)  |  Bottom: DeepReach Safety Active",
        fontsize=12
    )

    for col, (x0, y0, theta0, sc_label) in enumerate(SCENARIOS):
        idx0 = int(np.argmin(np.abs(t_lin - theta0)))
        V_sl = V_grid[:, :, idx0]

        xs_c, ys_c, thetas_c, vs_c, crash_c = simulate(
            x0, y0, theta0, False, V_itp, dVdtheta_itp)
        xs_s, ys_s, thetas_s, vs_s, crash_s = simulate(
            x0, y0, theta0, True,  V_itp, dVdtheta_itp)

        crashed = crash_c is not None

        # ── Crash row ──────────────────────────────────────────────────────
        ax = axes[0, col]
        setup_panel(ax, f"{sc_label}" + ("  — CRASH" if crashed else "  — no crash"),
                    show_vf=False)
        ax.plot(xs_c, ys_c, '-', color='tomato', linewidth=2.0, alpha=0.85, zorder=7)
        draw_car(ax, xs_c[0],  ys_c[0],  thetas_c[0],  color='tomato',  alpha=0.5)
        draw_car(ax, xs_c[-1], ys_c[-1], thetas_c[-1], color='darkred')
        if crashed and crash_c < len(xs_c):
            ax.plot(xs_c[crash_c], ys_c[crash_c], 'X', color='darkred',
                    markersize=12, zorder=13, label='collision')

        # ── Safe row ───────────────────────────────────────────────────────
        ax = axes[1, col]
        safe_label = (f"{sc_label}  —  min V = {vs_s.min():.3f}"
                      if len(vs_s) else sc_label)
        setup_panel(ax, safe_label, V_sl, x_lin, y_lin, show_vf=True)
        ax.plot(xs_s, ys_s, '-', color='steelblue', linewidth=2.0, alpha=0.85, zorder=7)
        draw_car(ax, xs_s[0],  ys_s[0],  thetas_s[0],  color='steelblue', alpha=0.5)
        draw_car(ax, xs_s[-1], ys_s[-1], thetas_s[-1], color='navy')

        # Shade override segments
        if len(vs_s) > 0:
            for i, v in enumerate(vs_s):
                if v < SAFETY_THRESH and i + 1 < len(xs_s):
                    ax.plot(xs_s[i:i+2], ys_s[i:i+2], '-',
                            color='orange', linewidth=3, alpha=0.7, zorder=8)

    axes[0, 0].set_ylabel("No Safety\ny", fontsize=9, color='tomato', fontweight='bold')
    axes[1, 0].set_ylabel("DeepReach Safety\ny", fontsize=9, color='steelblue', fontweight='bold')

    legend_elems = [
        mpatches.Patch(facecolor='tomato',    label='Crash traj.'),
        mpatches.Patch(facecolor='steelblue', label='Safe traj.'),
        mpatches.Patch(facecolor='orange',    label='Safety override active'),
        mpatches.Patch(facecolor='red', alpha=0.25, label='Unsafe set'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='BRT boundary'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "dubins3d_demo_3scenarios.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Animated GIF (optional, head-on scenario)
# ══════════════════════════════════════════════════════════════════════════════

def make_gif(V_grid, V_itp, dVdtheta_itp, scenario_idx=0):
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("  imageio not found — skipping GIF (pip install imageio)")
            return

    x0, y0, theta0, sc_label = SCENARIOS[scenario_idx]

    xs_c, ys_c, thetas_c, vs_c, crash_c = simulate(
        x0, y0, theta0, False, V_itp, dVdtheta_itp)
    xs_s, ys_s, thetas_s, vs_s, crash_s = simulate(
        x0, y0, theta0, True,  V_itp, dVdtheta_itp)

    x_lin = np.linspace(*X_BOUNDS,     GRID_POINTS)
    y_lin = np.linspace(*Y_BOUNDS,     GRID_POINTS)
    t_lin = np.linspace(*THETA_BOUNDS, GRID_POINTS)
    idx0  = int(np.argmin(np.abs(t_lin - theta0)))
    V_sl  = V_grid[:, :, idx0]

    n_frames = min(len(xs_c), len(xs_s))
    frame_stride = 3   # render every 3 steps for a reasonable file size
    frames = []

    for step in range(0, n_frames, frame_stride):
        t_sim = step * DT
        fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle(
            f"Dubins3D Safety Demo — {sc_label}   t = {t_sim:.2f}s",
            fontsize=11
        )

        # Crash panel
        is_crashed = crash_c is not None and step >= crash_c
        setup_panel(ax_c,
                    f"No Safety{'  ✖ CRASH' if is_crashed else ''}",
                    show_vf=False)
        ax_c.plot(xs_c[:step+1], ys_c[:step+1], '-',
                  color='tomato', linewidth=2, alpha=0.8, zorder=7)
        s = min(step, len(xs_c) - 1)
        draw_car(ax_c, xs_c[s], ys_c[s], thetas_c[s], color='tomato')
        if is_crashed and crash_c < len(xs_c):
            ax_c.plot(xs_c[crash_c], ys_c[crash_c], 'X', color='darkred',
                      markersize=10, zorder=13)

        # Safe panel
        V_now = vs_s[min(step, len(vs_s) - 1)] if len(vs_s) > 0 else 0.0
        override = V_now < SAFETY_THRESH
        setup_panel(ax_s,
                    "DeepReach Safety" + (" [override]" if override else ""),
                    V_sl, x_lin, y_lin, show_vf=True)
        ax_s.plot(xs_s[:step+1], ys_s[:step+1], '-',
                  color='steelblue', linewidth=2, alpha=0.8, zorder=7)
        s = min(step, len(xs_s) - 1)
        draw_car(ax_s, xs_s[s], ys_s[s], thetas_s[s], color='steelblue')
        ax_s.text(0.03, 0.97, f"V = {V_now:.3f}",
                  transform=ax_s.transAxes, fontsize=9, va='top',
                  color='darkred' if override else 'black',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3])
        plt.close(fig)

    out = os.path.join(OUT_DIR, "dubins3d_demo.gif")
    imageio.mimsave(out, frames, fps=12, loop=0)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Evaluating DeepReach model on full grid …")
    V_grid = load_deepreach_values(MODEL_PATH)
    print(f"  shape={V_grid.shape}  range=[{V_grid.min():.4f}, {V_grid.max():.4f}]")

    print("Building interpolators …")
    V_itp, dVdtheta_itp, _, _ = build_interpolators(V_grid)

    print("\n--- Figure 1: Snapshot panels (head-on scenario) ---")
    plot_snapshots(V_grid, V_itp, dVdtheta_itp)

    print("\n--- Figure 2: 3 approach scenarios ---")
    plot_3scenarios(V_grid, V_itp, dVdtheta_itp)

    print("\n--- Figure 3: Animated GIF ---")
    make_gif(V_grid, V_itp, dVdtheta_itp, scenario_idx=0)

    print("\nPhase 2D complete.")


if __name__ == "__main__":
    main()
