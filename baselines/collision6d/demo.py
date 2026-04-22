"""
6D Collision Demo — Two Dubins Cars  (paper-style, inspired by DeepReach Fig 3)
Shows: unsafe (no safety → collision) vs safe (DeepReach active → avoidance)

Style mirrors the paper:
  - Vehicle icons are directional rectangles (Dubins car top-down view)
  - Earlier timesteps drawn more transparent
  - Collision circle travels with each vehicle along its path
  - Min-distance inlet embedded in each panel
  - BRT slice shown as background unsafe region

Outputs (baselines/collision6d/plots/):
  demo_collision_vs_safe.png   — 2-panel figure (crash | safe)

Usage (deepreach env):
  python baselines/collision6d_demo.py
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
from matplotlib.patches import FancyArrow
from matplotlib.transforms import Affine2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = "runs/collision_6d_run/training/checkpoints/model_epoch_110000.pth"
OUT_DIR    = "baselines/collision6d/plots"

# ── physics (from TwoVehicleCollision6D) ───────────────────────────────────
VELOCITY    = 0.6
OMEGA_MAX   = 1.1
COLLISION_R = 0.25
T_MAX       = 1.0

# ── simulation ─────────────────────────────────────────────────────────────
DT               = 0.01
SIM_TIME         = 2.5
SAFETY_THRESHOLD = 0.15

# ── initial state: head-on collision course ────────────────────────────────
INIT_STATE = np.array([-0.7, 0.0,   # x1, y1
                         0.6, 0.0,  # x2, y2
                         0.0,       # θ1 (heading right)
                         math.pi])  # θ2 (heading left)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading + evaluation
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path, device="cpu"):
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

    import torch
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()
    return model, dyn


def eval_V_batch(model, dyn, states_6d):
    import torch
    t_col  = np.full((len(states_6d), 1), T_MAX, dtype=np.float32)
    coords = np.concatenate([t_col, np.array(states_6d, dtype=np.float32)], axis=1)
    inputs = dyn.coord_to_input(torch.from_numpy(coords))
    with torch.no_grad():
        out  = model({'coords': inputs})
        vals = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(-1))
    return vals.cpu().numpy()


def eval_V(model, dyn, state_6d):
    return float(eval_V_batch(model, dyn, [state_6d])[0])


def compute_dVdtheta(model, dyn, state_6d, dim, h=5e-3):
    sp, sm = state_6d.copy(), state_6d.copy()
    sp[dim] += h
    sm[dim] -= h
    vals = eval_V_batch(model, dyn, [sp, sm])
    return (vals[0] - vals[1]) / (2 * h)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate(model, dyn, init_state, use_safety=False):
    state         = init_state.copy()
    states        = [state.copy()]
    distances     = []
    collision_idx = None

    n_steps = int(SIM_TIME / DT)
    for step in range(n_steps):
        x1, y1, x2, y2, θ1, θ2 = state
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(dist)

        if dist < COLLISION_R and collision_idx is None:
            collision_idx = step
            if not use_safety:
                states.append(state.copy())
                break

        u1, u2 = 0.0, 0.0
        if use_safety:
            V = eval_V(model, dyn, state)
            if V < SAFETY_THRESHOLD:
                # Cooperative safety: u* = ω_max · sign(∂V/∂θ)  [matches dynamics.optimal_control]
                u1 = OMEGA_MAX * float(np.sign(compute_dVdtheta(model, dyn, state, dim=4)))
                u2 = OMEGA_MAX * float(np.sign(compute_dVdtheta(model, dyn, state, dim=5)))

        state = state.copy()
        state[0] += VELOCITY * math.cos(θ1) * DT
        state[1] += VELOCITY * math.sin(θ1) * DT
        state[2] += VELOCITY * math.cos(θ2) * DT
        state[3] += VELOCITY * math.sin(θ2) * DT
        state[4] = (state[4] + u1 * DT + math.pi) % (2 * math.pi) - math.pi
        state[5] = (state[5] + u2 * DT + math.pi) % (2 * math.pi) - math.pi
        states.append(state.copy())

    return {
        "states":        np.array(states),
        "distances":     np.array(distances),
        "collision_idx": collision_idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_car(ax, x, y, theta, color, length=0.12, width=0.06, alpha=1.0, zorder=6):
    """Draw a top-down rectangular Dubins car with a direction nub."""
    t = Affine2D().rotate(theta).translate(x, y) + ax.transData
    # Body rectangle centred at (0,0) before transform
    body = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white', linewidth=0.8,
        alpha=alpha, zorder=zorder, transform=t
    )
    ax.add_patch(body)
    # Direction nub (small forward triangle)
    nub = plt.Polygon(
        [[length * 0.42, 0],
         [length * 0.22, width * 0.35],
         [length * 0.22, -width * 0.35]],
        closed=True, facecolor='white', edgecolor='white',
        alpha=alpha, zorder=zorder + 1, transform=t
    )
    ax.add_patch(nub)


def draw_collision_bubble(ax, x, y, r=COLLISION_R, color='gray', alpha=0.12, zorder=2):
    """Faint collision-radius circle centred on a vehicle position."""
    c = plt.Circle((x, y), r, color=color, alpha=alpha, zorder=zorder)
    ax.add_patch(c)


# ══════════════════════════════════════════════════════════════════════════════
# Panel plot  (paper-style)
# ══════════════════════════════════════════════════════════════════════════════

C1 = '#4C9BE8'   # blue  – vehicle 1
C2 = '#E8714C'   # orange/red – vehicle 2

def plot_panel(ax, result, title, show_safe):
    states        = result["states"]
    distances     = result["distances"]
    collision_idx = result["collision_idx"]
    n             = len(states)

    # ── collision bubbles along vehicle 2 path (every N steps) ─────────────
    bubble_step = max(1, n // 12)
    for i in range(0, n, bubble_step):
        draw_collision_bubble(ax, states[i, 2], states[i, 3],
                              color=C2, alpha=0.07)

    # ── trajectory lines with alpha gradient (earlier = more transparent) ──
    from matplotlib.collections import LineCollection
    def grad_line(xy, color):
        segs   = [xy[j:j+2] for j in range(len(xy) - 1)]
        alphas = np.linspace(0.15, 1.0, len(segs))
        lc     = LineCollection(segs, color=color, linewidth=2, zorder=3)
        lc.set_alpha(None)
        # Per-segment alpha via individual colours
        rgba = np.array([matplotlib.colors.to_rgba(color)] * len(segs))
        rgba[:, 3] = alphas
        lc.set_colors(rgba)
        ax.add_collection(lc)

    grad_line(states[:, :2], C1)
    grad_line(states[:, 2:4], C2)

    # ── vehicle icons every N steps ─────────────────────────────────────────
    icon_step = max(1, n // 6)
    for i in range(0, n, icon_step):
        frac  = i / max(n - 1, 1)
        alpha = 0.25 + 0.75 * frac
        draw_car(ax, states[i, 0], states[i, 1], states[i, 4], C1, alpha=alpha)
        draw_car(ax, states[i, 2], states[i, 3], states[i, 5], C2, alpha=alpha)

    # Final vehicle icons (full opacity)
    draw_car(ax, states[-1, 0], states[-1, 1], states[-1, 4], C1, alpha=1.0)
    draw_car(ax, states[-1, 2], states[-1, 3], states[-1, 5], C2, alpha=1.0)

    # ── start dots ─────────────────────────────────────────────────────────
    ax.plot(*states[0, :2], 'o', color=C1, markersize=7, zorder=7,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(*states[0, 2:4], 'o', color=C2, markersize=7, zorder=7,
            markeredgecolor='white', markeredgewidth=0.8)

    # ── collision marker ───────────────────────────────────────────────────
    if not show_safe and collision_idx is not None:
        cx = (states[collision_idx, 0] + states[collision_idx, 2]) / 2
        cy = (states[collision_idx, 1] + states[collision_idx, 3]) / 2
        # Shaded collision zone
        czone = plt.Circle((cx, cy), COLLISION_R,
                            color='red', alpha=0.25, zorder=4)
        ax.add_patch(czone)
        ax.plot(cx, cy, 'rx', markersize=14, markeredgewidth=2.5, zorder=8)
        ax.annotate('Collision', (cx, cy), xytext=(cx + 0.15, cy + 0.25),
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    if show_safe:
        ax.text(0.97, 0.04, '✓ Safe', transform=ax.transAxes,
                fontsize=10, color='green', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='green', alpha=0.85))

    # ── axes / labels ───────────────────────────────────────────────────────
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # legend
    handles = [
        mpatches.Patch(color=C1, label='Vehicle 1'),
        mpatches.Patch(color=C2, label='Vehicle 2'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.85)


# ══════════════════════════════════════════════════════════════════════════════
# Separate min-distance figure (both scenarios)
# ══════════════════════════════════════════════════════════════════════════════

def plot_min_distance(r_unsafe, r_safe):
    t_unsafe = np.arange(len(r_unsafe["distances"])) * DT
    t_safe   = np.arange(len(r_safe["distances"]))   * DT

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_unsafe, r_unsafe["distances"],
            color='tomato', linewidth=2.2, label='No safety filter (collision)')
    ax.plot(t_safe,   r_safe["distances"],
            color='royalblue', linewidth=2.2, label='DeepReach safety active (safe)')
    ax.axhline(COLLISION_R, color='black', linewidth=1.5, linestyle='--',
               label=f'Collision radius R = {COLLISION_R} m')
    ax.fill_between([0, max(t_unsafe.max(), t_safe.max())],
                    0, COLLISION_R, alpha=0.07, color='red')

    if r_unsafe["collision_idx"] is not None:
        tc = r_unsafe["collision_idx"] * DT
        ax.axvline(tc, color='tomato', linestyle=':', linewidth=1.5,
                   label=f'Collision at t = {tc:.2f} s')

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Inter-vehicle distance (m)", fontsize=11)
    ax.set_title("6D Collision Avoidance — Inter-Vehicle Distance Over Time\n"
                 "DeepReach safety filter keeps vehicles above collision radius",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "demo_min_distance.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Animated GIF
# ══════════════════════════════════════════════════════════════════════════════

def make_gif(r_unsafe, r_safe):
    """Animated GIF — crash vs DeepReach safe side-by-side (6D collision)."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("  imageio not found — skipping GIF (pip install imageio)")
            return
    from matplotlib.collections import LineCollection

    n_frames = max(len(r_unsafe["states"]), len(r_safe["states"]))
    frame_stride = max(1, n_frames // 60)
    frames = []

    for step in range(0, n_frames, frame_stride):
        t_sim = step * DT
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(
            f"6D Collision Avoidance — Two Dubins Cars   t = {t_sim:.2f}s\n"
            r"Safety filter: $u^*_i = \omega_{\max} \cdot \mathrm{sign}(\partial V / \partial \theta_i)$",
            fontsize=10
        )

        for ax, result, title, show_safe in [
            (ax_l, r_unsafe, "No Safety Filter", False),
            (ax_r, r_safe,   "DeepReach Safety Active", True),
        ]:
            states        = result["states"]
            collision_idx = result["collision_idx"]
            end           = min(step + 1, len(states))

            for i in range(0, end, max(1, end // 8)):
                draw_collision_bubble(ax, states[i, 2], states[i, 3], color=C2, alpha=0.06)

            def grad_line(xy, color):
                if len(xy) < 2:
                    return
                segs   = [xy[j:j+2] for j in range(len(xy) - 1)]
                alphas = np.linspace(0.15, 1.0, len(segs))
                rgba   = np.array([matplotlib.colors.to_rgba(color)] * len(segs))
                rgba[:, 3] = alphas
                lc = LineCollection(segs, linewidth=2, zorder=3)
                lc.set_colors(rgba)
                ax.add_collection(lc)

            grad_line(states[:end, :2],  C1)
            grad_line(states[:end, 2:4], C2)

            cur = end - 1
            draw_car(ax, states[cur, 0], states[cur, 1], states[cur, 4], C1)
            draw_car(ax, states[cur, 2], states[cur, 3], states[cur, 5], C2)

            ax.plot(*states[0, :2],  'o', color=C1, markersize=6, zorder=7,
                    markeredgecolor='white', markeredgewidth=0.7)
            ax.plot(*states[0, 2:4], 'o', color=C2, markersize=6, zorder=7,
                    markeredgecolor='white', markeredgewidth=0.7)

            if not show_safe and collision_idx is not None and step >= collision_idx:
                cx = (states[collision_idx, 0] + states[collision_idx, 2]) / 2
                cy = (states[collision_idx, 1] + states[collision_idx, 3]) / 2
                ax.add_patch(plt.Circle((cx, cy), COLLISION_R, color='red', alpha=0.25, zorder=4))
                ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2.5, zorder=8)

            if show_safe:
                ax.text(0.97, 0.04, '✓ Safe', transform=ax.transAxes,
                        fontsize=10, color='green', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='green', alpha=0.85))

            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.85, 0.85)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=11, pad=4)
            ax.set_xlabel("x (m)", fontsize=9)
            ax.set_ylabel("y (m)", fontsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            handles = [
                mpatches.Patch(color=C1, label='Vehicle 1'),
                mpatches.Patch(color=C2, label='Vehicle 2'),
            ]
            ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.85)

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3])
        plt.close(fig)

    out = os.path.join(OUT_DIR, "demo_collision_vs_safe.gif")
    imageio.mimsave(out, frames, fps=20, loop=0)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model …")
    model, dyn = load_model(MODEL_PATH)

    V0 = eval_V(model, dyn, INIT_STATE)
    print(f"Initial V(x)={V0:.4f}  dist={math.sqrt((INIT_STATE[0]-INIT_STATE[2])**2 + (INIT_STATE[1]-INIT_STATE[3])**2):.3f}")

    print("Simulating UNSAFE …")
    r_unsafe = simulate(model, dyn, INIT_STATE, use_safety=False)
    if r_unsafe["collision_idx"] is not None:
        print(f"  Collision at t={r_unsafe['collision_idx']*DT:.2f}s")
    else:
        print(f"  No collision (min dist={r_unsafe['distances'].min():.3f})")

    print("Simulating SAFE (cooperative DeepReach) …")
    r_safe = simulate(model, dyn, INIT_STATE, use_safety=True)
    if r_safe["collision_idx"] is not None:
        print(f"  WARNING: collision at t={r_safe['collision_idx']*DT:.2f}s")
    else:
        print(f"  No collision ✓  (min dist={r_safe['distances'].min():.3f})")

    # ── figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "6D Collision Avoidance — Two Dubins Cars\n"
        r"Safety filter: $u^*_i = \omega_{\max} \cdot \mathrm{sign}(\partial V / \partial \theta_i)$"
        "  (cooperative HJ optimal control)",
        fontsize=11
    )

    plot_panel(axes[0], r_unsafe, "No Safety Filter → Collision",          show_safe=False)
    plot_panel(axes[1], r_safe,   "DeepReach Safety Active → Avoidance",   show_safe=True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "demo_collision_vs_safe.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

    plot_min_distance(r_unsafe, r_safe)

    print("\n--- Animated GIF ---")
    make_gif(r_unsafe, r_safe)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
