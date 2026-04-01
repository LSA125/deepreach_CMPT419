'''
Compare optimized_dp ground-truth value function against DeepReach output.
Supports two modes:
  Mode A: --deepreach_grid  (compare .npy vs .npy, no GPU needed)
  Mode B: --deepreach_model (load .pth, evaluate on grid, then compare)
Reference to 'comparison.py' at root
'''

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_deepreach_from_model(model_path, dynamics_name, grid_bounds, grid_points, device="cpu"):
    """Load a .pth model and evaluate on the same grid as the baseline."""
    import torch
    from utils.modules import SingleBVPNet
    from dynamics.dynamics import Air3D, Dubins3D

    # Load training config from orig_opt.pickle if available
    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')

    if os.path.exists(opt_path):
        import pickle
        with open(opt_path, 'rb') as f:
            orig_opt = pickle.load(f)
        print(f"Loaded training config from {opt_path}")

        if dynamics_name == "air3d":
            dyn = Air3D(collisionR=orig_opt.collisionR, velocity=orig_opt.velocity,
                        omega_max=orig_opt.omega_max, angle_alpha_factor=orig_opt.angle_alpha_factor)
        elif dynamics_name == "dubins3d":
            dyn = Dubins3D(goalR=orig_opt.goalR, velocity=orig_opt.velocity,
                           omega_max=orig_opt.omega_max, angle_alpha_factor=orig_opt.angle_alpha_factor,
                           set_mode=orig_opt.set_mode, freeze_model=False)
        else:
            raise ValueError(f"Unknown dynamics: {dynamics_name}")
        dyn.deepreach_model = orig_opt.deepreach_model

        model = SingleBVPNet(in_features=4, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
    else:
        # Fallback to config.py defaults
        print("Warning: orig_opt.pickle not found, using config.py defaults")
        if dynamics_name == "air3d":
            from baselines.config import VELOCITY, OMEGA_MAX, BETA
            dyn = Air3D(collisionR=BETA, velocity=VELOCITY, omega_max=OMEGA_MAX, angle_alpha_factor=1.2)
        elif dynamics_name == "dubins3d":
            from baselines.config import DUBINS_VELOCITY, DUBINS_OMEGA_MAX, GOAL_R, DUBINS_ANGLE_ALPHA, DUBINS_SET_MODE
            dyn = Dubins3D(goalR=GOAL_R, velocity=DUBINS_VELOCITY, omega_max=DUBINS_OMEGA_MAX,
                           angle_alpha_factor=DUBINS_ANGLE_ALPHA, set_mode=DUBINS_SET_MODE, freeze_model=False)
        else:
            raise ValueError(f"Unknown dynamics: {dynamics_name}")
        dyn.deepreach_model = "exact"

        model = SingleBVPNet(in_features=4, out_features=1, type='sine', mode='mlp',
                             hidden_features=512, num_hidden_layers=3)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Build grid matching baseline
    axes = [np.linspace(b[0], b[1], grid_points) for b in grid_bounds]
    X0, X1, X2 = np.meshgrid(*axes, indexing='ij')
    coords_flat = np.stack([X0.ravel(), X1.ravel(), X2.ravel()], axis=-1)

    # Evaluate at t=T_MAX (final time)
    from baselines.config import T_MAX
    t_col = np.full((coords_flat.shape[0], 1), T_MAX)
    coords_with_time = np.concatenate([t_col, coords_flat], axis=-1)

    coords_tensor = torch.from_numpy(coords_with_time).float().to(device)
    inputs = dyn.coord_to_input(coords_tensor)

    batch_size = 50000
    values_list = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            out = model({'coords': batch})
            vals = dyn.io_to_value(out['model_in'], out['model_out'].squeeze(dim=-1))
            values_list.append(vals.cpu().numpy())

    values = np.concatenate(values_list).reshape(grid_points, grid_points, grid_points)
    return values


def compute_metrics(baseline, deepreach):
    """Compute MSE and BRT volume error."""
    mse = np.mean((baseline - deepreach) ** 2)

    baseline_brt = (baseline <= 0)
    deepreach_brt = (deepreach <= 0)
    symmetric_diff = np.sum(baseline_brt != deepreach_brt)
    baseline_volume = np.sum(baseline_brt)
    volume_error = (symmetric_diff / baseline_volume * 100) if baseline_volume > 0 else 0.0

    return {
        "mse": float(mse),
        "brt_volume_error_pct": float(volume_error),
        "baseline_brt_points": int(baseline_volume),
        "deepreach_brt_points": int(np.sum(deepreach_brt)),
        "symmetric_diff_points": int(symmetric_diff),
    }


def compute_gradient_control_metrics(baseline, deepreach, dynamics_name, grid_bounds):
    """
    Compute gradient MSE and optimal control agreement rate.

    Dubins3D (uMode='min'): u* = -wMax * sign(dV/dtheta)
    Air3D    (uMode='max'): a = dV/dx*y - dV/dy*x - dV/dpsi
                            u* = +wMax * sign(a)
    """
    N = baseline.shape[0]
    spacings = [(b[1] - b[0]) / (N - 1) for b in grid_bounds]

    dV_bl = np.gradient(baseline, *spacings)
    dV_dr = np.gradient(deepreach, *spacings)

    # Per-component gradient MSE, averaged across components
    grad_mse = float(np.mean(
        sum((dV_bl[i] - dV_dr[i]) ** 2 for i in range(3)) / 3
    ))

    if dynamics_name == 'dubins3d':
        wMax = 1.1
        ctrl_bl = -wMax * np.sign(dV_bl[2])
        ctrl_dr = -wMax * np.sign(dV_dr[2])
    elif dynamics_name == 'air3d':
        wMax = 3.0
        axes = [np.linspace(b[0], b[1], N) for b in grid_bounds]
        X, Y, _ = np.meshgrid(*axes, indexing='ij')
        a_bl = dV_bl[0] * Y - dV_bl[1] * X - dV_bl[2]
        a_dr = dV_dr[0] * Y - dV_dr[1] * X - dV_dr[2]
        ctrl_bl = wMax * np.sign(a_bl)
        ctrl_dr = wMax * np.sign(a_dr)
    else:
        return None

    # Agreement rate over non-zero (non-tie) points
    valid = (ctrl_bl != 0) & (ctrl_dr != 0)
    agreement = float(np.mean(ctrl_bl[valid] == ctrl_dr[valid]) * 100)

    return {
        "gradient_mse": grad_mse,
        "control_agreement_pct": agreement,
        "dV_baseline": dV_bl,
        "dV_deepreach": dV_dr,
        "ctrl_baseline": ctrl_bl,
        "ctrl_deepreach": ctrl_dr,
    }


def plot_control_comparison(ctrl_metrics, bounds_x, bounds_y, slice_dim, slice_idx,
                             slice_value, dynamics_name, output_dir):
    """3-panel plot: baseline control | DeepReach control | disagreement map."""
    ctrl_bl = ctrl_metrics['ctrl_baseline']
    ctrl_dr = ctrl_metrics['ctrl_deepreach']

    dim_labels = {
        "air3d":    ["Relative X", "Relative Y", r"Relative Heading $\psi$"],
        "dubins3d": ["X Position", "Y Position", r"Heading $\theta$"],
    }
    labels = dim_labels.get(dynamics_name, ["Dim 0", "Dim 1", "Dim 2"])

    def _slice(arr):
        if slice_dim == 2:
            return arr[:, :, slice_idx], labels[0], labels[1], labels[2]
        elif slice_dim == 1:
            return arr[:, slice_idx, :], labels[0], labels[2], labels[1]
        else:
            return arr[slice_idx, :, :], labels[1], labels[2], labels[0]

    bl_sl, xlabel, ylabel, slice_label = _slice(ctrl_bl)
    dr_sl, _, _, _ = _slice(ctrl_dr)
    disagree = (bl_sl != dr_sl).astype(float)
    # mask tie points (both zero) as NaN so they appear neutral
    tie_mask = (bl_sl == 0) & (dr_sl == 0)
    disagree[tie_mask] = np.nan

    agreement_pct = ctrl_metrics['control_agreement_pct']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Optimal Control Comparison ({dynamics_name.upper()}, {slice_label} = {slice_value:.2f})\n"
        f"Control agreement: {agreement_pct:.1f}%  |  Gradient MSE: {ctrl_metrics['gradient_mse']:.4f}",
        fontsize=12
    )

    extent = [*bounds_x, *bounds_y]
    for ax, data, title, cmap in zip(
        axes,
        [bl_sl.T, dr_sl.T, disagree.T],
        ["Baseline control $u^*$", "DeepReach control $u^*$", "Disagreement (1 = differ)"],
        ['RdBu', 'RdBu', 'Reds'],
    ):
        im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap,
                       vmin=-1 if cmap == 'RdBu' else 0,
                       vmax=1 if cmap == 'RdBu' else 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'control_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_slice_comparison(baseline, deepreach, bounds_x, bounds_y, slice_dim, slice_idx, slice_value, dynamics_name, output_dir):
    """3-panel heatmap: baseline | deepreach | absolute error at a fixed slice."""
    dim_labels = {
        "air3d": ["Relative X", "Relative Y", r"Relative Heading $\psi$"],
        "dubins3d": ["X Position", "Y Position", r"Heading $\theta$"],
    }
    labels = dim_labels.get(dynamics_name, ["Dim 0", "Dim 1", "Dim 2"])

    if slice_dim == 2:
        bl_slice = baseline[:, :, slice_idx]
        dr_slice = deepreach[:, :, slice_idx]
        xlabel, ylabel = labels[0], labels[1]
        slice_label = labels[2]
    elif slice_dim == 1:
        bl_slice = baseline[:, slice_idx, :]
        dr_slice = deepreach[:, slice_idx, :]
        xlabel, ylabel = labels[0], labels[2]
        slice_label = labels[1]
    else:
        bl_slice = baseline[slice_idx, :, :]
        dr_slice = deepreach[slice_idx, :, :]
        xlabel, ylabel = labels[1], labels[2]
        slice_label = labels[0]

    error_slice = np.abs(bl_slice - dr_slice)
    vmin = min(bl_slice.min(), dr_slice.min())
    vmax = max(bl_slice.max(), dr_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Value Function Comparison ({dynamics_name.upper()}, {slice_label} = {slice_value:.2f})\n"
                 f"V(x) < 0: inside BRT (reachable)  |  V(x) > 0: outside BRT (not reachable)",
                 fontsize=12)

    titles = [
        "Ground Truth (optimized_dp)",
        "DeepReach (neural approx.)",
        f"|Error|  (MSE={np.mean(error_slice**2):.4f})",
    ]
    for ax, data, title in zip(axes, [bl_slice, dr_slice, error_slice], titles):
        if "Error" in title:
            im = ax.imshow(data.T, origin='lower', extent=[*bounds_x, *bounds_y], cmap='hot')
        else:
            im = ax.imshow(data.T, origin='lower', extent=[*bounds_x, *bounds_y], cmap='coolwarm',
                          vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        if "Error" not in title:
            cbar.set_label("V(x)")
        else:
            cbar.set_label("|V_baseline - V_deepreach|")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'slice_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/slice_comparison.png")


def plot_brt_overlay(baseline, deepreach, bounds_x, bounds_y, slice_dim, slice_idx, slice_value, dynamics_name, metrics, output_dir):
    """BRT boundary overlay: green = agreement, pink/orange = disagreement."""
    dim_labels = {
        "air3d": ["Relative X", "Relative Y", r"Relative Heading $\psi$"],
        "dubins3d": ["X Position", "Y Position", r"Heading $\theta$"],
    }
    labels = dim_labels.get(dynamics_name, ["Dim 0", "Dim 1", "Dim 2"])

    if slice_dim == 2:
        bl_slice = baseline[:, :, slice_idx]
        dr_slice = deepreach[:, :, slice_idx]
        xlabel, ylabel = labels[0], labels[1]
        slice_label = labels[2]
    elif slice_dim == 1:
        bl_slice = baseline[:, slice_idx, :]
        dr_slice = deepreach[:, slice_idx, :]
        xlabel, ylabel = labels[0], labels[2]
        slice_label = labels[1]
    else:
        bl_slice = baseline[slice_idx, :, :]
        dr_slice = deepreach[slice_idx, :, :]
        xlabel, ylabel = labels[1], labels[2]
        slice_label = labels[0]

    bl_brt = bl_slice <= 0
    dr_brt = dr_slice <= 0

    overlay = np.zeros((*bl_brt.shape, 3))
    both = bl_brt & dr_brt
    only_bl = bl_brt & ~dr_brt
    only_dr = ~bl_brt & dr_brt
    overlay[both] = [0.2, 0.8, 0.2]       # green
    overlay[only_bl] = [1.0, 0.4, 0.7]    # pink
    overlay[only_dr] = [1.0, 0.6, 0.2]    # orange

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(overlay.transpose(1, 0, 2), origin='lower', extent=[*bounds_x, *bounds_y])
    ax.set_title(f"BRT Overlap ({dynamics_name.upper()}, {slice_label} = {slice_value:.2f})\n"
                 f"Volume Error: {metrics['brt_volume_error_pct']:.1f}%  |  "
                 f"Baseline: {metrics['baseline_brt_points']} pts, DeepReach: {metrics['deepreach_brt_points']} pts",
                 fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='Both agree (in BRT)'),
        Patch(facecolor=[1.0, 0.4, 0.7], label='Baseline only (missed by DeepReach)'),
        Patch(facecolor=[1.0, 0.6, 0.2], label='DeepReach only (false positive)'),
        Patch(facecolor=[0, 0, 0], label='Both agree (outside BRT)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'brt_overlap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/brt_overlap.png")


def main():
    parser = argparse.ArgumentParser(description="Compare optimized_dp baseline vs DeepReach")
    parser.add_argument("--baseline_grid", required=True, help="Path to baseline .npy")
    parser.add_argument("--deepreach_grid", default=None, help="Path to DeepReach .npy (Mode A)")
    parser.add_argument("--deepreach_model", default=None, help="Path to DeepReach .pth (Mode B)")
    parser.add_argument("--dynamics", default=None, choices=["air3d", "dubins3d"], help="Dynamics name (Mode B only)")
    parser.add_argument("--slice_dim", type=int, default=2, help="Dimension to slice for plots")
    parser.add_argument("--slice_idx", type=int, default=None, help="Slice index (default: middle)")
    parser.add_argument("--output_dir", default="baselines/plots", help="Output directory for plots")
    parser.add_argument("--device", default="cpu", help="Device for model evaluation (Mode B)")
    args = parser.parse_args()

    if args.deepreach_grid is None and args.deepreach_model is None:
        parser.error("Provide either --deepreach_grid (.npy) or --deepreach_model (.pth)")

    # Load baseline
    baseline = np.load(args.baseline_grid)
    print(f"Baseline shape: {baseline.shape}, range: [{baseline.min():.4f}, {baseline.max():.4f}]")

    # Load or compute DeepReach values
    if args.deepreach_grid is not None:
        deepreach = np.load(args.deepreach_grid)
        print(f"DeepReach shape: {deepreach.shape}, range: [{deepreach.min():.4f}, {deepreach.max():.4f}]")
    else:
        if args.dynamics is None:
            parser.error("--dynamics required when using --deepreach_model")
        from baselines.config import GRID_POINTS
        if args.dynamics == "air3d":
            from baselines.config import X1_BOUNDS, X2_BOUNDS, X3_BOUNDS
            grid_bounds = [X1_BOUNDS, X2_BOUNDS, X3_BOUNDS]
        else:
            from baselines.config import D_X_BOUNDS, D_Y_BOUNDS, D_THETA_BOUNDS
            grid_bounds = [D_X_BOUNDS, D_Y_BOUNDS, D_THETA_BOUNDS]
        print("Evaluating DeepReach model on grid...")
        deepreach = load_deepreach_from_model(args.deepreach_model, args.dynamics, grid_bounds, GRID_POINTS, args.device)
        print(f"DeepReach shape: {deepreach.shape}, range: [{deepreach.min():.4f}, {deepreach.max():.4f}]")

    assert baseline.shape == deepreach.shape, f"Shape mismatch: {baseline.shape} vs {deepreach.shape}"

    # Metrics
    metrics = compute_metrics(baseline, deepreach)
    print(f"\nMSE: {metrics['mse']:.6e}")
    print(f"BRT volume error: {metrics['brt_volume_error_pct']:.2f}%")
    print(f"Baseline BRT points: {metrics['baseline_brt_points']}")
    print(f"DeepReach BRT points: {metrics['deepreach_brt_points']}")

    # Plots
    slice_idx = args.slice_idx if args.slice_idx is not None else baseline.shape[args.slice_dim] // 2
    bounds_x = (-1.0, 1.0)
    bounds_y = (-1.0, 1.0)
    import math
    if args.slice_dim == 2:
        slice_value = np.linspace(-math.pi, math.pi, baseline.shape[2])[slice_idx]
    else:
        slice_value = np.linspace(-1.0, 1.0, baseline.shape[args.slice_dim])[slice_idx]
    dynamics_name = args.dynamics or "unknown"

    plot_slice_comparison(baseline, deepreach, bounds_x, bounds_y, args.slice_dim, slice_idx, slice_value, dynamics_name, args.output_dir)
    plot_brt_overlay(baseline, deepreach, bounds_x, bounds_y, args.slice_dim, slice_idx, slice_value, dynamics_name, metrics, args.output_dir)

    # Gradient / control metrics (Dubins3D and Air3D only)
    if dynamics_name in ('dubins3d', 'air3d'):
        if dynamics_name == 'air3d':
            grid_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]
        else:
            grid_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]

        ctrl_metrics = compute_gradient_control_metrics(baseline, deepreach, dynamics_name, grid_bounds)
        if ctrl_metrics is not None:
            print(f"Gradient MSE: {ctrl_metrics['gradient_mse']:.6e}")
            print(f"Control agreement: {ctrl_metrics['control_agreement_pct']:.2f}%")
            plot_control_comparison(ctrl_metrics, bounds_x, bounds_y, args.slice_dim, slice_idx,
                                    slice_value, dynamics_name, args.output_dir)


if __name__ == "__main__":
    main()
