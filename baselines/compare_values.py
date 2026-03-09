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

    if dynamics_name == "air3d":
        from baselines.config import VELOCITY, OMEGA_MAX, BETA
        dyn = Air3D(collisionR=BETA, velocity=VELOCITY, omega_max=OMEGA_MAX, angle_alpha_factor=1.0)
    elif dynamics_name == "dubins3d":
        from baselines.config import DUBINS_VELOCITY, DUBINS_OMEGA_MAX, GOAL_R
        dyn = Dubins3D(goalR=GOAL_R, velocity=DUBINS_VELOCITY, omega_max=DUBINS_OMEGA_MAX,
                       angle_alpha_factor=1.0, set_mode="reach", freeze_model=False)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_name}")

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


def plot_slice_comparison(baseline, deepreach, bounds_x, bounds_y, slice_dim, slice_idx, output_dir):
    """3-panel heatmap: baseline | deepreach | absolute error at a fixed slice."""
    if slice_dim == 2:
        bl_slice = baseline[:, :, slice_idx]
        dr_slice = deepreach[:, :, slice_idx]
        xlabel, ylabel = "x1", "x2"
    elif slice_dim == 1:
        bl_slice = baseline[:, slice_idx, :]
        dr_slice = deepreach[:, slice_idx, :]
        xlabel, ylabel = "x1", "x3"
    else:
        bl_slice = baseline[slice_idx, :, :]
        dr_slice = deepreach[slice_idx, :, :]
        xlabel, ylabel = "x2", "x3"

    error_slice = np.abs(bl_slice - dr_slice)
    vmin = min(bl_slice.min(), dr_slice.min())
    vmax = max(bl_slice.max(), dr_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, data, title in zip(axes, [bl_slice, dr_slice, error_slice],
                                ["Baseline (optimized_dp)", "DeepReach", "Absolute Error"]):
        if title == "Absolute Error":
            im = ax.imshow(data.T, origin='lower', extent=[*bounds_x, *bounds_y], cmap='hot')
        else:
            im = ax.imshow(data.T, origin='lower', extent=[*bounds_x, *bounds_y], cmap='coolwarm',
                          vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'slice_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/slice_comparison.png")


def plot_brt_overlay(baseline, deepreach, bounds_x, bounds_y, slice_dim, slice_idx, output_dir):
    """BRT boundary overlay: green = agreement, pink = disagreement."""
    if slice_dim == 2:
        bl_slice = baseline[:, :, slice_idx]
        dr_slice = deepreach[:, :, slice_idx]
        xlabel, ylabel = "x1", "x2"
    elif slice_dim == 1:
        bl_slice = baseline[:, slice_idx, :]
        dr_slice = deepreach[:, slice_idx, :]
        xlabel, ylabel = "x1", "x3"
    else:
        bl_slice = baseline[slice_idx, :, :]
        dr_slice = deepreach[slice_idx, :, :]
        xlabel, ylabel = "x2", "x3"

    bl_brt = (bl_slice <= 0).astype(float)
    dr_brt = (dr_slice <= 0).astype(float)

    # RGB: green = both agree, pink = disagreement
    overlay = np.zeros((*bl_brt.shape, 3))
    both = (bl_brt.astype(bool)) & (dr_brt.astype(bool))
    only_bl = (bl_brt.astype(bool)) & (~dr_brt.astype(bool))
    only_dr = (~bl_brt.astype(bool)) & (dr_brt.astype(bool))
    overlay[both] = [0.2, 0.8, 0.2]       # green
    overlay[only_bl] = [1.0, 0.4, 0.7]    # pink
    overlay[only_dr] = [1.0, 0.6, 0.2]    # orange

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay.transpose(1, 0, 2), origin='lower', extent=[*bounds_x, *bounds_y])
    ax.set_title("BRT Overlay (green=agree, pink=baseline only, orange=deepreach only)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'brt_overlap.png'), dpi=150)
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

    plot_slice_comparison(baseline, deepreach, bounds_x, bounds_y, args.slice_dim, slice_idx, args.output_dir)
    plot_brt_overlay(baseline, deepreach, bounds_x, bounds_y, args.slice_dim, slice_idx, args.output_dir)


if __name__ == "__main__":
    main()
