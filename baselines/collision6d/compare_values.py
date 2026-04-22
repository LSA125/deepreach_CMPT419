'''
Compare optimized_dp ground-truth 6D value function against DeepReach output.
Separate from compare_values.py (3D only) for clarity.

Supports two modes:
  Mode A: --deepreach_grid  (compare .npy vs .npy, no GPU needed)
  Mode B: --deepreach_model (load .pth, evaluate on 6D grid, then compare)

6D state: [x1, y1, x2, y2, theta1, theta2]
Plots show 2D slices (x1 vs y1) with other 4 dims fixed.
'''

import sys
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 6D grid bounds matching TwoVehicleCollision6D
GRID_BOUNDS_6D = [
    (-1.0, 1.0),    # x1
    (-1.0, 1.0),    # y1
    (-1.0, 1.0),    # x2
    (-1.0, 1.0),    # y2
    (-math.pi, math.pi),  # theta1
    (-math.pi, math.pi),  # theta2
]

DIM_LABELS = [r'$x_1$', r'$y_1$', r'$x_2$', r'$y_2$', r'$\theta_1$', r'$\theta_2$']

# Default slice: fix x2=0.5, y2=0, theta1=-0.5, theta2=0
# (matches TwoVehicleCollision6D.plot_config state_slices)
DEFAULT_FIXED = {2: 0.5, 3: 0.0, 4: -0.5, 5: 0.0}


def load_deepreach_from_model(model_path, grid_points, device="cpu"):
    """Load a 6D .pth model and evaluate on the baseline grid."""
    import torch
    from utils.modules import SingleBVPNet
    from dynamics.dynamics import TwoVehicleCollision6D

    dyn = TwoVehicleCollision6D()

    # Load training config
    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')

    if os.path.exists(opt_path):
        import pickle
        with open(opt_path, 'rb') as f:
            orig_opt = pickle.load(f)
        print(f"Loaded training config from {opt_path}")
        dyn.deepreach_model = orig_opt.deepreach_model
        model = SingleBVPNet(in_features=7, out_features=1, type=orig_opt.model,
                             mode=orig_opt.model_mode, hidden_features=orig_opt.num_nl,
                             num_hidden_layers=orig_opt.num_hl)
    else:
        print("Warning: orig_opt.pickle not found, using defaults")
        dyn.deepreach_model = "exact"
        model = SingleBVPNet(in_features=7, out_features=1, type='sine',
                             mode='mlp', hidden_features=512, num_hidden_layers=3)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Build 6D grid
    axes = [np.linspace(b[0], b[1], grid_points) for b in GRID_BOUNDS_6D]
    grids = np.meshgrid(*axes, indexing='ij')
    coords_flat = np.stack([g.ravel() for g in grids], axis=-1)

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
            if (i // batch_size) % 20 == 0:
                print(f"  Evaluated {min(i+batch_size, len(inputs)):,}/{len(inputs):,} points")

    result_shape = tuple([grid_points] * 6)
    values = np.concatenate(values_list).reshape(result_shape)
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
        "total_points": int(baseline.size),
        "symmetric_diff_points": int(symmetric_diff),
    }


def extract_2d_slice(volume, plot_axes, fixed_values):
    """Extract a 2D slice from 6D volume.

    Args:
        volume: 6D numpy array
        plot_axes: (x_dim, y_dim) — dims to keep as the 2D plot axes
        fixed_values: dict {dim: value} for the 4 fixed dimensions
    Returns:
        2D numpy array
    """
    idx = [slice(None)] * 6
    for d in range(6):
        if d in plot_axes:
            continue
        val = fixed_values.get(d, 0.0)
        lo, hi = GRID_BOUNDS_6D[d]
        axis_pts = np.linspace(lo, hi, volume.shape[d])
        closest = int(np.argmin(np.abs(axis_pts - val)))
        idx[d] = closest
    return volume[tuple(idx)]


def format_fixed_desc(fixed_values):
    """Format fixed dimension values for plot titles."""
    parts = []
    for d, v in sorted(fixed_values.items()):
        parts.append(f"{DIM_LABELS[d]}={v:.2f}")
    return ", ".join(parts)


def plot_slice_comparison(baseline, deepreach, plot_axes, fixed_values, output_dir):
    """3-panel heatmap: baseline | deepreach | error for a 2D slice of 6D."""
    ax_x, ax_y = plot_axes
    bl_slice = extract_2d_slice(baseline, plot_axes, fixed_values)
    dr_slice = extract_2d_slice(deepreach, plot_axes, fixed_values)

    bounds_x = GRID_BOUNDS_6D[ax_x]
    bounds_y = GRID_BOUNDS_6D[ax_y]
    xlabel = DIM_LABELS[ax_x]
    ylabel = DIM_LABELS[ax_y]
    fixed_desc = format_fixed_desc(fixed_values)

    error_slice = np.abs(bl_slice - dr_slice)
    vmin = min(bl_slice.min(), dr_slice.min())
    vmax = max(bl_slice.max(), dr_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"6D TwoVehicleCollision — Value Function Comparison\n"
                 f"Slice: {fixed_desc}  |  Grid: {baseline.shape[0]}pts/dim",
                 fontsize=12)

    titles = [
        f"Ground Truth (optimized_dp, {baseline.shape[0]}pt)",
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
    fig.savefig(os.path.join(output_dir, 'slice_comparison_6d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/slice_comparison_6d.png")


def plot_brt_overlay(baseline, deepreach, plot_axes, fixed_values, metrics, output_dir):
    """BRT boundary overlay for a 2D slice of 6D."""
    ax_x, ax_y = plot_axes
    bl_slice = extract_2d_slice(baseline, plot_axes, fixed_values)
    dr_slice = extract_2d_slice(deepreach, plot_axes, fixed_values)

    bounds_x = GRID_BOUNDS_6D[ax_x]
    bounds_y = GRID_BOUNDS_6D[ax_y]
    xlabel = DIM_LABELS[ax_x]
    ylabel = DIM_LABELS[ax_y]
    fixed_desc = format_fixed_desc(fixed_values)

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
    ax.set_title(f"6D BRT Overlap — {fixed_desc}\n"
                 f"Volume Error: {metrics['brt_volume_error_pct']:.1f}%  |  "
                 f"Grid: {baseline.shape[0]}pt/dim  |  "
                 f"Baseline BRT: {metrics['baseline_brt_points']:,} pts",
                 fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    fig.savefig(os.path.join(output_dir, 'brt_overlap_6d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/brt_overlap_6d.png")


def main():
    parser = argparse.ArgumentParser(description="Compare 6D TwoVehicleCollision: optimized_dp vs DeepReach")
    parser.add_argument("--baseline_grid", required=True, help="Path to baseline 6D .npy")
    parser.add_argument("--deepreach_grid", default=None, help="Path to DeepReach 6D .npy (Mode A)")
    parser.add_argument("--deepreach_model", default=None, help="Path to DeepReach .pth (Mode B)")
    parser.add_argument("--plot_x", type=int, default=0, help="X-axis dimension for plots (default: 0 = x1)")
    parser.add_argument("--plot_y", type=int, default=1, help="Y-axis dimension for plots (default: 1 = y1)")
    parser.add_argument("--fix", nargs='*', default=None,
                        help="Fixed dim values as dim=val pairs, e.g. --fix 2=0.5 3=0.0 4=-0.5 5=0.0")
    parser.add_argument("--output_dir", default="baselines/collision6d/plots", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for model evaluation")
    args = parser.parse_args()

    if args.deepreach_grid is None and args.deepreach_model is None:
        parser.error("Provide either --deepreach_grid (.npy) or --deepreach_model (.pth)")

    # Parse fixed dimensions
    plot_axes = (args.plot_x, args.plot_y)
    if args.fix is not None:
        fixed_values = {}
        for pair in args.fix:
            d, v = pair.split('=')
            fixed_values[int(d)] = float(v)
    else:
        fixed_values = {d: v for d, v in DEFAULT_FIXED.items() if d not in plot_axes}

    # Load baseline
    baseline = np.load(args.baseline_grid)
    grid_points = baseline.shape[0]
    print(f"Baseline shape: {baseline.shape}, range: [{baseline.min():.4f}, {baseline.max():.4f}]")
    print(f"Grid: {grid_points}^6 = {baseline.size:,} points")

    # Load or compute DeepReach values
    if args.deepreach_grid is not None:
        deepreach = np.load(args.deepreach_grid)
        print(f"DeepReach shape: {deepreach.shape}, range: [{deepreach.min():.4f}, {deepreach.max():.4f}]")
    else:
        print(f"Evaluating DeepReach model on {grid_points}^6 grid...")
        deepreach = load_deepreach_from_model(args.deepreach_model, grid_points, args.device)
        print(f"DeepReach shape: {deepreach.shape}, range: [{deepreach.min():.4f}, {deepreach.max():.4f}]")

    assert baseline.shape == deepreach.shape, f"Shape mismatch: {baseline.shape} vs {deepreach.shape}"

    # Metrics
    metrics = compute_metrics(baseline, deepreach)
    print(f"\n{'='*50}")
    print(f"6D TwoVehicleCollision Comparison Results")
    print(f"{'='*50}")
    print(f"MSE:              {metrics['mse']:.6e}")
    print(f"BRT volume error: {metrics['brt_volume_error_pct']:.2f}%")
    print(f"Baseline BRT:     {metrics['baseline_brt_points']:,} / {metrics['total_points']:,} "
          f"({metrics['baseline_brt_points']/metrics['total_points']*100:.1f}%)")
    print(f"DeepReach BRT:    {metrics['deepreach_brt_points']:,} / {metrics['total_points']:,} "
          f"({metrics['deepreach_brt_points']/metrics['total_points']*100:.1f}%)")
    print(f"Baseline V range: [{baseline.min():.4f}, {baseline.max():.4f}]")
    print(f"DeepReach V range:[{deepreach.min():.4f}, {deepreach.max():.4f}]")
    print(f"{'='*50}")

    # Plots
    print(f"\nPlotting slice: {DIM_LABELS[plot_axes[0]]} vs {DIM_LABELS[plot_axes[1]]}")
    print(f"Fixed: {format_fixed_desc(fixed_values)}")

    plot_slice_comparison(baseline, deepreach, plot_axes, fixed_values, args.output_dir)
    plot_brt_overlay(baseline, deepreach, plot_axes, fixed_values, metrics, args.output_dir)


if __name__ == "__main__":
    main()
