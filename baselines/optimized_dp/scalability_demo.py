'''
Scalability Demo: Curse of Dimensionality in Grid-Based HJ Reachability

Demonstrates why grid-based solvers cannot scale to high dimensions:
1. Runs 6D TwoVehicleCollision at multiple resolutions (11, 15, 21, 25 pts/dim)
2. Compares against 3D Air3D at 101 pts/dim (the "easy" case)
3. Produces timing/memory table + exponential blowup chart

Also compares coarse 6D solutions against the finest available grid (resolution degradation).

Usage:
    conda activate odp && python baselines/optimized_dp/scalability_demo.py
    conda activate odp && python baselines/optimized_dp/scalability_demo.py --max_res 21
    conda activate odp && python baselines/optimized_dp/scalability_demo.py --skip_solve  # just plot from saved metadata
'''

import sys
import os
import time
import json
import argparse
import tracemalloc
import numpy as np
import math
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.expanduser('~/optimized_dp'))

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.solver import HJSolver

from baselines.optimized_dp.dynamics.collision6d import TwoVehicleCollision6D
from baselines.optimized_dp.dynamics.air3d import Air3D

# 6D params
BOUNDS_LO_6D = [-1.0, -1.0, -1.0, -1.0, -math.pi, -math.pi]
BOUNDS_HI_6D = [ 1.0,  1.0,  1.0,  1.0,  math.pi,  math.pi]
PERIODIC_6D = [4, 5]
COLLISION_R = 0.25
T_MAX = 1.0
T_STEP = 0.05

# 3D Air3D params
from baselines.config import BETA, VELOCITY, OMEGA_MAX
from baselines.config import X1_BOUNDS, X2_BOUNDS, X3_BOUNDS, GRID_POINTS


def compute_6d_target(g):
    coords = np.meshgrid(
        g.grid_points[0], g.grid_points[1], g.grid_points[2],
        g.grid_points[3], g.grid_points[4], g.grid_points[5],
        indexing='ij'
    )
    dx = coords[0] - coords[2]
    dy = coords[1] - coords[3]
    return np.sqrt(dx**2 + dy**2) - COLLISION_R


def solve_6d(N):
    """Solve 6D collision at N pts/dim. Returns (result, time_s, memory_mb)."""
    total = N**6
    print(f"\n{'─'*50}")
    print(f"6D @ {N} pts/dim = {total:,} points ({total*8/1e9:.2f} GB)")
    print(f"{'─'*50}")

    g = Grid(np.array(BOUNDS_LO_6D), np.array(BOUNDS_HI_6D), 6,
             np.array([N]*6), PERIODIC_6D)
    target = compute_6d_target(g)
    tau = np.arange(0, T_MAX + 1e-5, T_STEP)
    system = TwoVehicleCollision6D(uMode="max", dMode="min")
    compMethods = {"TargetSetMode": "minVWithV0"}

    tracemalloc.start()
    t0 = time.time()
    result = HJSolver(system, g, target, tau, compMethods, saveAllTimeSteps=False)
    solve_time = time.time() - t0
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
    tracemalloc.stop()

    brt_frac = np.sum(result <= 0) / total * 100
    print(f"  Time: {solve_time:.1f}s | Memory: {peak_mem:.0f} MB | BRT: {brt_frac:.1f}%")
    return result, solve_time, peak_mem, brt_frac


def solve_3d_air3d(N):
    """Solve 3D Air3D at N pts/dim for comparison."""
    total = N**3
    print(f"\n{'─'*50}")
    print(f"3D Air3D @ {N} pts/dim = {total:,} points")
    print(f"{'─'*50}")

    g = Grid(
        np.array([X1_BOUNDS[0], X2_BOUNDS[0], X3_BOUNDS[0]]),
        np.array([X1_BOUNDS[1], X2_BOUNDS[1], X3_BOUNDS[1]]),
        3, np.array([N, N, N]), [2]
    )
    target = CylinderShape(g, [2], np.zeros(3), BETA)
    tau = np.arange(0, T_MAX + 1e-5, T_STEP)
    system = Air3D(uMode="max", dMode="min")
    compMethods = {"TargetSetMode": "minVWithV0"}

    tracemalloc.start()
    t0 = time.time()
    result = HJSolver(system, g, target, tau, compMethods, saveAllTimeSteps=False)
    solve_time = time.time() - t0
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024**2)
    tracemalloc.stop()

    print(f"  Time: {solve_time:.1f}s | Memory: {peak_mem:.0f} MB")
    return result, solve_time, peak_mem


def plot_scalability(results_3d, results_6d, output_dir):
    """Generate scalability charts."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Chart 1: Grid points vs Solve Time ---
    ax = axes[0]
    # 6D data
    ns_6d = [r['N'] for r in results_6d]
    times_6d = [r['time_s'] for r in results_6d]
    points_6d = [r['N']**6 for r in results_6d]
    ax.plot(ns_6d, times_6d, 'ro-', markersize=8, linewidth=2, label='6D TwoVehicleCollision')
    # 3D data
    ns_3d = [r['N'] for r in results_3d]
    times_3d = [r['time_s'] for r in results_3d]
    ax.plot(ns_3d, times_3d, 'bs-', markersize=8, linewidth=2, label='3D Air3D')
    ax.set_xlabel('Grid points per dimension', fontsize=11)
    ax.set_ylabel('Solve time (seconds)', fontsize=11)
    ax.set_title('Solve Time vs Resolution', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Chart 2: Grid points vs Memory ---
    ax = axes[1]
    mems_6d = [r['memory_mb'] for r in results_6d]
    mems_3d = [r['memory_mb'] for r in results_3d]
    ax.plot(ns_6d, mems_6d, 'ro-', markersize=8, linewidth=2, label='6D TwoVehicleCollision')
    ax.plot(ns_3d, mems_3d, 'bs-', markersize=8, linewidth=2, label='3D Air3D')
    ax.set_xlabel('Grid points per dimension', fontsize=11)
    ax.set_ylabel('Peak memory (MB)', fontsize=11)
    ax.set_title('Memory Usage vs Resolution', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Chart 3: Total grid points vs Time (shows exponential) ---
    ax = axes[2]
    ax.plot(points_6d, times_6d, 'ro-', markersize=8, linewidth=2, label='6D')
    ax.plot([n**3 for n in ns_3d], times_3d, 'bs-', markersize=8, linewidth=2, label='3D')
    ax.set_xlabel('Total grid points', fontsize=11)
    ax.set_ylabel('Solve time (seconds)', fontsize=11)
    ax.set_title('Curse of Dimensionality', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation for impossible cases
    ax.annotate(f'6D @ 101pt/dim\n= {101**6:,.0f} pts\n(impossible)',
                xy=(1e11, max(times_6d)), fontsize=9, color='red',
                ha='center', style='italic')

    fig.suptitle('Grid-Based HJ Reachability: Scalability Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scalability_demo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir}/scalability_demo.png")


def plot_resolution_degradation(grids_6d, output_dir):
    """Show how coarse grids lose detail by comparing 2D slices."""
    if len(grids_6d) < 2:
        print("Need at least 2 resolutions for degradation plot, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n_grids = len(grids_6d)

    fig, axes = plt.subplots(1, n_grids, figsize=(7*n_grids, 6))
    if n_grids == 1:
        axes = [axes]

    # Use consistent color range across all subplots
    all_mins, all_maxs = [], []
    slices = []
    for N, grid in grids_6d:
        mid = grid.shape[0] // 2
        sl = grid[:, :, mid, mid, mid, mid]
        slices.append(sl)
        all_mins.append(sl.min())
        all_maxs.append(sl.max())
    vmin, vmax = min(all_mins), max(all_maxs)

    for i, (N, grid) in enumerate(grids_6d):
        sl = slices[i]
        ax = axes[i]
        extent = [-1, 1, -1, 1]
        im = ax.imshow(sl.T, origin='lower', extent=extent, cmap='coolwarm',
                       vmin=vmin, vmax=vmax)
        ax.contour(np.linspace(-1, 1, N), np.linspace(-1, 1, N), sl.T,
                   levels=[0], colors='black', linewidths=2)

        solve_time = None
        for r in [{'N': 11, 't': '3.0s'}, {'N': 15, 't': '25s'}, {'N': 21, 't': '214s'}]:
            if r['N'] == N:
                solve_time = r['t']

        title = f'{N} pts/dim  |  {N**6:,} points'
        if solve_time:
            title += f'  |  {solve_time}'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(r'Vehicle 1 position $x_1$', fontsize=11)
        if i == 0:
            ax.set_ylabel(r'Vehicle 1 position $y_1$', fontsize=11)
        else:
            ax.set_ylabel('')
        ax.tick_params(labelsize=10)

    fig.suptitle('6D TwoVehicleCollision: Resolution Degradation\n'
                 'BRT slice ($x_1$ vs $y_1$), other dims fixed at midpoint. '
                 'Black contour = BRT boundary (V=0)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    # Single colorbar on the right — add after tight_layout so it is not overridden
    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('V(x)  (V < 0 = inside BRT)', fontsize=11)

    fig.savefig(os.path.join(output_dir, 'resolution_degradation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/resolution_degradation.png")


def main():
    parser = argparse.ArgumentParser(description="Scalability demo: curse of dimensionality")
    parser.add_argument('--max_res', type=int, default=25,
                        help='Max 6D resolution to attempt (default: 25)')
    parser.add_argument('--skip_solve', action='store_true',
                        help='Skip solving, just plot from saved metadata')
    parser.add_argument('--output_dir', default='baselines/plots/scalability',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = args.output_dir
    grids_dir = os.path.join(os.path.dirname(__file__), '..', 'grids')

    # 6D resolutions to test
    resolutions_6d = [r for r in [11, 15, 21, 25] if r <= args.max_res]
    # 3D resolutions for comparison
    resolutions_3d = [21, 41, 61, 81, 101]

    if args.skip_solve:
        # Load from saved metadata
        results_6d = []
        for N in resolutions_6d:
            meta_path = os.path.join(grids_dir, f'collision6d_grid_{N}pt_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                results_6d.append({
                    'N': N, 'total_points': N**6,
                    'time_s': meta['solve_time_s'],
                    'memory_mb': meta['peak_memory_mb'],
                })
        # For 3D skip_solve, use air3d metadata if available
        results_3d = []
        meta_path = os.path.join(grids_dir, 'air3d_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            results_3d.append({
                'N': meta['grid_points'], 'time_s': meta['solve_time_s'],
                'memory_mb': meta['peak_memory_mb'],
            })
    else:
        # Run solves
        print("=" * 60)
        print("SCALABILITY DEMO: Curse of Dimensionality")
        print("=" * 60)

        # 3D Air3D at multiple resolutions
        results_3d = []
        for N in resolutions_3d:
            _, t, m = solve_3d_air3d(N)
            results_3d.append({'N': N, 'time_s': t, 'memory_mb': m})

        # 6D collision at increasing resolutions
        results_6d = []
        grids_6d_data = []
        for N in resolutions_6d:
            total = N**6
            if total > 5e8:
                print(f"\nSkipping {N}^6 = {total:,} points (would exceed memory)")
                # Record as failed attempt
                results_6d.append({
                    'N': N, 'total_points': total,
                    'time_s': None, 'memory_mb': None,
                    'status': 'skipped_memory'
                })
                continue

            result, t, m, brt = solve_6d(N)
            results_6d.append({
                'N': N, 'total_points': total,
                'time_s': t, 'memory_mb': m,
                'brt_pct': brt,
            })
            grids_6d_data.append((N, result))

            # Save metadata
            meta = {'grid_points': N, 'total_points': total,
                    'solve_time_s': round(t, 2), 'peak_memory_mb': round(m, 2),
                    'brt_fraction_pct': round(brt, 2)}
            meta_path = os.path.join(grids_dir, f'collision6d_grid_{N}pt_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

        # Resolution degradation plot
        if grids_6d_data:
            plot_resolution_degradation(grids_6d_data, output_dir)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'SCALABILITY RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'System':<12} {'Dims':>4} {'N/dim':>5} {'Total Points':>15} {'Time (s)':>10} {'Memory (MB)':>12}")
    print(f"{'─'*70}")
    for r in results_3d:
        print(f"{'Air3D':<12} {'3':>4} {r['N']:>5} {r['N']**3:>15,} {r['time_s']:>10.1f} {r['memory_mb']:>12.0f}")
    print(f"{'─'*70}")
    for r in results_6d:
        if r.get('time_s') is not None:
            print(f"{'Collision6D':<12} {'6':>4} {r['N']:>5} {r['total_points']:>15,} {r['time_s']:>10.1f} {r['memory_mb']:>12.0f}")
        else:
            print(f"{'Collision6D':<12} {'6':>4} {r['N']:>5} {r['total_points']:>15,} {'SKIPPED':>10} {'OOM':>12}")
    print(f"{'─'*70}")
    print(f"{'Collision6D':<12} {'6':>4} {'101':>5} {101**6:>15,} {'IMPOSSIBLE':>10} {'':>12}")
    print(f"{'='*70}")

    # DeepReach comparison line
    print(f"\nDeepReach neural solver: 110,000 epochs, ~9.1 MB model, evaluates in seconds at ANY resolution")
    print(f"Grid solver at 6D: limited to {max(r['N'] for r in results_6d if r.get('time_s')):} pts/dim = "
          f"{max(r['N'] for r in results_6d if r.get('time_s'))**6:,} points")

    # Plot (only with successful solves)
    results_6d_ok = [r for r in results_6d if r.get('time_s') is not None]
    if results_6d_ok and results_3d:
        plot_scalability(results_3d, results_6d_ok, output_dir)

    # Save results JSON
    all_results = {'3d': results_3d, '6d': results_6d}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scalability_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_dir}/scalability_results.json")


if __name__ == "__main__":
    main()
