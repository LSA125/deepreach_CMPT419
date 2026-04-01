import sys
import os
import time
import json
import argparse
import tracemalloc
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.expanduser('~/optimized_dp'))

from odp.Grid import Grid
from odp.solver import HJSolver

from baselines.optimized_dp.dynamics.collision6d import TwoVehicleCollision6D

# Params matching DeepReach TwoVehicleCollision6D
VELOCITY = 0.6
OMEGA_MAX = 1.1
COLLISION_R = 0.25
T_MAX = 1.0
T_STEP = 0.05

# 6D bounds: x1, y1, x2, y2 in [-1, 1], theta1, theta2 in [-pi, pi]
BOUNDS_LO = [-1.0, -1.0, -1.0, -1.0, -math.pi, -math.pi]
BOUNDS_HI = [ 1.0,  1.0,  1.0,  1.0,  math.pi,  math.pi]
PERIODIC_DIMS = [4, 5]  # theta1, theta2 are periodic


def compute_target_set(g):
    """Collision region: ||[x1,y1] - [x2,y2]|| - collisionR <= 0"""
    # Build meshgrid from the grid object
    # g.grid_points gives the 1D arrays for each dimension
    coords = np.meshgrid(
        g.grid_points[0], g.grid_points[1], g.grid_points[2],
        g.grid_points[3], g.grid_points[4], g.grid_points[5],
        indexing='ij'
    )
    dx = coords[0] - coords[2]  # x1 - x2
    dy = coords[1] - coords[3]  # y1 - y2
    target = np.sqrt(dx**2 + dy**2) - COLLISION_R
    return target


def main():
    parser = argparse.ArgumentParser(description="Solve 6D TwoVehicleCollision BRT")
    parser.add_argument('--grid_points', type=int, default=15,
                        help='Points per dimension (default: 15). '
                             'Memory: N^6 floats. 15→~91MB, 21→~1.4GB, 31→~28GB')
    args = parser.parse_args()

    N = args.grid_points
    total_points = N**6

    print(f"=" * 60)
    print(f"6D TwoVehicleCollision BRT Solve")
    print(f"=" * 60)
    print(f"Grid: {N}^6 = {total_points:,} points")
    print(f"Estimated array size: {total_points * 8 / 1e9:.2f} GB (float64)")
    print(f"Params: velocity={VELOCITY}, omega_max={OMEGA_MAX}, collisionR={COLLISION_R}")
    print(f"T_MAX={T_MAX}, T_STEP={T_STEP}")
    print()

    if total_points > 5e8:
        print(f"WARNING: {total_points:,} points will likely exceed available memory.")
        print("Consider using --grid_points 15 or 11.")
        return

    # Build 6D grid with periodic theta dimensions
    g = Grid(
        np.array(BOUNDS_LO),
        np.array(BOUNDS_HI),
        6,
        np.array([N, N, N, N, N, N]),
        PERIODIC_DIMS
    )

    print("Computing target set (collision region)...")
    target = compute_target_set(g)
    print(f"Target set shape: {target.shape}")
    collision_frac = np.sum(target <= 0) / total_points * 100
    print(f"Initial collision region: {collision_frac:.1f}% of grid")

    tau = np.arange(start=0, stop=T_MAX + 1e-5, step=T_STEP)

    # Both vehicles maximize (avoid collision)
    system = TwoVehicleCollision6D(uMode="max", dMode="min")

    # BRT: matches DeepReach minWith=target
    compMethods = {"TargetSetMode": "minVWithV0"}

    print(f"\nSolving... (this may take a while at 6D)")
    tracemalloc.start()
    start_time = time.time()

    result = HJSolver(system, g, target, tau, compMethods, saveAllTimeSteps=False)

    solve_time = time.time() - start_time
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    print(f"\nDone!")
    print(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    print(f"Peak memory: {peak_memory:.1f} MB ({peak_memory/1024:.2f} GB)")
    print(f"Result shape: {result.shape}")

    brt_frac = np.sum(result <= 0) / total_points * 100
    print(f"BRT region: {brt_frac:.1f}% of grid")

    # Save outputs
    grids_dir = os.path.join(os.path.dirname(__file__), '..', 'grids')
    os.makedirs(grids_dir, exist_ok=True)

    out_name = f'collision6d_grid_{N}pt'
    np.save(os.path.join(grids_dir, f'{out_name}.npy'), result)

    metadata = {
        "system": "TwoVehicleCollision6D",
        "grid_points": N,
        "total_points": total_points,
        "bounds": {
            "x1": [-1.0, 1.0], "y1": [-1.0, 1.0],
            "x2": [-1.0, 1.0], "y2": [-1.0, 1.0],
            "theta1": [-math.pi, math.pi], "theta2": [-math.pi, math.pi],
        },
        "periodic_dims": PERIODIC_DIMS,
        "params": {
            "velocity": VELOCITY, "omega_max": OMEGA_MAX,
            "collisionR": COLLISION_R,
        },
        "t_max": T_MAX,
        "t_step": T_STEP,
        "solve_time_s": round(solve_time, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "result_shape": list(result.shape),
        "brt_fraction_pct": round(brt_frac, 2),
    }
    with open(os.path.join(grids_dir, f'{out_name}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: grids/{out_name}.npy, grids/{out_name}_metadata.json")

    # Print scalability comparison
    print(f"\n{'=' * 60}")
    print("Scalability comparison (same system):")
    print(f"  3D @ 101 pts/dim: {101**3:>12,} points")
    print(f"  6D @ {N:>3} pts/dim: {N**6:>12,} points  ← you are here")
    print(f"  6D @ 101 pts/dim: {101**6:>12,} points  (impossible)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
