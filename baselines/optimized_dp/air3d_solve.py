import sys
import os
import time
import json
import tracemalloc
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.expanduser('~/optimized_dp'))

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.solver import HJSolver

from baselines.config import BETA, VELOCITY, OMEGA_MAX, T_MAX, T_STEP
from baselines.config import X1_BOUNDS, X2_BOUNDS, X3_BOUNDS, GRID_POINTS
from baselines.optimized_dp.dynamics.air3d import Air3D

# Grid: 3D, periodic in dim 2 (relative heading)
g = Grid(
    np.array([X1_BOUNDS[0], X2_BOUNDS[0], X3_BOUNDS[0]]),
    np.array([X1_BOUNDS[1], X2_BOUNDS[1], X3_BOUNDS[1]]),
    3,
    np.array([GRID_POINTS, GRID_POINTS, GRID_POINTS]),
    [2]
)

# Target set: collision cylinder ||x1, x2|| <= BETA
target = CylinderShape(g, [2], np.zeros(3), BETA)

# Time steps
tau = np.arange(start=0, stop=T_MAX + 1e-5, step=T_STEP)

# Air3D: evader maximizes (uMode=max), pursuer minimizes (dMode=min)
system = Air3D(uMode="max", dMode="min")

# BRT: matches DeepReach minWith=target
compMethods = {"TargetSetMode": "minVWithV0"}

print(f"Solving Air3D BRS: {GRID_POINTS}^3 grid, T={T_MAX}, dt={T_STEP}")
print(f"Params: speed={VELOCITY}, wMax={OMEGA_MAX}, beta={BETA}")

tracemalloc.start()
start_time = time.time()

result = HJSolver(system, g, target, tau, compMethods, saveAllTimeSteps=False)

solve_time = time.time() - start_time
peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
tracemalloc.stop()

print(f"Solve time: {solve_time:.1f}s")
print(f"Peak memory: {peak_memory:.1f} MB")
print(f"Result shape: {result.shape}")

# Save outputs
grids_dir = os.path.join(os.path.dirname(__file__), '..', 'grids')
os.makedirs(grids_dir, exist_ok=True)

np.save(os.path.join(grids_dir, 'air3d_grid.npy'), result)

metadata = {
    "system": "Air3D",
    "grid_points": GRID_POINTS,
    "bounds": {"x1": list(X1_BOUNDS), "x2": list(X2_BOUNDS), "x3": list(X3_BOUNDS)},
    "params": {"speed": VELOCITY, "omega_max": OMEGA_MAX, "beta": BETA},
    "t_max": T_MAX,
    "t_step": T_STEP,
    "solve_time_s": round(solve_time, 2),
    "peak_memory_mb": round(peak_memory, 2),
    "result_shape": list(result.shape),
}
with open(os.path.join(grids_dir, 'air3d_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: grids/air3d_grid.npy, grids/air3d_metadata.json")
