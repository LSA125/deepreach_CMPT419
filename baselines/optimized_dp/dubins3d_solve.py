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

from baselines.config import GOAL_R, DUBINS_VELOCITY, DUBINS_OMEGA_MAX, T_MAX, T_STEP
from baselines.config import D_X_BOUNDS, D_Y_BOUNDS, D_THETA_BOUNDS, GRID_POINTS
from baselines.optimized_dp.dynamics.dubins3d import Dubins3D

# Grid: 3D, periodic in dim 2 (theta)
g = Grid(
    np.array([D_X_BOUNDS[0], D_Y_BOUNDS[0], D_THETA_BOUNDS[0]]),
    np.array([D_X_BOUNDS[1], D_Y_BOUNDS[1], D_THETA_BOUNDS[1]]),
    3,
    np.array([GRID_POINTS, GRID_POINTS, GRID_POINTS]),
    [2]
)

# Target set: goal cylinder ||x, y|| <= GOAL_R
target = CylinderShape(g, [2], np.zeros(3), GOAL_R)

# Time steps
tau = np.arange(start=0, stop=T_MAX + 1e-5, step=T_STEP)

# Dubins3D: minimize to reach goal
system = Dubins3D(uMode="min")

# BRT: once reachable, always reachable
compMethods = {"TargetSetMode": "minVWithV0"}

print(f"Solving Dubins3D BRT: {GRID_POINTS}^3 grid, T={T_MAX}, dt={T_STEP}")
print(f"Params: speed={DUBINS_VELOCITY}, wMax={DUBINS_OMEGA_MAX}, goalR={GOAL_R}")

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

np.save(os.path.join(grids_dir, 'dubins3d_grid.npy'), result)

metadata = {
    "system": "Dubins3D",
    "grid_points": GRID_POINTS,
    "bounds": {"x": list(D_X_BOUNDS), "y": list(D_Y_BOUNDS), "theta": list(D_THETA_BOUNDS)},
    "params": {"speed": DUBINS_VELOCITY, "omega_max": DUBINS_OMEGA_MAX, "goal_r": GOAL_R},
    "t_max": T_MAX,
    "t_step": T_STEP,
    "solve_time_s": round(solve_time, 2),
    "peak_memory_mb": round(peak_memory, 2),
    "result_shape": list(result.shape),
}
with open(os.path.join(grids_dir, 'dubins3d_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: grids/dubins3d_grid.npy, grids/dubins3d_metadata.json")
