"""
Standalone script to generate the 10D BRAT demo GIF only.
Runs just nominal + BRAT safe simulations (no Phase 3/4 diagnostics).
Outputs to baselines/narrow_passage_10d/plots/demo_nominal_vs_brat.gif

Usage (deepreach env, from repo root):
  python baselines/narrow_passage_10d/make_gif.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import everything from the demo module
from baselines.narrow_passage_10d.brat_demo import (
    load_model, simulate, make_gif, eval_V, compute_avoid_fn,
    MODEL_PATH, INIT_STATE, OUT_DIR, T_MAX_MODEL, NOMINAL_SPEED,
    GOAL_1, GOAL_2, STRANDED_POS,
)
import math, numpy as np

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading 10D NarrowPassage BRAT model …")
    model, dyn, orig_opt = load_model(MODEL_PATH, device)

    print("Simulating NOMINAL (straight — no steering) …")
    r_nominal = simulate(model, dyn, INIT_STATE, use_safety=False, device=device)
    if r_nominal["collision_idx"] is not None:
        print(f"  CRASH at t={r_nominal['collision_idx']*0.02:.2f}s ({r_nominal['collision_who']})")
    else:
        print("  No crash")

    print("Simulating DEEPREACH BRAT safety filter …")
    r_safe = simulate(model, dyn, INIT_STATE, use_safety=True, use_nn=False, device=device)
    if r_safe["collision_idx"] is not None:
        print(f"  WARNING: crash at t={r_safe['collision_idx']*0.02:.2f}s")
    else:
        print(f"  No crash  (min d12={r_safe['dist_cars'].min():.3f}m)")

    print("\n--- Animated GIF ---")
    make_gif(r_nominal, r_safe)
    print("\nDone.")

if __name__ == "__main__":
    main()
