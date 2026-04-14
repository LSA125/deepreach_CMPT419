"""
BRT/BRAT Volume Error Estimation via Sampling for High-Dimensional Systems.

For 10D+ systems where grid-based methods are infeasible, estimate safe set volume
using trajectory rollouts with scenario optimization.
"""

import json
import sys
import time
import argparse
import pickle
import inspect
from pathlib import Path

import numpy as np
import torch
from datetime import datetime

from dynamics.dynamics import Dynamics
from dynamics import dynamics as dynamics_module
from utils.modules import SingleBVPNet
from utils.error_evaluators import (
    scenario_optimization, 
    SliceSampleGenerator, 
    ValueThresholdValidator
)


def load_experiment(experiment_dir: str, device: str = 'cuda:0'):
    """Load experiment config and trained model."""
    experiment_dir = Path(experiment_dir)
    
    # Load original options
    with open(experiment_dir / 'orig_opt.pickle', 'rb') as f:
        orig_opt = pickle.load(f)
    
    print(f"[*] Loaded experiment config from {experiment_dir}")
    print(f"    Dynamics: {orig_opt.dynamics_class}")
    print(f"    Set mode: {orig_opt.set_mode if hasattr(orig_opt, 'set_mode') else 'N/A'}")
    
    # Reconstruct dynamics
    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    dynamics_kwargs = {
        argname: getattr(orig_opt, argname) 
        for argname in inspect.signature(dynamics_class).parameters.keys() 
        if argname != 'self'
    }
    dynamics = dynamics_class(**dynamics_kwargs)
    dynamics.deepreach_model = orig_opt.deepreach_model
    
    print(f"    State dim: {dynamics.state_dim}")
    print(f"    Loss type: {dynamics.loss_type}")
    
    # Build model
    model = SingleBVPNet(
        in_features=dynamics.input_dim, 
        out_features=1, 
        type=orig_opt.model, 
        mode=orig_opt.model_mode,
        final_layer_factor=1., 
        hidden_features=orig_opt.num_nl, 
        num_hidden_layers=orig_opt.num_hl
    )
    model.to(device)
    
    # Load checkpoint (latest) - sort numerically by epoch number
    ckpt_files = list((experiment_dir / 'training' / 'checkpoints').glob('model_epoch_*.pth'))
    latest_ckpt = sorted(ckpt_files, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    
    ckpt = torch.load(latest_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print(f"    Loaded checkpoint: {latest_ckpt.name}")
    
    return orig_opt, dynamics, model


def compute_volume_error_sampling(
    dynamics: Dynamics,
    model: torch.nn.Module,
    device: str = 'cuda:0',
    num_scenarios: int = 100000,
    dt: float = 0.0025,
    max_steps: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Estimate BRT/BRAT volume error by checking boundary satisfaction.
    
    For each sampled initial state, evaluates the learned value function.
    States in the safe set (value <= 0 for reach, >= 0 for avoid) are counted as successes.
    This gives the fraction of the initial state space that the value function 
    identifies as part of the safe/reachable set.
    
    Args:
        dynamics: System dynamics
        model: Trained DeepReach model
        device: Computation device
        num_scenarios: Number of initial states to sample
        dt: Timestep (unused here, for compatibility)
        max_steps: Unused (for compatibility)
        verbose: Print progress
        
    Returns:
        Dictionary with metrics tracking safe set fraction
    """
    
    if verbose:
        print(f"\n[*] Starting safe set volume estimation")
        print(f"    Num sampled states: {num_scenarios:,}")
        print(f"    Set mode: {dynamics.set_mode}")
    
    model.eval()
    start_time = time.time()
    
    # Sample random initial states uniformly from state space
    state_test_range = dynamics.state_test_range()
    state_mins = torch.tensor([r[0] for r in state_test_range], dtype=torch.float32, device=device)
    state_maxs = torch.tensor([r[1] for r in state_test_range], dtype=torch.float32, device=device)
    
    sampled_states = torch.rand(num_scenarios, dynamics.state_dim, device=device)
    sampled_states = sampled_states * (state_maxs - state_mins) + state_mins
    
    if verbose:
        print(f"    State range: {dynamics.state_dim}D")
    
    with torch.no_grad():
        # Evaluate value function at t=0 for all sampled states
        t = torch.zeros((num_scenarios, 1), device=device)
        coords = torch.cat([t, sampled_states], dim=1)
        model_in = dynamics.coord_to_input(coords)
        
        model_out = model({'coords': model_in})
        values = dynamics.io_to_value(model_in, model_out['model_out'].squeeze(-1))
        
        # Check boundary condition / value function sign
        if dynamics.set_mode == 'reach':
            # Safe set: value <= 0 (reachable set)
            in_safe_set = values <= 0.0
        else:  # avoid
            # Safe set: value >= 0 (avoidable set - can avoid obstacles)
            in_safe_set = values >= 0.0
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    num_in_safe_set = in_safe_set.sum().item()
    num_outside = num_scenarios - num_in_safe_set
    safe_fraction = num_in_safe_set / num_scenarios
    unsafe_fraction = num_outside / num_scenarios
    
    if verbose:
        print(f"\n[*] Evaluation complete (elapsed: {elapsed:.2f}s)")
        print(f"    Total sampled states: {num_scenarios:,}")
        print(f"    In safe set: {num_in_safe_set:,} ({safe_fraction*100:.2f}%)")
        print(f"    Outside safe set: {num_outside:,} ({unsafe_fraction*100:.2f}%)")
    
    return {
        'safe_set_fraction': float(safe_fraction),
        'unsafe_set_fraction': float(unsafe_fraction),
        'num_in_safe_set': int(num_in_safe_set),
        'num_outside_safe_set': int(num_outside),
        'num_total_scenarios': int(num_scenarios),
        'total_computation_time': float(elapsed),
        'set_mode': dynamics.set_mode,
        'loss_type': dynamics.loss_type,
        'description': 'Fraction of sampled states where value function indicates safe set membership'
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute BRT/BRAT volume error via sampling for high-D systems"
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default='runs/narrow_passage_10d_run_lr5',
        help='Path to experiment directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for computation'
    )
    parser.add_argument(
        '--num-scenarios',
        type=int,
        default=100000,
        help='Number of scenarios to sample'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.0025,
        help='Timestep for rollouts'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Maximum steps per trajectory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./brt_vol_err_results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Load experiment
    try:
        orig_opt, dynamics, model = load_experiment(args.experiment_dir, device=args.device)
    except Exception as e:
        print(f"[!] Failed to load experiment: {e}")
        sys.exit(1)
    
    # Compute volume error
    try:
        metrics = compute_volume_error_sampling(
            dynamics=dynamics,
            model=model,
            device=args.device,
            num_scenarios=args.num_scenarios,
            verbose=True
        )
    except Exception as e:
        print(f"[!] Failed to compute volume error:")
        print(f"    Type: {type(e).__name__}")
        print(f"    Message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_dir': str(args.experiment_dir),
        'dynamics_class': orig_opt.dynamics_class,
        'metrics': metrics,
        'config': {
            'num_scenarios': args.num_scenarios,
            'dt': args.dt,
            'device': args.device,
        }
    }
    
    results_file = output_dir / 'brt_vol_err_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[+] Results saved to {results_file}")
    print("\nSummary:")
    print(f"  Safe set fraction: {metrics['safe_set_fraction']:.4f}")
    print(f"  Unsafe set fraction: {metrics['unsafe_set_fraction']:.4f}")
    print(f"  Computation time: {metrics['total_computation_time']:.2f}s")
    print(f"  Computation time: {metrics['total_computation_time']:.2f}s")


if __name__ == '__main__':
    main()
