"""
Trajectory validation for NarrowPassage scenario.

Tests the learned value function + optimal control on specific scenarios
with actual trajectory rollouts, comparing predictions vs actual outcomes.
"""

import json
import pickle
import inspect
import argparse
from pathlib import Path

import numpy as np
import torch
from datetime import datetime

from dynamics.dynamics import NarrowPassage
from dynamics import dynamics as dynamics_module
from utils.modules import SingleBVPNet


def load_experiment(experiment_dir: str, device: str = 'cuda:0'):
    """Load experiment config and trained model."""
    experiment_dir = Path(experiment_dir)
    
    with open(experiment_dir / 'orig_opt.pickle', 'rb') as f:
        orig_opt = pickle.load(f)
    
    # Reconstruct dynamics
    dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
    dynamics_kwargs = {
        argname: getattr(orig_opt, argname) 
        for argname in inspect.signature(dynamics_class).parameters.keys() 
        if argname != 'self'
    }
    dynamics = dynamics_class(**dynamics_kwargs)
    dynamics.deepreach_model = orig_opt.deepreach_model
    
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
    
    # Load checkpoint (latest, using numerical sort)
    ckpt_files = sorted(
        (experiment_dir / 'training' / 'checkpoints').glob('model_epoch_*.pth'),
        key=lambda x: int(x.stem.split('_')[-1])
    )[-1]
    
    ckpt = torch.load(ckpt_files, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print(f"[+] Loaded {experiment_dir.name}: {ckpt_files.name}")
    
    return dynamics, model


def get_optimal_control(model, dynamics, t, state, device='cuda:0'):
    """Get optimal control from learned value function via HJ derivative."""
    t_tensor = torch.full((1,), t, dtype=torch.float32, device=device)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    coords = torch.cat([t_tensor.unsqueeze(-1), state_tensor], dim=1)
    model_in = dynamics.coord_to_input(coords)
    
    # Enable gradient for backward pass
    model_in.requires_grad_(True)
    model_out = model({'coords': model_in})
    value = model_out['model_out'].squeeze(-1)
    
    # Compute gradient w.r.t. state (derivative of value)
    value.backward()
    grad_value = model_in.grad[0, 1:]  # Skip time dimension
    
    # Hamiltonian-based control: u* = argmin_u H(x, grad_V)
    # For minimum-time reach or reach-avoid, optimal control drives most negative Hamiltonian
    dv_state = dynamics.input_to_dv(model_in, grad_value.unsqueeze(0)).squeeze(0)
    
    # Extract control: next 4 dims are control derivatives
    control = dynamics.optimal_control(coords, dv_state.unsqueeze(0).unsqueeze(0)).squeeze()
    
    return control.detach().cpu().numpy()


def rollout_trajectory(model, dynamics, initial_state, T=2.0, dt=0.01, device='cuda:0'):
    """
    Roll out a trajectory using the learned optimal control policy.
    
    Returns:
        trajectory: (T/dt + 1, 10) array of states
        success: whether trajectory reached goals while avoiding obstacles
    """
    device_obj = torch.device(device)
    
    state = np.array(initial_state, dtype=np.float32)
    trajectory = [state.copy()]
    
    num_steps = int(T / dt)
    
    print(f"\n[*] Rolling out trajectory for {T}s ({num_steps} steps)")
    
    for step in range(num_steps):
        t = T - step * dt  # Time in reverse (counting down to 0)
        
        # Get optimal control from learned policy
        try:
            control = get_optimal_control(model, dynamics, t, state, device)
        except Exception as e:
            print(f"  Control computation failed at step {step}: {e}")
            return np.array(trajectory), False
        
        # Clamp control to bounds
        control[0] = np.clip(control[0], dynamics.aMin, dynamics.aMax)      # acceleration
        control[1] = np.clip(control[1], dynamics.psiMin, dynamics.psiMax)  # steering rate
        control[2] = np.clip(control[2], dynamics.aMin, dynamics.aMax)      # acceleration (car 2)
        control[3] = np.clip(control[3], dynamics.psiMin, dynamics.psiMax)  # steering rate (car 2)
        
        # Integrate dynamics
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        control_tensor = torch.tensor(control, dtype=torch.float32, device=device).unsqueeze(0)
        disturbance_tensor = torch.zeros((1, 0), device=device)
        
        dsdt = dynamics.dsdt(state_tensor, control_tensor, disturbance_tensor).squeeze(0)
        state = state + dt * dsdt.detach().cpu().numpy()
        
        # Wrap angles
        state = dynamics.equivalent_wrapped_state(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()
        trajectory.append(state.copy())
        
        # Check constraints at each step
        avoid_val = dynamics.avoid_fn(torch.tensor(state).unsqueeze(0)).item()
        if avoid_val < -0.1:  # Collision detected
            print(f"  COLLISION at step {step}: avoid_fn = {avoid_val:.4f}")
            return np.array(trajectory), False
    
    trajectory = np.array(trajectory)
    
    # Check final state
    final_state = trajectory[-1]
    avoid_val = dynamics.avoid_fn(torch.tensor(final_state).unsqueeze(0)).item()
    
    if dynamics.avoid_only:
        print(f"  Final: avoid_fn = {avoid_val:.4f}")
        success = (avoid_val >= -0.1)  # Only check collision avoidance in avoid_only mode
    else:
        reach_val = dynamics.reach_fn(torch.tensor(final_state).unsqueeze(0)).item()
        print(f"  Final: reach_fn = {reach_val:.4f}, avoid_fn = {avoid_val:.4f}")
        success = (reach_val <= 0.1) and (avoid_val >= -0.1)
    
    return trajectory, success


def test_narrow_passage_scenario(model, dynamics, device='cuda:0'):
    """
    Test scenario from image:
    - Car 1 (lower track): starts right, goal left
    - Car 2 (upper track): starts left, goal right  
    - Obstacle (stranded car) at center-lower blocks passage, forcing coordination
    """
    
    print("\n" + "="*60)
    print("NARROW PASSAGE SCENARIO TEST")
    print("="*60)
    
    print("\nScenario setup:")
    print("  Car 1 (lower): (6.0, -1.5, 0) -> goal (6.0, -1.4)")
    print("  Car 2 (upper): (-6.0, 1.5, π) -> goal (-6.0, 1.4)")
    print("  Obstacle (stranded car): (0.0, -1.8)")
    print("  Curbs: y ∈ [-2.8, 2.8]")
    
    # Initial conditions
    # Car 1: right side lower track, heading toward goal
    # Car 2: left side upper track, heading toward goal
    initial_state = np.array([
        -6.0, -1.5, 0.0, 1.0, 0.0,      # Car 1: position, heading, velocity, steering
        6.0, 1.5, np.pi, 1.0, 0.0     # Car 2: position, heading, velocity, steering (heading back)
    ], dtype=np.float32)
    
    # Get value function prediction at t=0
    print("\n[*] Value function prediction at initial state:")
    t_tensor = torch.zeros((1, 1), device=device)
    state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
    coords = torch.cat([t_tensor, state_tensor], dim=1)
    
    with torch.no_grad():
        model_in = dynamics.coord_to_input(coords)
        model_out = model({'coords': model_in})['model_out'].squeeze(-1)
        value = dynamics.io_to_value(model_in, model_out)
    
    avoid_val = dynamics.avoid_fn(state_tensor).item()
    value_pred = value.item()
    
    print(f"  Value (V): {value_pred:.4f}")
    print(f"  Avoid fn (g): {avoid_val:.4f}")
    print(f"  Avoid only mode: {dynamics.avoid_only}")
    if dynamics.avoid_only:
        print(f"  Safety: {'SAFE' if value_pred >= -0.1 else 'UNSAFE'} (V >= -0.1 in avoid mode)")
    else:
        reach_val = dynamics.reach_fn(state_tensor).item()
        print(f"  Reach fn (l): {reach_val:.4f}")
        print(f"  Safety: {'SAFE' if value_pred >= -0.1 else 'UNSAFE'} (reach-avoid mode)")
    
    # Rollout trajectory
    T = 10.0  # 10 seconds
    trajectory, success = rollout_trajectory(model, dynamics, initial_state, T=T, dt=0.025, device=device)
    
    print(f"\n[*] Trajectory rollout result:")
    print(f"  Duration: {T}s")
    print(f"  Steps: {len(trajectory)}")
    print(f"  SUCCESS: {success}")
    
    # Compute statistics
    final_state = trajectory[-1]
    distances_to_goals = {
        'car1_to_goal': np.linalg.norm(final_state[0:2] - np.array([6.0, -1.4])),
        'car2_to_goal': np.linalg.norm(final_state[5:7] - np.array([-6.0, 1.4])),
    }
    
    print(f"\n[*] Final state distances to goal:")
    for car, dist in distances_to_goals.items():
        print(f"  {car}: {dist:.3f}")
    
    return {
        'initial_value': value_pred,
        'avoid_fn': avoid_val,
        'success': success,
        'distances_to_goals': distances_to_goals,
        'trajectory': trajectory.tolist(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True, help='Path to experiment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()
    
    dynamics, model = load_experiment(args.experiment_dir, device=args.device)
    
    result = test_narrow_passage_scenario(model, dynamics, device=args.device)
    
    # Save results
    output_file = Path(args.experiment_dir) / 'scenario_test_result.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[+] Results saved to {output_file}")
