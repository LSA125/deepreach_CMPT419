"""
Visualize the two-car narrow passage scenario with the learned BRAT policy.
Creates a birds-eye view similar to the reference image.
"""

import json
import pickle
import inspect
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

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
    
    # Load checkpoint (latest)
    ckpt_dir = experiment_dir / 'training' / 'checkpoints'
    ckpt_files = sorted(
        ckpt_dir.glob('model_epoch_*.pth'),
        key=lambda x: int(x.stem.split('_')[-1]) # Extracts the number from the filename
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth files found in {ckpt_dir}")
    latest_ckpt = ckpt_files[-1]
    print(f"[+] Found {len(ckpt_files)} checkpoints. Latest: {latest_ckpt.name}")

    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    # Use train mode to enable gradient tracking through the model
    model.train()
    print(f"[+] Loaded {experiment_dir.name}: {latest_ckpt.name}")
    
    return dynamics, model


def get_optimal_control(model, dynamics, t, state, device='cuda:0', verbose=False):
    # 1. Create the raw coordinate tensor and enable grad
    # [Time, x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
    coords = torch.tensor(np.append([t], state), dtype=torch.float32, device=device).unsqueeze(0)
    coords.requires_grad_(True)
    
    # 2. Pass through dynamics normalization AND model
    model_in = dynamics.coord_to_input(coords)
    model_out = model({'coords': model_in})
    value = model_out['model_out'] # Raw value from model
    
    # 3. Direct gradient: dV / dcoords
    # This automatically handles the chain rule through coord_to_input
    dv_dall = torch.autograd.grad(
        value, coords, 
        grad_outputs=torch.ones_like(value),
        allow_unused=True
    )[0]
    
    if dv_dall is None:
        return np.zeros(dynamics.control_dim)

    # 4. DeepReach "Diff" mode adjustment
    # If the model was trained with the 'diff' architecture, we add the boundary gradient
    if dynamics.deepreach_model == "diff":
        # We need d(Boundary)/d(state)
        state_only = coords[:, 1:].detach().requires_grad_(True)
        boundary_val = dynamics.boundary_fn(state_only)
        d_boundary = torch.autograd.grad(boundary_val, state_only, torch.ones_like(boundary_val))[0]
        
        # Combine: Learned Gradient + Analytical Boundary Gradient
        # Note: Index 0 of dv_dall is time, so we add to index 1+
        dv_dall[:, 1:] += d_boundary

    # 5. Get optimal control from the dynamics class
    # We pass the full gradient vector [dV/dt, dV/dx1, ...]
    control = dynamics.optimal_control(coords.detach(), dv_dall.detach())
    
    return control.squeeze(0).cpu().numpy()


def get_simple_control(dynamics, state, goal):
    """Simple control policy: head toward goal, avoid obstacles."""
    x, y, theta, v, phi = state[0:5]
    
    # Direction to goal
    dx = goal[0] - x
    dy = goal[1] - y
    angle_to_goal = np.arctan2(dy, dx)
    
    # Steering to align with goal
    steering_error = angle_to_goal - theta
    steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))  # Normalize to [-pi, pi]
    
    # Control: accelerate and steer toward goal
    a = np.clip(2.0, dynamics.aMin, dynamics.aMax)  # Moderate acceleration
    psi = np.clip(steering_error * 3.0, dynamics.psiMin, dynamics.psiMax)  # Steering rate proportional to error
    
    return np.array([a, psi, a, psi])  # Same for both cars


def rollout_trajectory(model, dynamics, initial_state, T=1.0, dt=0.025, device='cuda:0', verbose=False):
    """Roll out trajectory using learned optimal control from value function."""
    state = np.array(initial_state, dtype=np.float32)
    trajectory = [state.copy()]
    
    num_steps = int(T / dt)
    
    for step in range(num_steps):
        t = T - step * dt
        
        try:
            # Get optimal control from learned value function
            control = get_optimal_control(model, dynamics, t, state, device=device, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  Control computation failed at step {step}: {e}")
            return np.array(trajectory), False
        
        # Clamp control to bounds
        control[0] = np.clip(control[0], dynamics.aMin, dynamics.aMax)
        control[1] = np.clip(control[1], dynamics.psiMin, dynamics.psiMax)
        control[2] = np.clip(control[2], dynamics.aMin, dynamics.aMax)
        control[3] = np.clip(control[3], dynamics.psiMin, dynamics.psiMax)
        
        # Integrate dynamics
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        control_tensor = torch.tensor(control, dtype=torch.float32, device=device).unsqueeze(0)
        disturbance_tensor = torch.zeros((1, 0), device=device)
        
        dsdt = dynamics.dsdt(state_tensor, control_tensor, disturbance_tensor).squeeze(0)
        state = state + dt * dsdt.detach().cpu().numpy()
        
        # Wrap angles
        state = dynamics.equivalent_wrapped_state(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()
        trajectory.append(state.copy())
        
        # Check collision
        state_t = torch.tensor(state).unsqueeze(0)
        avoid_val = dynamics.avoid_fn(state_t).item()
        if avoid_val < -0.1:
            if verbose:
                # Debug: show which component failed
                dist_lc_R1 = state[1] - dynamics.curb_positions[0] - 0.5*dynamics.L
                dist_lc_R2 = state[6] - dynamics.curb_positions[0] - 0.5*dynamics.L
                dist_uc_R1 = dynamics.curb_positions[1] - state[1] - 0.5*dynamics.L
                dist_uc_R2 = dynamics.curb_positions[1] - state[6] - 0.5*dynamics.L
                dist_R1R2 = np.linalg.norm(state[0:2] - state[5:7]) - dynamics.L
                print(f"  COLLISION at step {step}: avoid_fn = {avoid_val:.4f}")
                print(f"    dist_R1R2 (car-to-car) = {dist_R1R2:.4f}")
                print(f"    Car1 y={state[1]:.2f}, Car2 y={state[6]:.2f}")
            return np.array(trajectory), False
    
    trajectory = np.array(trajectory)
    return trajectory, True


def visualize_scenario(model, dynamics, device='cuda:0'):
    """Create birds-eye visualization of the narrow passage scenario."""
    
    print("\n[*] Running two-car narrow passage scenario...")
    print("[*] Scenario: Cars at crossing lanes, must coordinate to pass through narrow passage")
    
    print("\n[*] Value function analysis:")
    # Check if any state is "safe" (V >= -0.005)
    print("[*] Scanning for reachable states (V >= 0)...")
    safe_count = 0
    v_min = float('inf')
    v_max = float('-inf')
    
    for x1 in np.linspace(-8, 8, 9):
        for x2 in np.linspace(-8, 8, 9):
            test_state = np.array([
                x1, -1.5, 0.0, 0.1, 0.0,
                x2, 1.5, np.pi, 0.1, 0.0
            ], dtype=np.float32)
            t_tensor = torch.full((1,), 10.0, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(test_state, dtype=torch.float32, device=device).unsqueeze(0)
            coords = torch.cat([t_tensor.unsqueeze(-1), state_tensor], dim=1)
            model_in = dynamics.coord_to_input(coords)
            model_out = model({'coords': model_in})
            value_normalized = model_out['model_out'].squeeze(-1).item()
            v_min = min(v_min, value_normalized)
            v_max = max(v_max, value_normalized)
            if value_normalized >= 0:
                safe_count += 1
    
    print(f"  Value range: [{v_min:.6f}, {v_max:.6f}]")
    print(f"  Reachable states found: {safe_count}/81")
    if safe_count == 0:
        print(f"      Task appears UNSOLVABLE: The model judges all states as unreachable!")
        print(f"      Extreme control commands indicate: 'no solution exists for this scenario'")
    
    # Initial state: cars starting at their goal's x-coordinate but opposite lane
    # This forces them to move along opposite-y lanes AND coordinate
    # Car 1: starts at (6.0, 1.5) [upper lane, same x as goal], goal is (6.0, -1.4) [lower lane]
    # Car 2: starts at (-6.0, -1.5) [lower lane, same x as goal], goal is (-6.0, 1.4) [upper lane]
    # They must pass through the center where obstacle is
    initial_state = np.array([
        -2.0, -1.2, 0, 2.0, 0.0,     # Car 1: x=6 (goal x), upper lane, heading down
        2.0, 1.2, np.pi, 2.0, 0.0      # Car 2: x=-6 (goal x), lower lane, heading up
    ], dtype=np.float32)
    
    print(f"  Car 1: starts ({initial_state[0]:.1f}, {initial_state[1]:.1f}) → goal ({dynamics.goalX[0]}, {dynamics.goalY[0]})")
    print(f"  Car 2: starts ({initial_state[5]:.1f}, {initial_state[6]:.1f}) → goal ({dynamics.goalX[1]}, {dynamics.goalY[1]})")
    
    # Rollout with learned optimal control policy
    trajectory, success = rollout_trajectory(model, dynamics, initial_state, T=1.0, dt=0.025, device=device, verbose=True)
    
    print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILURE'}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set limits
    ax.set_xlim(-8, 8)
    ax.set_ylim(-3.8, 3.8)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Standard image coordinates (y down)
    
    # Draw curbs (lane boundaries)
    ax.axhline(y=dynamics.curb_positions[0], color='brown', linewidth=3, label='Lower curb', linestyle='-', alpha=0.7)
    ax.axhline(y=dynamics.curb_positions[1], color='brown', linewidth=3, label='Upper curb', linestyle='-', alpha=0.7)
    
    # Draw stranded car (obstacle)
    stranded_x, stranded_y = dynamics.stranded_car_pos
    stranded_size = 2 * dynamics.L
    rect = Rectangle((stranded_x - stranded_size/2, stranded_y - stranded_size/2), 
                      stranded_size, stranded_size, 
                      color='red', alpha=0.6, label='Obstacle (stranded car)')
    ax.add_patch(rect)
    
    # Draw goals
    ax.plot(dynamics.goalX[0], dynamics.goalY[0], 'g*', markersize=20, label='Car 1 goal', zorder=5)
    ax.plot(dynamics.goalX[1], dynamics.goalY[1], 'b*', markersize=20, label='Car 2 goal', zorder=5)
    
    # Draw trajectories
    car1_traj = trajectory[:, 0:2]
    car2_traj = trajectory[:, 5:7]
    
    ax.plot(car1_traj[:, 0], car1_traj[:, 1], 'g-', linewidth=2, alpha=0.7, label='Car 1 trajectory')
    ax.plot(car2_traj[:, 0], car2_traj[:, 1], 'b-', linewidth=2, alpha=0.7, label='Car 2 trajectory')
    
    # Draw start positions
    car1_start = car1_traj[0]
    car2_start = car2_traj[0]
    ax.plot(car1_start[0], car1_start[1], 'go', markersize=12, label='Car 1 start', zorder=5)
    ax.plot(car2_start[0], car2_start[1], 'bo', markersize=12, label='Car 2 start', zorder=5)
    
    # Draw final positions with arrows
    car1_final = car1_traj[-1]
    car2_final = car2_traj[-1]
    ax.plot(car1_final[0], car1_final[1], 'g^', markersize=14, label='Car 1 final', zorder=5)
    ax.plot(car2_final[0], car2_final[1], 'b^', markersize=14, label='Car 2 final', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Two-Car Narrow Passage\n(Birds-Eye View)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add status text
    status_text = f"Status: {'✓ SUCCESS' if success else '✗ COLLISION'}\nTime: {len(trajectory) * 0.05:.1f}s"
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightgreen' if success else 'lightcoral', alpha=0.8))
    
    # Save figure
    output_dir = Path('brt_vol_err_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'scenario_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[+] Figure saved to {output_file}")
    
    # Print statistics
    print(f"\n[*] Trajectory Statistics:")
    print(f"  Car 1 - Start: ({car1_start[0]:.2f}, {car1_start[1]:.2f})")
    print(f"         End:   ({car1_final[0]:.2f}, {car1_final[1]:.2f})")
    dist1 = np.linalg.norm(car1_final - np.array([dynamics.goalX[0], dynamics.goalY[0]]))
    print(f"         Distance to goal: {dist1:.2f} m")
    print(f"  Car 2 - Start: ({car2_start[0]:.2f}, {car2_start[1]:.2f})")
    print(f"         End:   ({car2_final[0]:.2f}, {car2_final[1]:.2f})")
    dist2 = np.linalg.norm(car2_final - np.array([dynamics.goalX[1], dynamics.goalY[1]]))
    print(f"         Distance to goal: {dist2:.2f} m")
    
    return trajectory, success

def check_value_gradient(model, dynamics, device='cuda:0'):
    # Sweep X for Car 1 from way behind the obstacle to way past it
    x_sweep = np.linspace(-7, 7, 100)
    values = []
    
    model.eval()
    with torch.no_grad():
        for x in x_sweep:
            # Create a full 10D state
            # [x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
            state = np.array([x, -1.2, 0.0, 2.0, 0.0,  # Car 1
                             5.0, 1.2, np.pi, 2.0, 0.0], # Car 2 (parked far away)
                             dtype=np.float32)
            
            # Use t=1.0 since that is your trained tMax
            t_tensor = torch.tensor([[1.0]], device=device) 
            s_tensor = torch.tensor(state, device=device).unsqueeze(0)
            
            coords = torch.cat([t_tensor, s_tensor], dim=1)
            model_in = dynamics.coord_to_input(coords)
            val = model({'coords': model_in})['model_out']
            values.append(val.item())
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_sweep, values, label='Learned Value V', color='blue', linewidth=2)
    
    # CRITICAL: If the range is tiny, this forces us to see the "stuckness"
    plt.ylim(min(values)-0.01, max(values)+0.01) 
    
    plt.axvline(x=0, color='red', linestyle='--', label='Obstacle Location')
    plt.title(f"Value Function 'Slice' (Min: {min(values):.4f}, Max: {max(values):.4f})")
    plt.xlabel("Car 1 X-Position")
    plt.ylabel("Value")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    output_dir = Path('brt_vol_err_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'gradient_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, default='runs/narrow_passage_10d_run_lr5', help='Path to experiment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()
    
    dynamics, model = load_experiment(args.experiment_dir, device=args.device)
    visualize_scenario(model, dynamics, device=args.device)
    
    plt.show()
    check_value_gradient(model, dynamics, device=args.device)
