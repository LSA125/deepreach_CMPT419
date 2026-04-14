import pickle
import inspect
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Make sure these imports match your project structure
from dynamics import dynamics as dynamics_module
from utils.modules import SingleBVPNet

def get_value_gradient(model, dynamics, t, state, device='cuda:0'):
    """Returns the gradient dV/dstate at a given time and state."""
    # Ensure coords tracks gradients
    coords = torch.tensor(np.append([t], state), dtype=torch.float32, device=device).unsqueeze(0)
    coords.requires_grad_(True)
    
    model_in = dynamics.coord_to_input(coords)
    model_out = model({'coords': model_in})
    value = model_out['model_out']
    
    # Calculate gradient
    dv_dall = torch.autograd.grad(
        value, coords, 
        grad_outputs=torch.ones_like(value),
        allow_unused=True
    )[0]

    # --- CRITICAL FIX: Handle the NoneType case ---
    if dv_dall is None:
        # If the graph is broken or the value is constant, return zeros
        return torch.zeros_like(coords)

    # Handle 'diff' architecture (analytical boundary gradients)
    if hasattr(dynamics, 'deepreach_model') and dynamics.deepreach_model == "diff":
        state_only = coords[:, 1:].detach().requires_grad_(True)
        boundary_val = dynamics.boundary_fn(state_only)
        d_boundary = torch.autograd.grad(boundary_val, state_only, torch.ones_like(boundary_val))[0]
        dv_dall[:, 1:] += d_boundary

    return dv_dall.detach()


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
        if argname != 'self' and hasattr(orig_opt, argname)
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
        key=lambda x: int(x.stem.split('_')[-1])
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
    
    model.train()
    print(f"[+] Loaded {experiment_dir.name}: {latest_ckpt.name}")
    
    return dynamics, model


def get_optimal_control(model, dynamics, t, state, device='cuda:0'):
    """Fetches the optimal control for the Evader based on the BRT gradient."""
    coords = torch.tensor(np.append([t], state), dtype=torch.float32, device=device).unsqueeze(0)
    coords.requires_grad_(True)
    
    model_in = dynamics.coord_to_input(coords)
    model_out = model({'coords': model_in})
    value = model_out['model_out']
    
    dv_dall = torch.autograd.grad(
        value, coords, 
        grad_outputs=torch.ones_like(value),
        allow_unused=True
    )[0]
    
    if dv_dall is None:
        return np.zeros(dynamics.control_dim)

    if dynamics.deepreach_model == "diff":
        state_only = coords[:, 1:].detach().requires_grad_(True)
        boundary_val = dynamics.boundary_fn(state_only)
        d_boundary = torch.autograd.grad(boundary_val, state_only, torch.ones_like(boundary_val))[0]
        dv_dall[:, 1:] += d_boundary

    control = dynamics.optimal_control(coords.detach(), dv_dall.detach())
    return control.detach().cpu().numpy().flatten()


def run_air3d_safety_test(model, dynamics, device='cuda:0'):
    # Relative state: [x, y, relative_heading]
    # In Air3D, the Pursuer is fixed at the origin facing +Y (or +X depending on convention).
    # We start the evader nearby.
    state = np.array([0.8, 0.8, -np.pi*0.75], dtype=np.float32)
    dt = 0.01
    max_steps = 500
    
    trajectory = [state.copy()]
    safety_values = []
    active_intervention = []

    print(f"[*] Starting AIR3D Safety Test (Collision Radius: {dynamics.collisionR})")

    for step in range(max_steps):
        # 1. Evaluate current Safety Value V(t, s)
        t_tensor = torch.tensor([[1.0]], device=device)
        s_tensor = torch.tensor(state, device=device).unsqueeze(0)
        
        with torch.no_grad():
            coords = torch.cat([t_tensor, s_tensor], dim=1)
            model_in = dynamics.coord_to_input(coords)
            val = model({'coords': model_in})['model_out'].item()
            safety_values.append(val)

        # 2. Evader Control Logic
        u_nominal = np.zeros(dynamics.control_dim)
        
        # Buffer zone just above 0.0
        if val < 0.05: 
            u_evader = get_optimal_control(model, dynamics, 1.0, state, device=device)
            active_intervention.append(True)
        else:
            u_evader = u_nominal
            active_intervention.append(False)

        # 3. Pursuer Logic (Optimal Disturbance trying to catch Evader)
        dv_dall = get_value_gradient(model, dynamics, 1.0, state, device=device)
        d_pursuer = dynamics.optimal_disturbance(torch.tensor(state, device=device).unsqueeze(0), dv_dall)
        d_pursuer = d_pursuer.detach().cpu().numpy().flatten()

        # 4. Integrate Dynamics
        state_tensor = torch.tensor(state, device=device).unsqueeze(0)
        
        # Ensure controls/disturbances are correctly shaped for dsdt (Batch, Dim)
        ctrl_tensor = torch.tensor(u_evader, device=device, dtype=torch.float32).reshape(1, -1)
        dist_tensor = torch.tensor(d_pursuer, device=device, dtype=torch.float32).reshape(1, -1)
        
        with torch.no_grad():
            dsdt = dynamics.dsdt(state_tensor, ctrl_tensor, dist_tensor).squeeze(0).cpu().numpy()
        
        state = state + dsdt * dt
        
        # Wrap relative heading angle between -pi and pi
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        trajectory.append(state.copy())

        # Check for capture
        if np.linalg.norm(state[:2]) <= dynamics.collisionR:
            print(f"[!] CAPTURE at step {step}! Pursuer caught the Evader.")
            break

    plot_air3d_results(np.array(trajectory), safety_values, active_intervention, dynamics.collisionR)


def plot_air3d_results(traj, values, interventions, collision_r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Path Plot (Relative Frame) ---
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Evader Path (Relative)')
    
    # Pursuer is fixed at the origin in the relative frame
    ax1.plot(0, 0, 'ro', markersize=8, label='Pursuer (Origin)')
    ax1.add_patch(Circle((0, 0), collision_r, color='red', alpha=0.3, label=f'Capture Zone (R={collision_r})'))
    
    # Mark Start and End
    ax1.scatter(traj[0, 0], traj[0, 1], c='green', marker='o', s=100, label='Evader Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], c='black', marker='x', s=100, label='Evader End')

    # Highlight intervention points
    interv_idx = np.where(interventions)[0]
    if len(interv_idx) > 0:
        ax1.scatter(traj[interv_idx, 0], traj[interv_idx, 1], c='orange', s=20, zorder=5, label='Safety Control Active')
    
    ax1.set_title("Air3D: Relative Flight Path")
    ax1.set_xlabel("Relative X")
    ax1.set_ylabel("Relative Y")
    ax1.axis('equal') # Keeps the capture zone perfectly circular
    ax1.legend()
    ax1.grid(True)

    # --- Value Plot ---
    ax2.plot(values, color='purple', linewidth=2, label='Safety Value V(s)')
    ax2.axhline(y=0, color='r', linestyle='--', label='Collision Boundary')
    ax2.set_title("Value Function over Time")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("V (Positive is Safe)")
    ax2.legend()
    ax2.grid(True)
    
    output_dir = Path('brt_vol_err_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'air3d_sim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[*] Saved visualization to {output_file}")


if __name__ == '__main__':
    # Usage: python3 safety_test.py --experiment-dir runs/air3d_run_lr5
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True)
    args = parser.parse_args()
    
    dyn, net = load_experiment(args.experiment_dir)
    run_air3d_safety_test(net, dyn)