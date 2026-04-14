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


def check_value_gradient(model, dynamics, device='cuda:0'):
    # Sweep X for the relative position (e.g., from -3 to 3 units away)
    x_sweep = np.linspace(-3, 3, 100)
    values = []
    
    # Use train() mode to ensure the model behaves exactly as it did during training
    model.train() 
    
    with torch.no_grad():
        for x in x_sweep:
            # FIX: Use a 3D state [relative_x, relative_y, relative_theta]
            # matching your Air3D / Dubins3D model's expected 4D input (1+3)
            state = np.array([x, 0.2, 0.0], dtype=np.float32) 
            
            t_tensor = torch.tensor([[1.0]], device=device) 
            s_tensor = torch.tensor(state, device=device).unsqueeze(0)
            
            # coords becomes [1, 4] -> [Time, X, Y, Theta]
            coords = torch.cat([t_tensor, s_tensor], dim=1)
            
            model_in = dynamics.coord_to_input(coords)
            model_out = model({'coords': model_in})
            val = model_out['model_out']
            values.append(val.item())
    
    # Plotting the slice
    plt.figure(figsize=(10, 5))
    plt.plot(x_sweep, values, label='Value V(t, s)', color='blue', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', label='Pursuer/Target Center')
    
    plt.title(f"Air3D Value Slice (t=1.0, y=0.2, th=0.0)")
    plt.xlabel("Relative X Position")
    plt.ylabel("Value (V > 0 is Safe)")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    
    output_dir = Path('brt_vol_err_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'air3d_gradient_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[+] Gradient sweep saved to {output_file}")
    

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, default='runs/narrow_passage_10d_run_lr5', help='Path to experiment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()
    
    dynamics, model = load_experiment(args.experiment_dir, device=args.device)
    check_value_gradient(model, dynamics, device=args.device)