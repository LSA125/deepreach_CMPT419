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

from dynamics import dynamics as dynamics_module
from utils.modules import SingleBVPNet

def load_experiment(experiment_dir: str, device: str = 'cuda:0'):
    """Load experiment config and trained model."""
    experiment_dir = Path(experiment_dir)
    
    with open(experiment_dir / 'orig_opt.pickle', 'rb') as f:
        orig_opt = pickle.load(f)
    print(orig_opt)
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

    dynamics, model = load_experiment(args.experiment_dir, args.device) 