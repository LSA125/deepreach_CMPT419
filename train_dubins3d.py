"""
Offline Dubins3D Training Script
=================================

This script trains a SIREN network to learn the Hamilton-Jacobi value function
for the Dubins3D dynamic system using the DeepReach framework.

The script generates model_final.pth with learned safe control policy.

Usage:
    python train_dubins3d.py [--num_epochs 5000] [--device cuda:0]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def get_default_config(args):
    """Generate default training configuration for Dubins3D."""
    return {
        'mode': 'train',
        'experiment_class': 'DeepReach',
        'dynamics_class': 'Dubins3D',
        'experiment_name': args.experiment_name,
        'experiments_dir': './runs',
        
        # Dynamics parameters
        'goalR': args.goalR,
        'velocity': args.velocity,
        'omega_max': args.omega_max,
        'angle_alpha_factor': args.angle_alpha_factor,
        'set_mode': args.set_mode,
        'freeze_model': False,
        
        # Training parameters
        'device': args.device,
        'seed': args.seed,
        'num_epochs': args.num_epochs,
        'batch_size': 1,
        'lr': args.lr,
        'clip_grad': 0.0,
        'use_lbfgs': False,
        'adj_rel_grads': True,
        
        # Model architecture
        'model': 'sine',  # Use SIREN for smooth gradients
        'model_mode': 'mlp',
        'num_hl': args.num_hl,
        'num_nl': args.num_nl,
        'deepreach_model': 'exact',
        
        # Simulation data
        'numpoints': 65000,
        'pretrain': args.pretrain,
        'pretrain_iters': 2000,
        'tMin': 0.0,
        'tMax': 1.0,
        'counter_start': 0,
        'counter_end': args.counter_end,
        'num_src_samples': 1000,
        'num_target_samples': 0,
        
        # Loss options
        'minWith': args.minWith,
        'dirichlet_loss_divisor': 1.0,
        
        # Checkpointing
        'epochs_til_ckpt': args.epochs_til_ckpt,
        'steps_til_summary': 100,
        
        # Validation
        'val_x_resolution': 200,
        'val_y_resolution': 200,
        'val_z_resolution': 5,
        'val_time_resolution': 3,
        
        # CSL options (disabled by default)
        'use_CSL': False,
        'CSL_lr': 2e-5,
        'CSL_dt': 0.0025,
        'epochs_til_CSL': 10000,
        'num_CSL_samples': 1000000,
        'CSL_loss_frac_cutoff': 0.1,
        'max_CSL_epochs': 100,
        'CSL_loss_weight': 1.0,
        'CSL_batch_size': 1000,
    }

def build_command(config):
    """Build the run_experiment.py command from config."""
    cmd = ['python', 'run_experiment.py']
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    return cmd

def main():
    parser = argparse.ArgumentParser(
        description='Train a Dubins3D model using DeepReach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic avoid mode training
  python train_dubins3d.py --goalR 0.25 --velocity 0.6 --omega_max 1.1
  
  # Reach mode with custom epochs
  python train_dubins3d.py --goalR 0.25 --set_mode reach --num_epochs 10000
  
  # Full configurations
  python train_dubins3d.py --goalR 0.25 --velocity 0.6 --omega_max 1.1 \\
                           --angle_alpha_factor 1.2 --set_mode avoid \\
                           --num_epochs 5000 --device cuda:0
        '''
    )
    
    # Dynamics parameters
    parser.add_argument('--goalR', type=float, default=0.25,
                       help='Goal region radius (default: 0.25)')
    parser.add_argument('--velocity', type=float, default=0.6,
                       help='Agent forward velocity (default: 0.6)')
    parser.add_argument('--omega_max', type=float, default=1.1,
                       help='Max angular velocity (default: 1.1)')
    parser.add_argument('--angle_alpha_factor', type=float, default=1.2,
                       help='Angle normalization factor (default: 1.2)')
    parser.add_argument('--set_mode', type=str, choices=['reach', 'avoid'], 
                       default='avoid',
                       help='Reach or avoid mode (default: avoid)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5000,
                       help='Number of training epochs (default: 5000)')
    parser.add_argument('--num_hl', type=int, default=3,
                       help='Number of hidden layers (default: 3)')
    parser.add_argument('--num_nl', type=int, default=512,
                       help='Hidden layer width (default: 512)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--counter_end', type=int, default=-1,
                       help='Curriculum learning steps (default: -1, no curriculum)')
    
    # Experiment control
    parser.add_argument('--experiment_name', type=str, 
                       default='dubins3d_offline_training',
                       help='Experiment directory name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='PyTorch device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--minWith', type=str, choices=['none', 'zero', 'target'], 
                       default='target',
                       help='Loss mode (default: target for BRT)')
    parser.add_argument('--pretrain', action='store_true',
                       help='Pretrain with boundary conditions only')
    parser.add_argument('--epochs_til_ckpt', type=int, default=1000,
                       help='Save checkpoint every N epochs (default: 1000)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config(args)
    
    # Print configuration
    print("\n" + "="*70)
    print("  DeepReach Dubins3D Training Configuration")
    print("="*70 + "\n")
    
    print("Dynamics:")
    print(f"  Goal Radius:        {args.goalR}")
    print(f"  Velocity:           {args.velocity}")
    print(f"  Max Angular Vel:    {args.omega_max}")
    print(f"  Angle Factor:       {args.angle_alpha_factor}")
    print(f"  Mode:               {args.set_mode}")
    
    print("\nNetwork:")
    print(f"  Model Type:         SIREN (sine activation)")
    print(f"  Hidden Layers:      {args.num_hl}")
    print(f"  Hidden Width:       {args.num_nl}")
    print(f"  Input Dimension:    4 (t, x, y, θ)")
    print(f"  Output Dimension:   1 (Value function V)")
    
    print("\nTraining:")
    print(f"  Device:             {args.device}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Learning Rate:      {args.lr}")
    print(f"  Batch Size:         1")
    print(f"  Checkpoint Every:   {args.epochs_til_ckpt} epochs")
    print(f"  Pretrain:           {'Yes' if args.pretrain else 'No'}")
    
    print("\nOutput:")
    print(f"  Experiment Dir:     ./runs/{args.experiment_name}")
    print(f"  Model Checkpoints:  ./runs/{args.experiment_name}/training/checkpoints/")
    print(f"  Tensorboard Logs:   ./runs/{args.experiment_name}/training/summaries/")
    
    print("\n" + "="*70)
    print("  Starting training...")
    print("="*70 + "\n")
    
    # Build and run command
    cmd = build_command(config)
    
    try:
        result = subprocess.run(cmd, check=True)
        
        # Post-training information
        print("\n" + "="*70)
        print("  Training Complete!")
        print("="*70 + "\n")
        
        model_dir = Path('./runs') / args.experiment_name / 'training' / 'checkpoints'
        if model_dir.exists():
            print(f"Saved models in: {model_dir}")
            
            # List available models
            models = list(model_dir.glob('*.pth'))
            if models:
                print(f"\nAvailable checkpoints ({len(models)} total):")
                for model_file in sorted(models)[-5:]:  # Show last 5
                    print(f"  - {model_file.name}")
                
                print("\n✓ To use the final model (if model_final.pth exists):")
                print(f"  model_path = './runs/{args.experiment_name}/training/checkpoints/model_final.pth'")
        
        print("\n")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        print("Check the output above for details.")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
