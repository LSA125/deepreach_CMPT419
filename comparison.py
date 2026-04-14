"""
Comparison module for evaluating DeepReach neural network against ground-truth PDE solver.

Currently supports Dubins3D dynamics.
Extensible to all 9 DeepReach dynamics via configurable comparison runners.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import jax
import jax.numpy as jnp

# Ensure local hj_reachability copy is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "sub_project" / "hj_reachability"))

# DeepReach imports
from dynamics.dynamics import Dubins3D

# hj_reachability imports
try:
    from sub_project.hj_reachability.hj_reachability import solver as hj_solver
    from sub_project.hj_reachability.hj_reachability import sets as hj_sets
    from sub_project.hj_reachability.hj_reachability import grid as hj_grid
    from sub_project.hj_reachability.hj_reachability.systems import air3d
    HJ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hj_reachability: {e}")
    HJ_AVAILABLE = False

class Air3DComparisonConfig:
    """Configuration for your specific Air3D (lr5 scenario) comparison."""
    def __init__(
        self,
        collision_radius: float = 0.25,
        velocity: float = 0.5,
        omega_max: float = 1.0,
        angle_alpha_factor: float = 1.0,
        domain_bounds: Optional[list] = None,
        resolution: int = 50,
        time_horizon: float = 1.0,
        num_time_steps: int = 11,
    ):
        self.collision_radius = collision_radius
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.domain_bounds = domain_bounds or [[-1.5, 1.5], [-1.5, 1.5], [-np.pi, np.pi]]
        self.resolution = resolution
        self.time_horizon = time_horizon
        self.num_time_steps = num_time_steps

class Air3DComparisonRunner:
    def __init__(self, config: Air3DComparisonConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Initialize your custom DeepReach dynamics
        # Note: Importing your specific Air3D class here
        from dynamics.dynamics import Air3D as DeepReachAir3D
        self.dr_dynamics = DeepReachAir3D(
            collisionR=config.collision_radius,
            velocity=config.velocity,
            omega_max=config.omega_max,
            angle_alpha_factor=config.angle_alpha_factor
        )

        if HJ_AVAILABLE:
            # hj_reachability Air3d dynamics
            # evader_speed, pursuer_speed, evader_turn, pursuer_turn
            self.hj_system = air3d.Air3d(
                evader_speed=config.velocity,
                pursuer_speed=config.velocity,
                evader_max_turn_rate=config.omega_max,
                pursuer_max_turn_rate=config.omega_max
            )
            
            # Setup periodic grid for HJ
            domain_lo = np.array([config.domain_bounds[0][0], config.domain_bounds[1][0], 0.0])
            domain_hi = np.array([config.domain_bounds[0][1], config.domain_bounds[1][1], 2 * np.pi])
            self.hj_grid = hj_grid.Grid.from_lattice_parameters_and_boundary_conditions(
                domain=hj_sets.Box(lo=domain_lo, hi=domain_hi),
                shape=(config.resolution,) * 3,
                periodic_dims=(2,)
            )

    def compute_ground_truth(self):
        # Boundary function: norm(x,y) - R
        boundary_vals = np.linalg.norm(self.hj_grid.states[..., :2], axis=-1) - self.config.collision_radius
        
        times = np.linspace(0, self.config.time_horizon, self.config.num_time_steps)
        
        # Air3D lr5 is usually a BRT (Avoid set)
        settings = hj_solver.SolverSettings.with_accuracy("high")
        settings = settings.replace(
            hamiltonian_postprocessor=hj_solver.backwards_reachable_tube
        )
        
        start = time.time()
        values = hj_solver.solve(settings, self.hj_system, self.hj_grid, -times, boundary_vals)
        return np.array(values)[::-1], time.time() - start

    def evaluate_deepreach(self, model):
        """Evaluate DeepReach over the HJ grid states, mapping theta back to [-pi, pi]."""
        hj_states = self.hj_grid.states.reshape(-1, 3) # [x, y, theta_hj]
        
        # Map HJ theta [0, 2pi] back to DR theta [-pi, pi]
        dr_theta = (hj_states[:, 2] + np.pi) % (2 * np.pi) - np.pi
        dr_states = np.stack([hj_states[:, 0], hj_states[:, 1], dr_theta], axis=-1)
        
        # We'll evaluate at the final time horizon (t=1.0 usually)
        t_vec = np.full((dr_states.shape[0], 1), self.config.time_horizon)
        coords = torch.from_numpy(np.hstack([t_vec, dr_states])).float().to(self.device)
        
        model.eval()
        start = time.time()
        with torch.no_grad():
            inputs = self.dr_dynamics.coord_to_input(coords)
            model_out = model({'coords': inputs})['model_out'][..., 0]
            values = self.dr_dynamics.io_to_value(inputs, model_out)
        
        return values.cpu().numpy().reshape(self.hj_grid.shape), time.time() - start
    
    def run(self, model: torch.nn.Module, output_dir: Optional[Path] = None) -> Dict:
        """Runs the full evaluation pipeline for Air3D."""
        print("\n" + "=" * 50)
        print("Starting Air3D Comparison (DeepReach vs HJ)")
        print("=" * 50)

        # 1. Compute HJ Ground Truth
        print("[1/3] Solving ground truth via hj_reachability...")
        gt_values, gt_time = self.compute_ground_truth()
        print(f"      Solve complete in {gt_time:.2f}s")

        # 2. Evaluate DeepReach
        print("[2/3] Evaluating DeepReach model at time horizon...")
        pred_values, pred_time = self.evaluate_deepreach(model)
        print(f"      Evaluation complete in {pred_time:.2f}s")

        # 3. Compute Metrics
        print("[3/3] Computing performance metrics...")
        mse = np.mean((gt_values - pred_values) ** 2)
        
        # BRT Volume (V <= 0)
        gt_mask = (gt_values <= 0).astype(float)
        pred_mask = (pred_values <= 0).astype(float)
        
        # Calculate Intersection over Union (IoU) for the safety set
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 1.0

        results = {
            "metrics": {
                "mse": float(mse),
                "iou": float(iou),
                "gt_volume": int(gt_mask.sum()),
                "pred_volume": int(pred_mask.sum())
            },
            "timing": {
                "hj_solve_time": gt_time,
                "dr_eval_time": pred_time
            }
        }

        if output_dir:
            out_path = Path(output_dir) / "air3d_comparison.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {out_path}")

        return results

class DubinsComparisonConfig:
    """Configuration for Dubins3D vs DubinsCarCAvoid comparison."""
    
    def __init__(
        self,
        goal_radius: float = 0.2,
        velocity: float = 0.6,
        omega_max: float = 1.1,
        angle_alpha_factor: float = 1.2,
        domain_bounds: Optional[list] = None,
        resolution: int = 40,
        time_horizon: float = 0.5,
        num_time_steps: int = 11,
        set_mode: str = "reach",
    ):
        """
        Args:
            goal_radius: Radius of goal/collision region.
            velocity: Constant forward velocity.
            omega_max: Max angular velocity.
            angle_alpha_factor: Scaling factor for angle in model units.
            domain_bounds: [[x_min, x_max], [y_min, y_max], [theta_min, theta_max]]
            resolution: Grid points per dimension.
            time_horizon: Max time to solve to.
            num_time_steps: Number of time slices to evaluate.
            set_mode: "reach" or "avoid" (BRT).
        """
        self.goal_radius = goal_radius
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.domain_bounds = domain_bounds or [[-1.0, 1.0], [-1.0, 1.0], [-np.pi, np.pi]]
        self.resolution = resolution
        self.time_horizon = time_horizon
        self.num_time_steps = num_time_steps
        self.set_mode = set_mode


class DubinsComparisonRunner:
    """Runs comparison between DeepReach Dubins3D and hj_reachability DubinsCarCAvoid."""
    
    def __init__(self, config: DubinsComparisonConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.results = {}
        
        # Initialize DeepReach dynamics
        self.deepreach_dynamics = Dubins3D(
            goalR=config.goal_radius,
            velocity=config.velocity,
            omega_max=config.omega_max,
            angle_alpha_factor=config.angle_alpha_factor,
            set_mode=config.set_mode,
            freeze_model=False,
        )
        
        # Initialize hj_reachability dynamics (if available)
        if HJ_AVAILABLE:
            self._init_hj_dynamics()
        else:
            raise ImportError("hj_reachability not available")
    
    def _init_hj_dynamics(self):
        """Initialize hj_reachability system and grid."""
        # DubinsCarCAvoid is an alias for Air3d with specific configs
        # We configure it as a simple Dubins car (evader speed = velocity, pursuer speed = 0)
        self.hj_system = air3d.DubinsCarCAvoid(
            evader_speed=self.config.velocity,
            pursuer_speed=0.0,  # No disturbance
        )
        
        # Create grid: [x, y, theta] with theta in [0, 2π]
        # Map DeepReach's θ ∈ [-π, π] to [0, 2π]
        domain_lo = np.array([
            self.config.domain_bounds[0][0],
            self.config.domain_bounds[1][0],
            0.0,  # theta in [0, 2π]
        ])
        domain_hi = np.array([
            self.config.domain_bounds[0][1],
            self.config.domain_bounds[1][1],
            2.0 * np.pi,
        ])
        
        # Create hj.sets.Box domain
        self.hj_domain = hj_sets.Box(lo=domain_lo, hi=domain_hi)
        
        # Create grid with periodic dimension for angle (dim 2)
        self.hj_grid = hj_grid.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=self.hj_domain,
            shape=tuple([self.config.resolution] * 3),
            periodic_dims=(2,),  # theta is periodic
        )
    
    def build_evaluation_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build shared evaluation grid for both solvers.
        
        Returns:
            coords: (N, 4) array of [t, x, y, theta] in DeepReach units.
            hj_coords: (N, 3) array of [x, y, theta] in hj_reachability units (theta in [0, 2π]).
        """
        # Create spatial grid (map DeepReach [-π, π] → [0, 2π])
        x_lin = np.linspace(self.config.domain_bounds[0][0], self.config.domain_bounds[0][1], self.config.resolution)
        y_lin = np.linspace(self.config.domain_bounds[1][0], self.config.domain_bounds[1][1], self.config.resolution)
        theta_dr = np.linspace(self.config.domain_bounds[2][0], self.config.domain_bounds[2][1], self.config.resolution)
        theta_hj = (theta_dr + np.pi) % (2 * np.pi)  # Convert to [0, 2π]
        
        X, Y, Theta_dr = np.meshgrid(x_lin, y_lin, theta_dr, indexing='ij')
        _, _, Theta_hj = np.meshgrid(x_lin, y_lin, theta_hj, indexing='ij')
        
        spatial_coords_dr = np.stack([X, Y, Theta_dr], axis=-1)  # (res, res, res, 3)
        spatial_coords_hj = np.stack([X, Y, Theta_hj], axis=-1)
        
        # Time grid
        times = np.linspace(0, self.config.time_horizon, self.config.num_time_steps)
        
        # Flatten spatial grid (res^3 points)
        spatial_flat_dr = spatial_coords_dr.reshape(-1, 3)
        spatial_flat_hj = spatial_coords_hj.reshape(-1, 3)

        # Build time-stacked coordinate arrays
        coords_dr_list = []
        coords_hj_list = []
        for t in times:
            t_col = np.full((spatial_flat_dr.shape[0], 1), t)
            coords_dr_list.append(np.concatenate([t_col, spatial_flat_dr], axis=-1))
            coords_hj_list.append(spatial_flat_hj)

        coords_dr = np.concatenate(coords_dr_list, axis=0)
        coords_hj = np.concatenate(coords_hj_list, axis=0)

        return coords_dr, coords_hj, times
    
    def _boundary_fn_hj(self, x: np.ndarray) -> np.ndarray:
        """
        Compute boundary function for hj_reachability.
        
        Args:
            x: (N, 3) array of [x, y, theta]
        
        Returns:
            boundary: (N,) array of distance to goal.
        """
        return np.linalg.norm(x[..., :2], axis=-1) - self.config.goal_radius
    
    def compute_ground_truth(self) -> Tuple[np.ndarray, float]:
        """
        Solve BRT using hj_reachability PDE solver.
        
        Returns:
            values: (num_time_steps, *grid_shape) value function.
            solve_time: Wall-clock time for solve.
        """
        # Initial value: boundary function
        boundary_vals = self._boundary_fn_hj(self.hj_grid.states)
        
        # Time grid
        times = np.linspace(0, self.config.time_horizon, self.config.num_time_steps)
        times_negative = -times  # hj_reachability typically solves backward
        
        # Solver settings
        settings = hj_solver.SolverSettings.with_accuracy("high")
        settings = settings.replace(
            hamiltonian_postprocessor=hj_solver.backwards_reachable_tube,  # BRT postprocessor
        )
        
        # Solve
        start_time = time.time()
        try:
            values = hj_solver.solve(
                solver_settings=settings,
                dynamics=self.hj_system,
                grid=self.hj_grid,
                times=times_negative,  # Negative times for backward solve
                initial_values=boundary_vals,
                progress_bar=True,
            )
            # Reverse time axis since we solved backward
            values = np.array(values)[::-1]
        except Exception as e:
            print(f"Error during hj_reachability solve: {e}")
            raise
        solve_time = time.time() - start_time
        
        return np.array(values), solve_time
    
    def compute_deepreach_values(self, model: torch.nn.Module, coords_dr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate trained DeepReach model on evaluation grid.
        
        Args:
            model: Trained DeepReach neural network.
            coords_dr: (N, 4) array of [t, x, y, theta] in model units.
        
        Returns:
            values: (N,) array of predicted values.
            eval_time: Wall-clock time for evaluation.
        """
        coords_tensor = torch.from_numpy(coords_dr).float().to(self.device)
        
        # Convert to model input
        inputs = self.deepreach_dynamics.coord_to_input(coords_tensor)
        
        start_time = time.time()
        with torch.no_grad():
            model_out = model({'coords': inputs})
            output = model_out['model_out'][..., 0]
            values = self.deepreach_dynamics.io_to_value(inputs, output)
        eval_time = time.time() - start_time
        
        return values.detach().cpu().numpy(), eval_time
    
    def compute_mse(self, gt_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
        """
        Compute MSE between ground truth and predictions.
        
        Args:
            gt_values: (num_time, res, res, res) ground truth values.
            pred_values: (num_time * res^3,) predicted values.
        
        Returns:
            metric dict with overall MSE and per-time MSE.
        """
        # Reshape predictions to match ground truth shape
        pred_reshaped = pred_values.reshape(gt_values.shape)
        
        mse_overall = np.mean((gt_values - pred_reshaped) ** 2)
        mse_per_time = np.mean((gt_values - pred_reshaped) ** 2, axis=(1, 2, 3))
        
        return {
            'mse_overall': float(mse_overall),
            'mse_per_time': mse_per_time.tolist(),
        }
    
    def compute_brt_volume_error(self, gt_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
        """
        Compute BRT volume error (set indicator difference).
        
        Args:
            gt_values: (num_time, res, res, res) ground truth values.
            pred_values: (num_time * res^3,) predicted values.
        
        Returns:
            metric dict with volume errors.
        """
        pred_reshaped = pred_values.reshape(gt_values.shape)
        
        # BRT indicator: <= 0
        gt_mask = (gt_values <= 0).astype(float)
        pred_mask = (pred_reshaped <= 0).astype(float)
        
        # Voxel volume
        dx = (self.config.domain_bounds[0][1] - self.config.domain_bounds[0][0]) / self.config.resolution
        dy = (self.config.domain_bounds[1][1] - self.config.domain_bounds[1][0]) / self.config.resolution
        dtheta = (self.config.domain_bounds[2][1] - self.config.domain_bounds[2][0]) / self.config.resolution
        voxel_volume = dx * dy * dtheta
        
        vol_gt = np.sum(gt_mask) * voxel_volume
        vol_pred = np.sum(pred_mask) * voxel_volume
        
        abs_error = np.abs(vol_pred - vol_gt)
        rel_error = abs_error / vol_gt if vol_gt > 0 else 0.0
        
        return {
            'brt_volume_gt': float(vol_gt),
            'brt_volume_pred': float(vol_pred),
            'brt_volume_abs_error': float(abs_error),
            'brt_volume_rel_error': float(rel_error),
        }
    
    def run(self, model: torch.nn.Module, output_dir: Optional[Path] = None) -> Dict:
        """
        Run full comparison pipeline.
        
        Args:
            model: Trained DeepReach model.
            output_dir: Directory to save results JSON.
        
        Returns:
            results dict with all metrics.
        """
        print("=" * 80)
        print("DeepReach vs hj_reachability Comparison: Dubins3D")
        print("=" * 80)
        
        # Build grids
        print("\n[1/4] Building evaluation grids...")
        coords_dr, coords_hj, times = self.build_evaluation_grid()
        print(f"  Grid resolution: {self.config.resolution}^3")
        print(f"  Time steps: {self.config.num_time_steps}")
        print(f"  Total evaluation points: {coords_dr.shape[0]}")
        
        # Compute ground truth
        print("\n[2/4] Computing ground truth via hj_reachability...")
        gt_values, gt_time = self.compute_ground_truth()
        print(f"  Ground truth solve time: {gt_time:.2f}s")
        print(f"  Ground truth shape: {gt_values.shape}")
        
        # Evaluate DeepReach model
        print("\n[3/4] Evaluating DeepReach model...")
        pred_values, pred_time = self.compute_deepreach_values(model, coords_dr)
        print(f"  DeepReach evaluation time: {pred_time:.2f}s")
        print(f"  Predictions shape: {pred_values.shape}")
        
        # Compute metrics
        print("\n[4/4] Computing metrics...")
        mse_metrics = self.compute_mse(gt_values, pred_values)
        print(f"  Overall MSE: {mse_metrics['mse_overall']:.6e}")
        
        volume_metrics = self.compute_brt_volume_error(gt_values, pred_values)
        print(f"  Ground truth BRT volume: {volume_metrics['brt_volume_gt']:.6f}")
        print(f"  Predicted BRT volume: {volume_metrics['brt_volume_pred']:.6f}")
        print(f"  Relative volume error: {volume_metrics['brt_volume_rel_error']:.4%}")
        
        # Compile results
        self.results = {
            'config': {
                'goal_radius': self.config.goal_radius,
                'velocity': self.config.velocity,
                'omega_max': self.config.omega_max,
                'resolution': self.config.resolution,
                'time_horizon': self.config.time_horizon,
                'num_time_steps': self.config.num_time_steps,
                'set_mode': self.config.set_mode,
            },
            'timing': {
                'ground_truth_solve_time': float(gt_time),
                'deepreach_eval_time': float(pred_time),
                'speedup': float(gt_time / pred_time),
            },
            'metrics': {
                **mse_metrics,
                **volume_metrics,
            },
        }
        
        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "comparison_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n  Results saved to {results_file}")
        
        print("\n" + "=" * 80)
        return self.results


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare DeepReach vs hj_reachability")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained DeepReach model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./comparison_results", help="Output directory for results")
    parser.add_argument("--resolution", type=int, default=40, help="Grid resolution per dimension")
    parser.add_argument("--time-horizon", type=float, default=0.5, help="Time horizon for solve")
    parser.add_argument("--device", type=str, default="cpu", help="Device for DeepReach eval (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Load model checkpoint (state dict expected)
    print(f"Loading model checkpoint from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=args.device)
    state_dict = None
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise RuntimeError('Checkpoint does not contain a model state dict.')

    # Build the network architecture used by DeepReach
    from utils.modules import SingleBVPNet
    model = SingleBVPNet(in_features=4, out_features=1, type='sine', mode='mlp', hidden_features=512, num_hidden_layers=3)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    # Setup config
    config = Air3DComparisonConfig(
        collision_radius=0.25, # Matches your Air3D dynamics.collisionR
        velocity=0.5,          # Matches your Air3D lr5 speed
        omega_max=1.0,         # Matches your Air3D max turn rate
        angle_alpha_factor=1.0,
        resolution=args.resolution,
        time_horizon=args.time_horizon,
        num_time_steps=11,
    )
    
    # Run comparison
    runner = Air3DComparisonRunner(config, device=args.device)
    results = runner.run(model, output_dir=args.output_dir)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()