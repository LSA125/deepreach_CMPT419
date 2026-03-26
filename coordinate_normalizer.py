"""
ROS Coordinate Normalization and Denormalization
==================================================

This module provides utilities to normalize real-world ROS coordinates into
the [-1, 1] range expected by the DeepReach neural network, and denormalize
network outputs back to real-world coordinates.

The normalization scheme allows the neural network to work with consistent
input ranges regardless of the physical system's coordinate system.

Usage:
    from coordinate_normalizer import CoordinateNormalizer
    
    # Initialize with your system parameters
    normalizer = CoordinateNormalizer(
        x_range=(-5.0, 5.0),      # Real-world x bounds
        y_range=(-5.0, 5.0),      # Real-world y bounds
        theta_range=(-np.pi, np.pi),  # Heading range
        angle_alpha_factor=1.2     # From Dubins3D config
    )
    
    # Normalize ROS coordinates
    normalized = normalizer.normalize_state([2.5, -1.3, 0.7])
    
    # Denormalize network output (gradients, values)
    real_gradient = normalizer.denormalize_gradient(network_grad)
"""

import numpy as np
import torch
from typing import Union, Tuple, List
from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """Configuration for coordinate normalization."""
    x_range: Tuple[float, float]           # (min, max) in real-world coordinates
    y_range: Tuple[float, float]           # (min, max) in real-world coordinates
    theta_range: Tuple[float, float] = (-np.pi, np.pi)  # Angle range
    angle_alpha_factor: float = 1.2        # From Dubins3D config (multiplies π)
    use_time_coordinate: bool = True       # Include normalized time in network input
    time_max: float = 1.0                  # Maximum time value (typically 1.0)


class CoordinateNormalizer:
    """
    Normalize/denormalize coordinates between ROS and network coordinate systems.
    
    The DeepReach neural network expects:
    - Input: [t, x_norm, y_norm, theta_norm] where all values are in [-1, 1]
    - Output: V_norm (normalized value function)
    
    Real-world coordinates:
    - Input: [x_real, y_real, theta_real] in physical units (meters, radians)
    - Output: dV/dx, dV/dy, dV/dθ in physical units
    """
    
    def __init__(self, config: NormalizationConfig = None, **kwargs):
        """
        Initialize normalizer.
        
        Args:
            config: NormalizationConfig object, or use kwargs
            **kwargs: Alternative to config (x_range, y_range, theta_range, etc.)
        """
        if config is None:
            # Build config from kwargs
            if not kwargs:
                raise ValueError("Must provide either config or kwargs (x_range, y_range, etc.)")
            config = NormalizationConfig(**kwargs)
        
        self.config = config
        
        # Extract and compute normalization factors
        self.x_min, self.x_max = config.x_range
        self.y_min, self.y_max = config.y_range
        self.theta_min, self.theta_max = config.theta_range
        
        # Center and scale for each coordinate
        self.x_center = (self.x_min + self.x_max) / 2
        self.y_center = (self.y_min + self.y_max) / 2
        self.theta_center = (self.theta_min + self.theta_max) / 2
        
        self.x_scale = (self.x_max - self.x_min) / 2
        self.y_scale = (self.y_max - self.y_min) / 2
        self.theta_scale = (self.theta_max - self.theta_min) / 2
        
        # Time normalization
        self.time_max = config.time_max
        
        # Angle normalization factor for network (from Dubins3D config)
        self.angle_alpha_factor = config.angle_alpha_factor
        self.theta_network_scale = self.angle_alpha_factor * np.pi
    
    def normalize_state(self, state: Union[np.ndarray, list, torch.Tensor],
                        time: float = 0.0) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert real-world state to network input format.
        
        Input (real-world):
            state = [x, y, theta]  or  state = [x, y] (theta=0)
            time = t ∈ [0, time_max]
        
        Output (network format):
            [t_norm, x_norm, y_norm, theta_norm]  where all ∈ [-1, 1]
        
        Args:
            state: Real-world state [x, y, theta]
            time: Time coordinate (default 0.0)
        
        Returns:
            Normalized state for network input
        """
        is_torch = isinstance(state, torch.Tensor)
        
        if is_torch:
            device = state.device
            state_np = state.cpu().numpy() if state.is_cuda else state.numpy()
        else:
            state_np = np.asarray(state, dtype=np.float32)
        
        # Normalize input shape to (N, D)
        original_shape = state_np.shape
        if state_np.ndim == 1:
            if original_shape[0] not in (2, 3):
                raise ValueError(f"Expected state with 2 or 3 components, got shape {original_shape}")
            state_np = state_np[None, :]
            batch_shape = None
        elif state_np.ndim > 1:
            batch_shape = state_np.shape[:-1]
            state_np = state_np.reshape(-1, original_shape[-1])
        else:
            batch_shape = None
        
        n_samples = state_np.shape[0]
        
        # Normalize time
        time_norm = np.full(n_samples, time / self.time_max, dtype=np.float32)
        
        # Normalize spatial coordinates
        x_norm = (state_np[:, 0] - self.x_center) / self.x_scale
        y_norm = (state_np[:, 1] - self.y_center) / self.y_scale
        
        # Normalize heading (special handling for angle wrapping)
        theta = state_np[:, 2] if state_np.shape[1] > 2 else np.zeros(n_samples)
        theta_wrapped = self._wrap_angle(theta)
        theta_norm = (theta_wrapped - self.theta_center) / self.theta_scale
        
        # Stack into network input format: [t, x, y, theta]
        normalized = np.stack([time_norm, x_norm, y_norm, theta_norm], axis=-1)
        
        # Restore shape
        if original_shape and len(original_shape) == 1:
            normalized = normalized[0]
        elif batch_shape is not None:
            normalized = normalized.reshape(*batch_shape, 4)
        
        # Convert back to torch if input was torch
        if is_torch:
            normalized = torch.from_numpy(normalized).to(device)
        
        return normalized
    
    def denormalize_state(self, normalized: Union[np.ndarray, torch.Tensor],
                         include_time: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert network coordinates back to real-world state format.
        
        Input (network format):
            [t_norm, x_norm, y_norm, theta_norm]  where all ∈ [-1, 1]
        
        Output (real-world):
            [x, y, theta]  or  [t, x, y, theta] if include_time=True
        
        Args:
            normalized: Normalized state from network
            include_time: If True, include time in output
        
        Returns:
            Real-world state
        """
        is_torch = isinstance(normalized, torch.Tensor)
        
        if is_torch:
            device = normalized.device
            norm_np = normalized.cpu().numpy() if normalized.is_cuda else normalized.numpy()
        else:
            norm_np = np.asarray(normalized, dtype=np.float32)
        
        # Handle different input formats
        if norm_np.shape[-1] == 4:
            # Format: [t_norm, x_norm, y_norm, theta_norm]
            t_norm = norm_np[..., 0]
            x_norm = norm_np[..., 1]
            y_norm = norm_np[..., 2]
            theta_norm = norm_np[..., 3]
        elif norm_np.shape[-1] == 3:
            # Format: [x_norm, y_norm, theta_norm]
            t_norm = None
            x_norm = norm_np[..., 0]
            y_norm = norm_np[..., 1]
            theta_norm = norm_np[..., 2]
        else:
            raise ValueError(f"Expected normalized state with 3 or 4 coordinates, got {norm_np.shape[-1]}")
        
        # Denormalize coordinates
        x = x_norm * self.x_scale + self.x_center
        y = y_norm * self.y_scale + self.y_center
        theta = theta_norm * self.theta_scale + self.theta_center
        
        # Wrap theta to [-π, π]
        theta = self._wrap_angle(theta)
        
        # Build output
        if include_time and t_norm is not None:
            denormalized = np.stack([t_norm * self.time_max, x, y, theta], axis=-1)
        else:
            denormalized = np.stack([x, y, theta], axis=-1)
        
        # Convert back to torch if input was torch
        if is_torch:
            denormalized = torch.from_numpy(denormalized).to(device)
        
        return denormalized
    
    def denormalize_gradient(self, gradient: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert network gradient to real-world gradient via coordinate change.
        
        The chain rule: if z = f(x) and x = a + b·x_norm, then
        df/dx = (1/b) · df/dx_norm
        
        For spatial coordinates:
            ∂V/∂x = (1/x_scale) · (∂V/∂x_norm)
            ∂V/∂y = (1/y_scale) · (∂V/∂y_norm)
            ∂V/∂θ = (1/theta_scale) · (∂V/∂θ_norm)
        
        Args:
            gradient: Network gradient [∂V/∂x_norm, ∂V/∂y_norm, ∂V/∂theta_norm]
        
        Returns:
            Real-world gradient [∂V/∂x, ∂V/∂y, ∂V/∂theta]
        """
        is_torch = isinstance(gradient, torch.Tensor)
        
        if is_torch:
            device = gradient.device
            grad_np = gradient.cpu().numpy() if gradient.is_cuda else gradient.numpy()
        else:
            grad_np = np.asarray(gradient, dtype=np.float32)
        
        # Apply scaling to convert normalized gradients to real-world coordinates
        if grad_np.shape[-1] == 3:
            # Format: [∂V/∂x_norm, ∂V/∂y_norm, ∂V/∂theta_norm]
            denormalized_grad = np.zeros_like(grad_np)
            denormalized_grad[..., 0] = grad_np[..., 0] / self.x_scale
            denormalized_grad[..., 1] = grad_np[..., 1] / self.y_scale
            denormalized_grad[..., 2] = grad_np[..., 2] / self.theta_scale
        else:
            raise ValueError(f"Expected gradient with 3 components, got {grad_np.shape[-1]}")
        
        # Convert back to torch if input was torch
        if is_torch:
            denormalized_grad = torch.from_numpy(denormalized_grad).to(device)
        
        return denormalized_grad
    
    def normalize_gradient(self, gradient: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert real-world gradient to network gradient (inverse operation).
        
        ∂V/∂x_norm = x_scale · ∂V/∂x
        ∂V/∂y_norm = y_scale · ∂V/∂y
        ∂V/∂θ_norm = theta_scale · ∂V/∂θ
        
        Args:
            gradient: Real-world gradient [∂V/∂x, ∂V/∂y, ∂V/∂θ]
        
        Returns:
            Network gradient [∂V/∂x_norm, ∂V/∂y_norm, ∂V/∂θ_norm]
        """
        is_torch = isinstance(gradient, torch.Tensor)
        
        if is_torch:
            device = gradient.device
            grad_np = gradient.cpu().numpy() if gradient.is_cuda else gradient.numpy()
        else:
            grad_np = np.asarray(gradient, dtype=np.float32)
        
        # Apply scaling to convert real-world gradients to normalized coordinates
        if grad_np.shape[-1] == 3:
            normalized_grad = np.zeros_like(grad_np)
            normalized_grad[..., 0] = grad_np[..., 0] * self.x_scale
            normalized_grad[..., 1] = grad_np[..., 1] * self.y_scale
            normalized_grad[..., 2] = grad_np[..., 2] * self.theta_scale
        else:
            raise ValueError(f"Expected gradient with 3 components, got {grad_np.shape[-1]}")
        
        # Convert back to torch if input was torch
        if is_torch:
            normalized_grad = torch.from_numpy(normalized_grad).to(device)
        
        return normalized_grad
    
    def _wrap_angle(self, theta: np.ndarray) -> np.ndarray:
        """Wrap angle to [-π, π] range."""
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    def print_config(self):
        """Print normalization configuration for debugging."""
        print("\n" + "="*70)
        print("Coordinate Normalization Configuration")
        print("="*70)
        print(f"\nReal-World Coordinate Ranges:")
        print(f"  x:     [{self.x_min:8.3f}, {self.x_max:8.3f}] → center={self.x_center:8.3f}, scale={self.x_scale:8.3f}")
        print(f"  y:     [{self.y_min:8.3f}, {self.y_max:8.3f}] → center={self.y_center:8.3f}, scale={self.y_scale:8.3f}")
        print(f"  θ:     [{self.theta_min:8.3f}, {self.theta_max:8.3f}] → center={self.theta_center:8.3f}, scale={self.theta_scale:8.3f}")
        print(f"\nNetwork Coordinate Ranges:")
        print(f"  t_norm:       [-1.0, 1.0]")
        print(f"  x_norm:       [-1.0, 1.0]")
        print(f"  y_norm:       [-1.0, 1.0]")
        print(f"  theta_norm:   [-1.0, 1.0]")
        print(f"\nScaling Factors:")
        print(f"  angle_alpha_factor: {self.angle_alpha_factor}")
        print(f"  max_time: {self.time_max}")
        print("="*70 + "\n")


# Convenience function for quick initialization
def create_normalizer_from_bounds(x_range, y_range, theta_range=(-np.pi, np.pi),
                                 angle_alpha_factor=1.2):
    """
    Quick way to create a normalizer from coordinate bounds.
    
    Args:
        x_range: (x_min, x_max) in real-world coordinates
        y_range: (y_min, y_max) in real-world coordinates
        theta_range: (theta_min, theta_max) in radians
        angle_alpha_factor: Angle normalization factor from Dubins3D
    
    Returns:
        CoordinateNormalizer instance
    """
    return CoordinateNormalizer(
        x_range=x_range,
        y_range=y_range,
        theta_range=theta_range,
        angle_alpha_factor=angle_alpha_factor
    )


if __name__ == '__main__':
    # Example usage
    print("\nExample: Dubins3D Coordinate Normalization")
    print("="*70)
    
    # Create normalizer for a typical workspace
    normalizer = create_normalizer_from_bounds(
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        theta_range=(-np.pi, np.pi),
        angle_alpha_factor=1.2
    )
    
    normalizer.print_config()
    
    # Example real-world state
    real_state = np.array([2.5, -1.3, 0.7])
    print("\nReal-world state (ROS coordinates):")
    print(f"  x = {real_state[0]:.3f} m")
    print(f"  y = {real_state[1]:.3f} m")
    print(f"  θ = {real_state[2]:.3f} rad = {np.degrees(real_state[2]):.1f}°")
    
    # Normalize
    normalized = normalizer.normalize_state(real_state, time=0.5)
    print("\nNormalized state (network input):")
    print(f"  t_norm    = {normalized[0]:.3f}")
    print(f"  x_norm    = {normalized[1]:.3f}")
    print(f"  y_norm    = {normalized[2]:.3f}")
    print(f"  θ_norm    = {normalized[3]:.3f}")
    
    # Denormalize back
    denormalized = normalizer.denormalize_state(normalized)
    print("\nDenormalized (should match original):")
    print(f"  x = {denormalized[0]:.3f} m")
    print(f"  y = {denormalized[1]:.3f} m")
    print(f"  θ = {denormalized[2]:.3f} rad = {np.degrees(denormalized[2]):.1f}°")
    
    # Gradient example
    network_grad = np.array([0.5, -0.2, 0.1])  # Network gradients in normalized coords
    print("\nNetwork gradient (normalized coords):")
    print(f"  ∂V/∂x_norm = {network_grad[0]:.3f}")
    print(f"  ∂V/∂y_norm = {network_grad[1]:.3f}")
    print(f"  ∂V/∂θ_norm = {network_grad[2]:.3f}")
    
    real_grad = normalizer.denormalize_gradient(network_grad)
    print("\nReal-world gradient:")
    print(f"  ∂V/∂x = {real_grad[0]:.5f}")
    print(f"  ∂V/∂y = {real_grad[1]:.5f}")
    print(f"  ∂V/∂θ = {real_grad[2]:.5f}")
    
    print("\n" + "="*70 + "\n")
