import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class NormalizationConfig:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    theta_range: Tuple[float, float] = (-np.pi, np.pi)
    angle_alpha_factor: float = 1.2
    time_max: float = 1.0


class CoordinateNormalizer:
    def __init__(self, config: NormalizationConfig):
        self.config = config

        self.x_min, self.x_max = config.x_range
        self.y_min, self.y_max = config.y_range
        self.theta_min, self.theta_max = config.theta_range

        self.x_center = (self.x_min + self.x_max) / 2.0
        self.y_center = (self.y_min + self.y_max) / 2.0
        self.theta_center = (self.theta_min + self.theta_max) / 2.0

        self.x_scale = (self.x_max - self.x_min) / 2.0
        self.y_scale = (self.y_max - self.y_min) / 2.0
        self.theta_scale = (self.theta_max - self.theta_min) / 2.0

        self.time_max = config.time_max

    def normalize_state(self, state: Union[np.ndarray, torch.Tensor], time: float = 0.0):
        is_torch = isinstance(state, torch.Tensor)
        if is_torch:
            device = state.device
            state_np = state.detach().cpu().numpy()
        else:
            state_np = np.asarray(state, dtype=np.float32)

        if state_np.ndim == 1:
            state_np = state_np[None, :]
            squeeze = True
        else:
            squeeze = False

        n = state_np.shape[0]
        t_norm = np.full((n,), time / self.time_max, dtype=np.float32)
        x_norm = (state_np[:, 0] - self.x_center) / self.x_scale
        y_norm = (state_np[:, 1] - self.y_center) / self.y_scale

        theta = state_np[:, 2] if state_np.shape[1] > 2 else np.zeros((n,), dtype=np.float32)
        theta = self._wrap(theta)
        theta_norm = (theta - self.theta_center) / self.theta_scale

        out = np.stack([t_norm, x_norm, y_norm, theta_norm], axis=-1)
        if squeeze:
            out = out[0]

        if is_torch:
            return torch.from_numpy(out).to(device)
        return out

    def denormalize_gradient(self, grad: Union[np.ndarray, torch.Tensor]):
        is_torch = isinstance(grad, torch.Tensor)
        if is_torch:
            device = grad.device
            grad_np = grad.detach().cpu().numpy()
        else:
            grad_np = np.asarray(grad, dtype=np.float32)

        out = np.zeros_like(grad_np)
        out[..., 0] = grad_np[..., 0] / self.x_scale
        out[..., 1] = grad_np[..., 1] / self.y_scale
        out[..., 2] = grad_np[..., 2] / self.theta_scale

        if is_torch:
            return torch.from_numpy(out).to(device)
        return out

    @staticmethod
    def _wrap(theta: np.ndarray) -> np.ndarray:
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
