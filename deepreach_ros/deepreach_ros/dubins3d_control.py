import numpy as np


def dubins3d_optimal_control_from_grad_theta(
    dV_dtheta: float,
    omega_max: float,
    set_mode: str = 'avoid',
) -> float:
    sign = np.sign(dV_dtheta)
    if set_mode == 'reach':
        return float(-omega_max * sign)
    return float(omega_max * sign)
