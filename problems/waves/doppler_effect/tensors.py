"""Build the concrete tensors: emission times, emission points and the two listeners."""

import torch

C_SOUND = 340.0  # m/s


def emission_times(f: float, t_max: float, device: torch.device) -> torch.Tensor:
    """t_e = 0, 1/f, 2/f, … < t_max: (E,)."""
    return torch.arange(0.0, t_max, 1.0 / f, device=device)


def source_position(t: torch.Tensor, v_s: float, x_start: float) -> torch.Tensor:
    """s_i(t) = (x_start + v_s t, 0) for every time in t: (…, 2)."""
    x = x_start + v_s * t
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


def observers(x_start: float, v_s: float, t_max: float, c: float, device: torch.device) -> torch.Tensor:
    """o_{oi} of shape (2, 2): one listener ahead of where the source ends, one behind where it starts."""
    margin = 0.15 * c * t_max
    return torch.tensor([[x_start + v_s * t_max + margin, 0.0], [x_start - margin, 0.0]], device=device)
