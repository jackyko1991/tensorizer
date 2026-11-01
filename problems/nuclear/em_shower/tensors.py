"""Build the concrete tensors: the primary particle and the depth axis."""

import torch

from .equation import ELECTRON


def primary(e0: float, device: torch.device) -> dict[str, torch.Tensor]:
    """One electron of energy E0 (MeV) entering at depth 0 on the axis. Each entry has shape (P,) = (1,).

    energy E_p, kind s_p, lateral position x_p and angle a_p (in X0 per X0), the depth z0_p and lateral
    position x0_p where it was born (for drawing its track), and its birth step (for the fixed-length mode).
    """
    as_float = {"dtype": torch.float64, "device": device}
    return {
        "energy": torch.tensor([e0], **as_float),
        "kind": torch.tensor([ELECTRON], device=device),
        "x": torch.zeros(1, **as_float),
        "angle": torch.zeros(1, **as_float),
        "z0": torch.zeros(1, **as_float),
        "x0": torch.zeros(1, **as_float),
        "born": torch.zeros(1, dtype=torch.long, device=device),
    }


def depth_axis(t_end: float, steps_per_x0: int, device: torch.device) -> torch.Tensor:
    """t_k = k / steps_per_x0 radiation lengths: shape (T,)."""
    n = int(round(t_end * steps_per_x0))
    return torch.arange(n + 1, dtype=torch.float64, device=device) / steps_per_x0
