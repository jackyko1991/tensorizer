"""Build the concrete tensors: the bundled FIRAS spectrum and the batch of models to fit.

data/firas_monopole.csv is Table 4 of Fixsen et al. (1996), as served by NASA LAMBDA (public domain).
"""

from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data" / "firas_monopole.csv"
MODELS = ["T only", "T + μ", "T + y", "T + μ + y"]
# which of (T, μ, y) each model may vary: (B, 3)
FREE = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.float64)


def load_firas(device: torch.device) -> dict:
    """Wavenumber (cm⁻¹), intensity and 1σ uncertainty (MJy/sr), each (F,) with F = 43, in float64."""
    table = np.loadtxt(DATA, delimiter=",", comments="#", skiprows=4)  # (F, 5)
    col = {name: torch.tensor(table[:, k], dtype=torch.float64, device=device) for k, name in enumerate(
        ["nu", "intensity", "residual", "sigma", "galaxy"])}  # fmt: skip
    for name in ("residual", "sigma", "galaxy"):
        col[name] = col[name] * 1e-3  # kJy/sr → MJy/sr
    return col


def initial_params(T_init: float, device: torch.device) -> torch.Tensor:
    """One row (T, μ, y) per model, all starting from the same guess with no distortion: (B, 3)."""
    theta = torch.zeros(len(MODELS), 3, dtype=torch.float64, device=device)
    theta[:, 0] = T_init
    return theta
