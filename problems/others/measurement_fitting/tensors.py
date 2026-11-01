"""Build the concrete tensors: load the measurements and make a starting guess for the parameters.

measurements.csv (next to this file) holds a synthetic lab run: time t_s, position x_mm, error bar sigma_mm.
float64 is used throughout: the Hessian and (JᵀJ)⁻¹ are more trustworthy in double precision.
"""

import math
from pathlib import Path

import torch

from . import equation

DATA = Path(__file__).parent / "measurements.csv"
DTYPE = torch.float64


def load_measurements(device: torch.device, path: Path = DATA) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the CSV with plain Python (lines starting with # are comments): t, x, σ, each of shape (N,)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("t_s"):  # the header row: t_s,x_mm,sigma_mm
                continue
            rows.append([float(v) for v in line.split(",")])
    data = torch.tensor(rows, dtype=DTYPE, device=device)  # (N, 3)
    return data[:, 0], data[:, 1], data[:, 2]


def initial_parameters(t, x, sigma, omega0: float, gamma0: float) -> torch.Tensor:
    """p0 = (A, γ, ω, φ, c, ln s), shape (6,).

    Only γ0 and ω0 are guessed; for those values the model is linear in (A cos φ, A sin φ, c), so we get
    A, φ and c from one linear least-squares solve. The jitter starts small, s = 0.1 mm.
    """
    beta = equation.linear_fit(t, x, sigma, gamma0, omega0)
    A, phi = equation.amplitude_phase(beta)
    return torch.tensor(
        [A.item(), gamma0, omega0, phi.item(), beta[2].item(), math.log(0.1)], dtype=DTYPE, device=t.device
    )
