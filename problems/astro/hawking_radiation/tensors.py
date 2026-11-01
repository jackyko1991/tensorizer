"""Build the concrete tensors: a batch of black-hole masses and a logarithmic wavelength grid (SI units)."""

import torch

from .equation import M_SUN

EXAMPLES = {  # name → mass in kg, for the printed table
    "Sun": M_SUN,
    "Earth": 5.97e24,
    "Moon": 7.35e22,
    "Mount Everest": 8.1e14,
    "dying today (primordial)": 1.7e11,
}


def example_masses(M0: float, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """Names and masses (B,) in kg: a few famous ones plus the hole we animate."""
    names = [*EXAMPLES, "animated hole"]
    masses = torch.tensor([*EXAMPLES.values(), M0], dtype=torch.float64, device=device)
    return names, masses


def wavelengths(lam_min: float, lam_max: float, n: int, device: torch.device) -> torch.Tensor:
    """λ grid (L,) spaced evenly in log λ, in metres."""
    return torch.logspace(torch.log10(torch.tensor(lam_min)), torch.log10(torch.tensor(lam_max)), n,
                          dtype=torch.float64, device=device)  # fmt: skip
