"""Evaluate Planck's law for every temperature at once, then read off Wien's peak and the total energy."""

import torch

from .equation import approximate_color, peak_wavelength, planck, total_energy


def solve(T: torch.Tensor, lam: torch.Tensor) -> dict[str, torch.Tensor]:
    """T: (A,), lam: (L,).  Returns the spectra (A, L) and per-temperature results (A,)."""
    u = planck(lam, T)  # (A, L)
    return {
        "u": u,
        "lam_max": peak_wavelength(lam, u),  # (A,)
        "U": total_energy(lam, u),  # (A,)
        "rgb": approximate_color(T),  # (A, 3)
    }
