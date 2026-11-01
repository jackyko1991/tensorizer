"""Build the concrete tensors: the grid of initial hot spots (T₀, ρR₀) stacked into one batch."""

import torch


def initial_grid(n_t: int, n_r: int, t_range: tuple, rho_r_range: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-spaced starting temperatures T₀ (keV) and areal densities ρR₀ (g/cm²): (n_t,), (n_r,)."""
    t0 = torch.logspace(torch.log10(torch.tensor(t_range[0])), torch.log10(torch.tensor(t_range[1])), n_t)
    r0 = torch.logspace(torch.log10(torch.tensor(rho_r_range[0])), torch.log10(torch.tensor(rho_r_range[1])), n_r)
    return t0.double(), r0.double()


def initial_state(t0: torch.Tensor, r0: torch.Tensor) -> torch.Tensor:
    """y0 (B = n_t·n_r, 3): row b = (ln T₀, ln ρR₀, φ = 0) with b = i·n_r + j for T₀[i], ρR₀[j]."""
    T, R = torch.meshgrid(t0, r0, indexing="ij")
    return torch.stack([T.log().ravel(), R.log().ravel(), torch.zeros(T.numel(), dtype=torch.float64)], dim=1)
