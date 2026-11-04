"""Build the concrete tensors: the z = M/T clock and the (ε, K) grid of models solved together."""

import torch

ETA_OBSERVED = 6.12e-10  # baryons per photon: Planck 2018 Ω_b h² = 0.0224 (η = 273.9e-10 Ω_b h²), BBN agrees
Z_START = 0.01  # start of the integration, T = 100 M: the heavy neutrinos are relativistic


def z_grid(z_max: float, n: int, device: torch.device | None = None) -> torch.Tensor:
    """Log-spaced z = M/T from Z_START to z_max (n points): (n,). Equal steps in ln z suit rates that vary as powers."""
    return torch.logspace(torch.log10(torch.tensor(Z_START)), torch.log10(torch.tensor(z_max)), n,
                          dtype=torch.float64, device=device)  # fmt: skip


def parameter_grid(eps_range, K_range, n_eps: int, n_K: int, device: torch.device | None = None):
    """ε as a column (E, 1) and K as a row (1, K), log-spaced, so every rate broadcasts to the (E, K) grid."""
    eps = torch.logspace(*eps_range, n_eps, dtype=torch.float64, device=device)[:, None]
    K = torch.logspace(*K_range, n_K, dtype=torch.float64, device=device)[None, :]
    return eps, K
