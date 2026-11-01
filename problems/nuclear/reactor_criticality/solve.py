"""k_eff two ways: power iteration (the reactor-physics workhorse) and a direct eigenvalue solve."""

import torch

from problems._common.progress import steps

from .equation import loss_operator, power_step, production_operator
from .tensors import laplacian


def power_iteration(
    m: torch.Tensor, f: torch.Tensor, phi0: torch.Tensor, n_iter: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (k history (I+1,), flux history (I+1, N)); each flux is normalised to peak 1 for plotting."""
    phi, k = phi0, 1.0
    ks, fluxes = [k], [phi / phi.max()]
    for _ in steps(n_iter, "power iterations"):
        phi, k = power_step(m, f, phi, k)
        ks.append(k)
        fluxes.append(phi / phi.max())
    return torch.tensor(ks, dtype=torch.float64), torch.stack(fluxes)


def k_eigen(m: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Largest eigenvalue of M⁻¹F (the fundamental mode's k). Works batched: (..., N, N) → (...)."""
    return torch.linalg.eigvals(torch.linalg.solve(m, f)).real.max(dim=-1).values


def k_versus_width(widths: torch.Tensor, n: int, d: float, sigma_a: float, nu_sigma_f: float) -> torch.Tensor:
    """k_eff for a whole batch of slab widths at once: one (S, N, N) stack of loss matrices → (S,)."""
    h = widths / (n + 1)  # (S,) grid spacing of each slab
    m = loss_operator(laplacian(n, h, widths.device), d, sigma_a)  # (S, N, N)
    f = production_operator(n, nu_sigma_f, widths.device).expand_as(m)  # (S, N, N)
    return k_eigen(m, f)
