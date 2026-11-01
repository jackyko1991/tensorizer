"""Build the concrete tensors: state points, an fcc starting lattice in a periodic box, Maxwell–Boltzmann velocities.

Reduced Lennard-Jones units: lengths in σ, energies in ε, mass m = 1, temperature T* = k_BT/ε, density ρ* = Nσ³/V,
time τ = σ √(m/ε). For argon σ = 3.4 Å, ε/k_B = 120 K and τ = 2.2 ps.
"""

import torch

NAMES = ("gas", "liquid", "solid")


def state_points(rho_liquid: float, T_liquid: float, device: torch.device):
    """ρ*_s, T*_s, and a hotter start temperature (S,) for each of the three state points, all run as one batch.

    gas: well above the critical temperature (T*_c ≈ 1.3) at low density; liquid: near the triple point;
    solid: compressed fcc crystal. The gas and liquid start hot so the lattice melts quickly.
    """
    rho = torch.tensor([0.05, rho_liquid, 1.10], device=device)
    T = torch.tensor([1.5, T_liquid, 0.60], device=device)
    T_start = torch.tensor([1.5, 1.5, 0.60], device=device)
    return rho, T, T_start


def fcc_lattice(n_cells: int, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x (S, N, 3) on an fcc lattice of n_cells³ cubic cells (N = 4 n_cells³) and box lengths L_s = (N/ρ_s)^{1/3}."""
    basis = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], device=rho.device)
    r = torch.arange(n_cells, dtype=torch.float32, device=rho.device)
    cells = torch.cartesian_prod(r, r, r)  # (n³, 3)
    unit = (cells[:, None, :] + basis[None, :, :] + 0.25).reshape(-1, 3) / n_cells  # (N, 3) in [0, 1)
    L = (unit.shape[0] / rho) ** (1 / 3)
    return unit[None] * L[:, None, None], L


def maxwell_velocities(shape: tuple, T: torch.Tensor, device: torch.device, seed: int = 0) -> torch.Tensor:
    """v (S, N, 3): Gaussian velocities with zero total momentum, scaled to exactly T_s."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(shape, generator=gen).to(device)
    v = v - v.mean(1, keepdim=True)
    n_dof = 3 * shape[1] - 3
    T_now = (v * v).sum((1, 2)) / n_dof
    return v * torch.sqrt(T / T_now)[:, None, None]
