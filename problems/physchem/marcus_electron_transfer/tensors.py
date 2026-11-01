"""Build the concrete tensors: driving forces, reorganisation energies, temperatures and the starting ensemble.

Units: energies in eV, temperatures in K, time in ps. The solvent coordinate x is dimensionless: the reactant
parabola has its minimum at x = 0 and the product parabola at x = 1.
"""

import torch

KB = 8.617333e-5  # Boltzmann constant, eV/K
HBAR = 6.582120e-4  # reduced Planck constant, eV·ps


def driving_forces(dG_min: float, dG_max: float, n: int, device: torch.device) -> torch.Tensor:
    """ΔG°_g, shape (G,): reaction free energies for the rate curves (negative = downhill)."""
    return torch.linspace(dG_min, dG_max, n, device=device)


def reorganisation_energies(lam: float, device: torch.device) -> torch.Tensor:
    """λ_l, shape (L,): half, once and twice the chosen reorganisation energy."""
    return lam * torch.tensor([0.5, 1.0, 2.0], device=device)


def temperatures(T: float, device: torch.device) -> torch.Tensor:
    """T_t, shape (3,): the chosen temperature and ±100 K around it."""
    return torch.tensor([T - 100.0, T, T + 100.0], device=device).clamp_min(50.0)


def simulated_driving_forces(lam: float, n: int, device: torch.device) -> torch.Tensor:
    """ΔG°_d, shape (D,): from 0 (normal region) through -λ (top of the curve) to -2λ (deep in the inverted region)."""
    return torch.linspace(0.0, -2 * lam, n, device=device)


def thermal_ensemble(n_dG: int, n_traj: int, lam: float, kT: float, omega: float, device: torch.device, seed: int = 0):
    """x, v of shape (D, N): every trajectory starts on the reactant surface, drawn from its Boltzmann distribution.

    G_R = λx² is a harmonic well, so ⟨x²⟩ = k_BT / 2λ and, with frequency ω, ⟨v²⟩ = ω² k_BT / 2λ.
    """
    gen = torch.Generator().manual_seed(seed)
    sx = (kT / (2 * lam)) ** 0.5
    x = sx * torch.randn(n_dG, n_traj, generator=gen)
    v = omega * sx * torch.randn(n_dG, n_traj, generator=gen)
    return x.to(device), v.to(device)
