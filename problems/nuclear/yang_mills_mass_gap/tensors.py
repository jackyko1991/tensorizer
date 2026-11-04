"""Build the concrete tensors: random group elements as unit quaternions, hot-start lattices and the checkerboard.

A link variable U_μ(x) ∈ SU(2) is stored as a unit quaternion (a₀, a₁, a₂, a₃), U = a₀ + i aₖσₖ with ½ Tr U = a₀.
U(1) is the subgroup (cos θ, 0, 0, sin θ): products stay in it, and ½ Tr U = cos θ gives the U(1) Wilson action,
so the same code runs both groups. A lattice is a tensor (B, D, L, …, L, 4): batch of couplings β_b, link direction μ,
D site coordinates, quaternion component.
"""

import math

import torch


def random_su2(shape: tuple, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Haar-random SU(2): a normalised 4D Gaussian is uniform on the 3-sphere S³ ≅ SU(2).   → (*shape, 4)"""
    q = torch.randn(*shape, 4, generator=gen, dtype=torch.float64).to(device)
    return q / q.norm(dim=-1, keepdim=True)


def random_u1(shape: tuple, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Haar-random U(1) phase e^{iθ} as the quaternion (cos θ, 0, 0, sin θ).   → (*shape, 4)"""
    theta = (2 * math.pi * torch.rand(*shape, generator=gen, dtype=torch.float64)).to(device)
    zero = torch.zeros_like(theta)
    return torch.stack([torch.cos(theta), zero, zero, torch.sin(theta)], dim=-1)


def random_element(shape: tuple, group: str, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return random_su2(shape, gen, device) if group == "SU(2)" else random_u1(shape, gen, device)


def hot_start(n_beta: int, dim: int, size: int, group: str, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Every link random (infinite temperature, β = 0): (B, D, L, …, L, 4)."""
    return random_element((n_beta, dim) + (size,) * dim, group, gen, device)


def random_gauge(shape: tuple, group: str, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """A gauge transformation g(x), one group element per site: (B, L, …, L, 4)."""
    return random_element(shape, group, gen, device)


def parity(dim: int, size: int, device: torch.device) -> torch.Tensor:
    """(x₁ + … + x_D) mod 2 for every site: (L, …, L). Links of one direction on one parity share no staple."""
    coords = torch.meshgrid(*[torch.arange(size, device=device)] * dim, indexing="ij")
    return sum(coords) % 2


def coupling_grid(beta: float, group: str) -> list[float]:
    """β values simulated side by side (one batch); the chosen β is added so it gets the long analysis."""
    top = 4.0 if group == "SU(2)" else 2.0
    grid = [round(top * k / 6, 3) for k in range(1, 7)]
    return sorted(set(grid + [beta]))
