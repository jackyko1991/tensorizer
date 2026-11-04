"""Concrete tensors: measurement directions, CHSH settings, the shared state Psi in C^D, the Bell-test data.

Psi is uniform on the unit sphere of C^D (a complex Gaussian vector divided by its norm). Its hidden coordinates give
the race energies (blocks of m) and, in the local model, a hidden direction n = (Re z1, Im z1, Re z2) / norm that is
uniform on the sphere whatever D is (step 1).
"""

import csv
import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"


def unit(phi: torch.Tensor) -> torch.Tensor:
    """(..., 3) unit vector at angle phi from the z axis in the x-z plane."""
    return torch.stack([torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)], -1)


def random_directions(B: int, generator, device) -> torch.Tensor:
    """(B, 3) uniform unit vectors."""
    v = torch.randn(B, 3, generator=generator, dtype=torch.float64)
    return torch.nn.functional.normalize(v, dim=-1).to(device)


def planar_pairs(theta: torch.Tensor, device):
    """a along z and b at angle theta from it: (B, 3) each."""
    theta = theta.to(device)
    return unit(torch.zeros_like(theta)), unit(theta)


def optimal_settings(device):
    """CHSH-optimal coplanar settings a = 0, a' = 90, b = 45, b' = 135 degrees, each (1, 3)."""
    deg = torch.tensor([0.0, 90.0, 45.0, 135.0], dtype=torch.float64, device=device) * math.pi / 180
    return tuple(unit(deg[i : i + 1]) for i in range(4))


def random_settings(B: int, generator, device):
    """Four independent batches (B, 3) of random directions a, a', b, b'."""
    return tuple(random_directions(B, generator, device) for _ in range(4))


def psi_hidden(T: int, D: int, generator, device) -> torch.Tensor:
    """(T, D) samples of the shared state Psi, uniform on the unit sphere of C^D."""
    x = torch.randn(T, D, 2, generator=generator, dtype=torch.float64)
    z = torch.complex(x[..., 0], x[..., 1])
    return (z / z.norm(dim=1, keepdim=True)).to(device)


def hidden_block_energies(T: int, K: int, m: int, D: int, generator, device) -> torch.Tensor:
    """(T, K) E_k = ||hidden block k||^2: Psi in C^D, 4 visible coordinates, then K blocks of m hidden ones.

    Dividing by ||Psi|| rescales every E_k by the same factor, so the race argmax_k w_k / E_k is that of
    E_k ~ Gamma(m) (step 1).
    """
    if D < 4 + K * m:
        raise ValueError("need D >= 4 + K m")
    h = psi_hidden(T, D, generator, device)[:, 4 : 4 + K * m]
    return (h.abs() ** 2).view(T, K, m).sum(-1)


def shared_directions(N: int, D: int, generator, device) -> torch.Tensor:
    """(N, 3) hidden direction n read from three real coordinates of Psi in C^D: uniform on S^2 for any D."""
    z = psi_hidden(N, D, generator, device)
    n = torch.stack([z[:, -1].real, z[:, -1].imag, z[:, -2].real], -1)
    return torch.nn.functional.normalize(n, dim=-1)


def bell_tests() -> list[dict]:
    """Cited loophole-free Bell tests: experiment, year, platform, inequality, S, sigma, p_value, separation, doi."""
    with open(DATA / "bell_tests.csv", encoding="utf-8") as f:
        return [row for row in csv.DictReader(r for r in f if not r.startswith("#"))]
