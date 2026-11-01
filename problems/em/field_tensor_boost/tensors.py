"""Build the concrete tensors: metric, Levi-Civita symbols, fields and boost matrices."""

import itertools

import torch

DTYPE = torch.float64


def permutation_sign(p: tuple[int, ...]) -> float:
    """+1 for an even permutation, −1 for an odd one (count the pairs that are out of order)."""
    inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
    return -1.0 if inversions % 2 else 1.0


def levi_civita(dim: int) -> torch.Tensor:
    """ε with dim indices, shape (dim,) * dim: ε_{ijk} for dim = 3, ε^{μνρσ} (ε^{0123} = +1) for dim = 4."""
    eps = torch.zeros((dim,) * dim, dtype=DTYPE)
    for p in itertools.permutations(range(dim)):
        eps[p] = permutation_sign(p)
    return eps


def metric() -> torch.Tensor:
    """η_{μν} = diag(+1, −1, −1, −1) of shape (4, 4)."""
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=DTYPE))


def fields(params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """E_i and B_i from the parameters, each of shape (3,)."""
    E = torch.tensor([params["Ex"], params["Ey"], params["Ez"]], dtype=DTYPE)
    B = torch.tensor([params["Bx"], params["By"], params["Bz"]], dtype=DTYPE)
    return E, B


def boost(beta: torch.Tensor) -> torch.Tensor:
    """Λ^μ_ν for a boost with speed β along x, for each β: (S,) → (S, 4, 4).

        ⎡  γ   -γβ  0  0 ⎤
    Λ = ⎢ -γβ   γ   0  0 ⎥        γ = 1 / √(1 − β²)
        ⎢  0    0   1  0 ⎥
        ⎣  0    0   0  1 ⎦
    """
    gamma = 1.0 / torch.sqrt(1.0 - beta**2)  # (S,)
    Lam = torch.eye(4, dtype=DTYPE).repeat(len(beta), 1, 1)  # (S, 4, 4)
    Lam[:, 0, 0] = gamma
    Lam[:, 1, 1] = gamma
    Lam[:, 0, 1] = -gamma * beta
    Lam[:, 1, 0] = -gamma * beta
    return Lam
