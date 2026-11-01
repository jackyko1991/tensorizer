r"""Physics → tensor translation: the Gō-model energy of a bead chain, and forces from autograd.

Indices: s system (temperature × replica), i bead, p pair (i_p, j_p) with j - i ≥ 3, a Cartesian axis.

    r_sp = √((x_{s i_p a} - x_{s j_p a})(x_{s i_p a} - x_{s j_p a}))      (a summed)
    E_s = Σ_i k_b (|x_{s,i+1} - x_{si}| - b_i)²                            bonds
        + Σ_i k_θ (θ_si - θ⁰_i)²                                           bond angles
        + Σ_p n_p ε [5 (r⁰_p / r_sp)¹² - 6 (r⁰_p / r_sp)¹⁰]                 native contacts (n_p = 1), min. -ε at r⁰_p
        + Σ_p (1 - n_p) ε (σ / r_sp)¹²                                     every other pair only repels
    F_sia = -∂E_s / ∂x_sia                                                 (torch.autograd)
    Q_s = (1 / M) Σ_p n_p [r_sp < 1.2 r⁰_p]                                fraction of native contacts
"""

import torch

K_BOND = 100.0  # ε / Å²
K_ANGLE = 20.0  # ε / rad²
EPS = 1.0  # ε, depth of one native contact
SIGMA_REP = 4.0  # Å, size of the non-native repulsion


def pair_distances(x: torch.Tensor) -> torch.Tensor:
    """Full distance matrix r_ij for coordinates (..., N, 3) → (..., N, N); the axis index a is summed."""
    diff = x[..., :, None, :] - x[..., None, :, :]
    return torch.sqrt(torch.einsum("...ija,...ija->...ij", diff, diff) + 1e-9)


def pair_list_distances(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """r_sp = |x_{s,i_p} - x_{s,j_p}| for the listed pairs only.   x (S, N, 3), i, j (P,) → (S, P).

    The a-sum has just 3 terms, so it is (diff * diff).sum(-1) rather than an einsum: this runs every time step,
    and a few larger tensor operations keep a small simulation fast on a CPU.
    """
    diff = x[:, i] - x[:, j]  # (S, P, 3)
    return torch.sqrt((diff * diff).sum(-1))


def bond_energy(x: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
    """Σ_i k_b (|x_{i+1} - x_i| - b_i)²  for x (S, N, 3).   (S,)"""
    length = (x[:, 1:] - x[:, :-1]).norm(dim=-1)  # (S, N-1)
    return K_BOND * ((length - b0) ** 2).sum(-1)


def angle_energy(x: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
    """Σ_i k_θ (θ_i - θ⁰_i)², θ_i the angle at bead i+1 between its two bonds.   (S,)"""
    b = x[:, 1:] - x[:, :-1]  # (S, N-1, 3) bond vectors
    cos = -(b[:, :-1] * b[:, 1:]).sum(-1) / (b[:, :-1].norm(dim=-1) * b[:, 1:].norm(dim=-1))  # a summed
    theta = torch.arccos(cos.clamp(-0.9999, 0.9999))
    return K_ANGLE * ((theta - theta0) ** 2).sum(-1)


def pair_energy(r: torch.Tensor, r0: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Native pairs: ε [5 (r⁰/r)¹² - 6 (r⁰/r)¹⁰], a 12–10 well with its minimum -ε exactly at r⁰.
    Other pairs: ε (σ/r)¹², pure repulsion.   r (S, P), r0 (P,), native (P,) as 0/1 floats → (S,)."""
    q2 = (r0 / r) ** 2
    q10 = q2**5
    well = EPS * (5 * q10 * q2 - 6 * q10)
    wall = EPS * ((SIGMA_REP / r) ** 2) ** 6
    return (native * well + (1 - native) * wall).sum(-1)


def total_energy(x: torch.Tensor, model: dict) -> torch.Tensor:
    """E_s for every system at once, x (S, N, 3) → (S,)."""
    r = pair_list_distances(x, model["i"], model["j"])  # (S, P)
    return bond_energy(x, model["b0"]) + angle_energy(x, model["theta0"]) + pair_energy(r, model["r0"], model["native"])


def forces(x: torch.Tensor, model: dict) -> torch.Tensor:
    """F_sia = -∂E_s/∂x_sia by automatic differentiation.   (S, N, 3)

    Summing E over systems is safe: system s's energy depends only on its own coordinates.
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(total_energy(x, model).sum(), x)
    return -grad


def fraction_native(x: torch.Tensor, model: dict) -> torch.Tensor:
    """Q_s: share of native contacts within 1.2 × their native distance.   (S,)"""
    r = pair_list_distances(x, model["i"], model["j"])
    formed = (r < 1.2 * model["r0"]).float() * model["native"]
    return formed.sum(-1) / model["native"].sum()
