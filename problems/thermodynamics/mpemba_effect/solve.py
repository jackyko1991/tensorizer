"""Diagonalise W once, then relax every initial temperature exactly; locate the strong Mpemba point."""

import torch

from . import tensors
from .equation import metropolis_rates, mode_amplitudes


def spectrum(W: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigenvalues λ_k (0 = λ_1 > λ_2 > …) with right r_ik and left ℓ_ik eigenvectors of W, columns k.

    W is not symmetric, but detailed balance makes S = Π^{-1/2} W Π^{1/2} symmetric (Π = diag π), so the
    stable torch.linalg.eigh works: S v_k = λ_k v_k gives r_k = Π^{1/2} v_k and ℓ_k = Π^{-1/2} v_k, with ℓ_j·r_k = δ_jk.
    """
    s = pi.sqrt()  # (N,)
    S = W * s[None, :] / s[:, None]  # (N, N)  S_ij = W_ij √π_j / √π_i
    lam, V = torch.linalg.eigh(0.5 * (S + S.T))  # ascending; symmetrise away round-off
    lam, V = lam.flip(0), V.flip(1)  # slowest first: λ_1 = 0, λ_2 < 0, …
    V = V * torch.sign(V[0])[None, :]  # sign convention: every eigenvector positive on site 0
    right = s[:, None] * V  # (N, N)  r_ik
    left = V / s[:, None]  # (N, N)  ℓ_ik
    return lam, right, left


def a2_curve(u: torch.Tensor, T: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    """a_2(T) for a whole batch of initial temperatures in one einsum: (B,) → (B,)."""
    return mode_amplitudes(tensors.boltzmann(u, T), left)[:, 1]


def mpemba_point(u: torch.Tensor, T_bath: float, left: torch.Tensor, n_grid: int = 400) -> float:
    """Temperature T* > T_b where a_2 crosses zero: scan a log grid (batched), then refine by bisection."""
    T = torch.logspace(0.0, 2.0, n_grid, dtype=u.dtype, device=u.device) * 1.01 * T_bath  # (G,)
    a2 = a2_curve(u, T, left)
    flips = (torch.sign(a2[1:]) != torch.sign(a2[:-1])).nonzero()
    if len(flips) == 0:
        return float("nan")
    lo, hi = T[flips[0, 0]].item(), T[flips[0, 0] + 1].item()
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        a_lo, a_mid = a2_curve(u, torch.tensor([lo, mid], dtype=u.dtype, device=u.device), left)
        lo, hi = (lo, mid) if torch.sign(a_lo) != torch.sign(a_mid) else (mid, hi)
    return 0.5 * (lo + hi)


def build(barrier: float, T_bath: float, device: torch.device) -> dict:
    """Potential, rate matrix, Boltzmann state and spectrum for one bath temperature."""
    u = tensors.potential(barrier, device)
    A = tensors.neighbours(len(u), device)
    W = metropolis_rates(u, A, T_bath)
    pi = tensors.boltzmann(u, torch.tensor([T_bath], dtype=u.dtype, device=device))[0]  # (N,)
    lam, right, left = spectrum(W, pi)
    return {"u": u, "W": W, "pi": pi, "lam": lam, "right": right, "left": left}
