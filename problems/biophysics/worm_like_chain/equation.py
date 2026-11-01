r"""Physics → tensor translation for a stretched DNA molecule.

Continuous laws (z = extension / contour length, P = persistence length, b = Kuhn length, k_B T = thermal energy):

    worm-like chain (Marko–Siggia):   F P / k_B T = z + 1 / (4 (1 - z)²) - 1/4
    freely jointed chain:             z = coth(F b / k_B T) - k_B T / (F b),      b = 2 P

Discrete worm-like chain: N unit tangents t_{bmik} (force b, chain m, segment i, component k) of length a.
In units of k_B T the energy is bending plus the work done by the force along z:

    βE_{bm} = -K t_{bmik} t_{bm,i+1,k} - f_b t_{bmiz}         summed over i and k;   K = κ / k_B T,  f_b = F_b a / k_B T

Turning one tangent t_i → t'_i changes the energy through its local field h only:

    h_{bmik} = K (t_{bm,i-1,k} + t_{bm,i+1,k}) + f_b δ_{kz}
    ΔβE_{bmi} = -h_{bmik} (t'_{bmik} - t_{bmik})          (k summed)

Neighbouring tangents are correlated as ⟨t_i·t_{i+n}⟩ = (coth K - 1/K)ⁿ = exp(-n a / P).
"""

import math

import torch

KT = 4.11  # thermal energy k_B T at room temperature, pN·nm


def marko_siggia_force(z: torch.Tensor, P: float) -> torch.Tensor:
    """F(z) = k_B T / P · (z + 1/(4(1-z)²) - 1/4) in pN, for relative extension 0 ≤ z < 1."""
    return KT / P * (z + 0.25 / (1 - z) ** 2 - 0.25)


def marko_siggia_extension(F: torch.Tensor, P: float, iterations: int = 60) -> torch.Tensor:
    """Invert Marko–Siggia for z(F) by bisection (the force grows monotonically with z), elementwise on any shape."""
    lo, hi = torch.zeros_like(F), torch.ones_like(F)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        too_far = marko_siggia_force(mid, P) > F
        hi = torch.where(too_far, mid, hi)
        lo = torch.where(too_far, lo, mid)
    return 0.5 * (lo + hi)


def fjc_extension(F: torch.Tensor, b: float) -> torch.Tensor:
    """Freely jointed chain: z = L(F b / k_B T) with the Langevin function L(x) = coth x - 1/x."""
    x = F * b / KT
    return 1 / torch.tanh(x) - 1 / x


def bending_energy(t: torch.Tensor, K: float, f: torch.Tensor) -> torch.Tensor:
    """βE_bm = -K t_{bmik} t_{bm,i+1,k} - f_b t_{bmiz}, repeated i and k summed.   t: (B, M, N, 3), f: (B,) → (B, M)."""
    bend = torch.einsum("bmik,bmik->bm", t[:, :, :-1], t[:, :, 1:])  # Σ_i t_i·t_{i+1}
    pull = t[..., 2].sum(-1)  # Σ_i t_iz
    return -K * bend - f[:, None] * pull


def local_field(t: torch.Tensor, K: float, f: torch.Tensor) -> torch.Tensor:
    """h_{bmik} = K (t_{i-1} + t_{i+1})_k + f_b δ_{kz}; the two free ends have only one neighbour.   (B, M, N, 3)."""
    pad = torch.nn.functional.pad(t, (0, 0, 1, 1))  # zero tangents beyond both ends: (B, M, N+2, 3)
    h = K * (pad[:, :, :-2] + pad[:, :, 2:])
    h[..., 2] += f[:, None, None]
    return h


def local_energy_change(h: torch.Tensor, t: torch.Tensor, t_new: torch.Tensor) -> torch.Tensor:
    """ΔβE_{bmi} = -h_{bmik} (t'_{bmik} - t_{bmik}), repeated k summed.   (B, M, N)."""
    return -torch.einsum("bmik,bmik->bmi", h, t_new - t)


def relative_extension(t: torch.Tensor) -> torch.Tensor:
    """z / L = (1/N) Σ_i t_{bmiz}: end-to-end distance along the force over the contour length.   (B, M)."""
    return t[..., 2].mean(-1)


def tangent_correlation(t: torch.Tensor, max_lag: int) -> torch.Tensor:
    """⟨t_i · t_{i+n}⟩ averaged over i and all chains in t (..., N, 3), for n = 0 … max_lag.   (max_lag+1,)."""
    n = t.shape[-2]
    return torch.stack(
        [torch.einsum("...ik,...ik->...i", t[..., : n - s, :], t[..., s:, :]).mean() for s in range(max_lag + 1)]
    )


def segment_correlation(K: float) -> float:
    """⟨t_i · t_{i+1}⟩ = coth K - 1/K for a discrete chain with bending stiffness K (the Langevin function)."""
    return 1 / math.tanh(K) - 1 / K


def persistence_from_stiffness(K: float, a: float) -> float:
    """P = -a / ln(coth K - 1/K), from ⟨t_i·t_{i+n}⟩ = exp(-n a / P)."""
    return -a / math.log(segment_correlation(K))


def persistence_from_correlation(c: torch.Tensor, a: float) -> float:
    """Least-squares slope of ln⟨t_i·t_{i+n}⟩ = -n a / P through the origin:  P = -a Σ n² / Σ n ln c_n.

    c: (n_max+1,) correlations for lags n = 0 … n_max.
    """
    n = torch.arange(len(c), device=c.device, dtype=c.dtype)
    return (-a * (n**2).sum() / (n * torch.log(c)).sum()).item()


def persistence_from_force(F: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Solve Marko–Siggia for P at each measured point: P_b = k_B T (z + 1/(4(1-z)²) - 1/4) / F_b.   (B,)"""
    return KT * (z + 0.25 / (1 - z) ** 2 - 0.25) / F
