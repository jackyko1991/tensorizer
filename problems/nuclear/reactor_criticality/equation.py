r"""Physics → tensor translation.

One-group neutron diffusion in a bare slab of width L: neutrons diffuse (D), are absorbed (Σ_a) and fission
produces ν new neutrons per fission (νΣ_f). A steady state exists only if the production is scaled by 1/k:

    -D \nabla^2 φ + Σ_a φ = \frac{1}{k} νΣ_f φ,        φ = 0 at x = 0 and x = L (extrapolated edges).

k_eff > 1: supercritical, = 1: critical, < 1: subcritical. On a grid x_i with the second-difference matrix
Δ_{ij} this becomes a generalised eigenproblem with a loss matrix M and a production matrix F:

    M_{ij} φ_j = \frac{1}{k} F_{ij} φ_j,        M_{ij} = -D Δ_{ij} + Σ_a δ_{ij},    F_{ij} = νΣ_f δ_{ij}.

Power iteration (what reactor codes do): solve M φ^{(n+1)} = F φ^{(n)} / k^{(n)}, then
k^{(n+1)} = k^{(n)} Σ_i (Fφ^{(n+1)})_i / Σ_i (Fφ^{(n)})_i.  The analytic answer for the bare slab is

    k = \frac{νΣ_f}{Σ_a + D B^2},    B = π / L,    φ(x) ∝ \sin(π x / L).
"""

import math

import torch


def loss_operator(lap: torch.Tensor, d: float, sigma_a: float) -> torch.Tensor:
    r"""M_{ij} = −D Δ_{ij} + Σ_a δ_{ij}  (leakage + absorption).   lap: (..., N, N) → (..., N, N)"""
    eye = torch.eye(lap.shape[-1], dtype=lap.dtype, device=lap.device)
    return -d * lap + sigma_a * eye


def production_operator(n: int, nu_sigma_f: float, device: torch.device) -> torch.Tensor:
    r"""F_{ij} = νΣ_f δ_{ij}  (fission source).   → (N, N)"""
    return nu_sigma_f * torch.eye(n, dtype=torch.float64, device=device)


def power_step(m: torch.Tensor, f: torch.Tensor, phi: torch.Tensor, k: float) -> tuple[torch.Tensor, float]:
    r"""φ' = M^{-1} F φ / k,   k' = k · Σ_i (Fφ')_i / Σ_i (Fφ)_i.   phi: (N,) → ((N,), float)"""
    source = torch.einsum("ij,j->i", f, phi)  # F_{ij} φ_j
    phi_new = torch.linalg.solve(m, source / k)  # M_{ij} φ'_j = F_{ij} φ_j / k
    k_new = k * torch.einsum("ij,j->", f, phi_new).item() / source.sum().item()
    return phi_new, k_new


def analytic_k(width: torch.Tensor | float, d: float, sigma_a: float, nu_sigma_f: float):
    r"""k = νΣ_f / (Σ_a + D B²) with buckling B = π/L."""
    return nu_sigma_f / (sigma_a + d * (math.pi / width) ** 2)


def critical_width(d: float, sigma_a: float, nu_sigma_f: float) -> float:
    r"""k = 1  ⇔  B² = (νΣ_f − Σ_a)/D  ⇔  L_c = π \sqrt{D / (νΣ_f − Σ_a)}  (∞ if νΣ_f ≤ Σ_a)."""
    excess = nu_sigma_f - sigma_a
    return math.pi * math.sqrt(d / excess) if excess > 0 else math.inf
