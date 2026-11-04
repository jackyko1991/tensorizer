"""Constants and initial states. Planck units: G = ħ = c = 1, so lengths in ℓ_P, times in t_P, densities in ρ_P.

The state of each universe in the batch is y_b = (V_b, b_b, φ_b): physical volume of a fixed comoving cell, the
connection variable b (conjugate to V, b ≈ γH classically) and the scalar field φ. Its momentum p_φ is conserved.
"""

import math

import torch

G = 1.0
GAMMA = 0.2375  # Barbero–Immirzi parameter (black-hole entropy counting)
LAMBDA = math.sqrt(4 * math.sqrt(3) * math.pi * GAMMA)  # λ² = Δ, the smallest non-zero area eigenvalue of LQG
RHO_C = 3 / (8 * math.pi * G * GAMMA**2 * LAMBDA**2)  # critical density ≈ 0.41 ρ_Planck


def initial_state(p_phi: torch.Tensor, lam: torch.Tensor, rho_start: float) -> torch.Tensor:
    """Contracting universes at density rho_start, on the constraint surface.   p_phi, lam: (B,) → y0 (B, 3)

    ρ = p_φ²/(2V²) fixes V; the constraint sin²(λb) = ρ/ρ_c fixes b, on the branch λb ∈ (π/2, π) where H < 0.
    """
    rho_c = 3 / (8 * math.pi * G * GAMMA**2 * lam**2)
    V0 = p_phi / math.sqrt(2 * rho_start)
    b0 = (math.pi - torch.asin(torch.sqrt(rho_start / rho_c))) / lam
    return torch.stack([V0, b0, torch.zeros_like(V0)], dim=-1)


def wdw_grid(x_min: float, x_max: float, phi_min: float, phi_max: float, nx: int, nphi: int, device: torch.device):
    """Grid of x = ln V and scalar-field clock φ for the Wheeler–DeWitt wave packet: (X,), (P,)."""
    x = torch.linspace(x_min, x_max, nx, dtype=torch.float64, device=device)
    phi = torch.linspace(phi_min, phi_max, nphi, dtype=torch.float64, device=device)
    return x, phi
