"""Build the concrete tensors: particle masses, the parent four-momentum and the rest-frame decay products."""

import math

import torch

# decay: (label, M, m1, m2, name1, name2), masses in GeV
DECAYS = {
    "kaon": ("K⁰ → π⁺ π⁻", 0.497611, 0.139570, 0.139570, "π⁺", "π⁻"),
    "pion": ("π⁰ → γ γ", 0.134977, 0.0, 0.0, "γ", "γ"),
    "lambda": ("Λ → p π⁻", 1.115683, 0.938272, 0.139570, "p", "π⁻"),
}


def parent(big_m: float, p_lab: float, device: torch.device) -> torch.Tensor:
    """P^μ = (√(M² + p²), 0, 0, p): the parent flies along z in the lab: shape (4,)."""
    return torch.tensor([math.sqrt(big_m**2 + p_lab**2), 0.0, 0.0, p_lab], dtype=torch.float64, device=device)


def decay_angles(n: int, device: torch.device) -> torch.Tensor:
    """Rest-frame emission angles θ* of daughter 1 relative to the flight direction, 0 … π: shape (B,)."""
    return torch.linspace(0.0, math.pi, n, dtype=torch.float64, device=device)


def rest_frame_daughters(
    theta: torch.Tensor, p_star: float, big_m: float, m1: float, m2: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-to-back four-momenta in the parent's rest frame, in the x–z plane: two tensors of shape (B, 4).

    p1 = (E1,  p* sin θ, 0,  p* cos θ),   p2 = (E2, −p* sin θ, 0, −p* cos θ),   E_k = √(m_k² + p*²).
    """
    e1, e2 = math.sqrt(m1**2 + p_star**2), math.sqrt(m2**2 + p_star**2)
    n = torch.stack([torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=-1)  # (B, 3) unit vectors
    p1 = torch.cat([torch.full_like(theta, e1)[:, None], p_star * n], dim=-1)
    p2 = torch.cat([torch.full_like(theta, e2)[:, None], -p_star * n], dim=-1)
    return p1, p2
