r"""Physics → tensor translation: effective loop quantum cosmology of a flat FRW universe with a massless scalar.

Classical GR (flat FRW, massless scalar φ with momentum p_φ, cell volume V = a³): the Hamiltonian constraint is

    C_GR = p_φ²/(2V) − 3 V b²/(8πGγ²) = 0,       H = b/γ   ⇒   H² = (8πG/3) ρ,   ρ = p_φ²/(2V²).

Loop quantum cosmology replaces the connection b by holonomies around a loop of the smallest LQG area λ² = Δ,
b → sin(λb)/λ (Ashtekar–Pawlowski–Singh 2006). The *effective* constraint (an approximation that tracks sharply
peaked quantum states, checked numerically by APS) is

    C = p_φ²/(2V) − 3 V sin²(λb) / (8πGγ²λ²) = 0.

Hamilton's equations with {b, V} = 4πGγ and lapse 1 (proper time t), for every batch member b at once:

    V̇ = −4πGγ ∂C/∂b = 3 V sin(λb) cos(λb) / (γλ)       ⇒ H = V̇/(3V) = sin(2λb)/(2γλ)
    ḃ =  4πGγ ∂C/∂V = 4πGγ [ −p_φ²/(2V²) − 3 sin²(λb)/(8πGγ²λ²) ]
    φ̇ = p_φ / V

On C = 0 this gives the effective Friedmann equation H² = (8πG/3) ρ (1 − ρ/ρ_c), ρ_c = 3/(8πGγ²λ²): the Big Bang
is replaced by a bounce at ρ = ρ_c. The exact solution is V(t) = √(V_B² + 12πG p_φ² t²), V_B = p_φ/√(2ρ_c).

Wheeler–DeWitt (the same model quantised without loops): with x = ln V the constraint becomes a free Klein–Gordon
equation in the clock φ, ∂²_φ Ψ = 12πG ∂²_x Ψ, solved exactly by a superposition of plane waves.
"""

import math

import torch

from .tensors import GAMMA, G


def critical_density(lam: torch.Tensor) -> torch.Tensor:
    """ρ_c = 3/(8πGγ²λ²); λ → 0 sends it to infinity (classical limit)."""
    return 3 / (8 * math.pi * G * GAMMA**2 * lam**2)


def density(y: torch.Tensor, p_phi: torch.Tensor) -> torch.Tensor:
    """ρ = p_φ²/(2V²), the scalar's energy density.   y (..., B, 3) → (..., B)"""
    return p_phi**2 / (2 * y[..., 0] ** 2)


def matter_energy(y: torch.Tensor, p_phi: torch.Tensor) -> torch.Tensor:
    """p_φ²/(2V): the matter term of the constraint (energy in the cell)."""
    return p_phi**2 / (2 * y[..., 0])


def constraint(y: torch.Tensor, p_phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r"""C = p_φ²/(2V) − 3V sin²(λb)/(8πGγ²λ²), which must stay 0.   → (..., B)"""
    V, b = y[..., 0], y[..., 1]
    return p_phi**2 / (2 * V) - 3 * V * torch.sin(lam * b) ** 2 / (8 * math.pi * G * GAMMA**2 * lam**2)


def hubble(y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """H = sin(2λb)/(2γλ) → b/γ as λ → 0."""
    return torch.sin(2 * lam * y[..., 1]) / (2 * GAMMA * lam)


def rhs(y: torch.Tensor, p_phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """dy/dt from Hamilton's equations, all batch members at once.   y (B, 3) → (B, 3)"""
    V, b = y[:, 0], y[:, 1]
    s, c = torch.sin(lam * b), torch.cos(lam * b)
    V_dot = 3 * V * s * c / (GAMMA * lam)
    b_dot = 4 * math.pi * G * GAMMA * (-(p_phi**2) / (2 * V**2) - 3 * s**2 / (8 * math.pi * G * GAMMA**2 * lam**2))
    phi_dot = p_phi / V
    return torch.stack([V_dot, b_dot, phi_dot], dim=-1)


def exact_volume(t: torch.Tensor, p_phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """V(t) = √(V_B² + 12πG p_φ² t²), t measured from the bounce."""
    V_B2 = p_phi**2 / (2 * critical_density(lam))
    return torch.sqrt(V_B2 + 12 * math.pi * G * p_phi**2 * t**2)


def time_to_bounce(y0: torch.Tensor, p_phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Invert exact_volume: how long until V reaches V_B.   → (B,)"""
    V_B2 = p_phi**2 / (2 * critical_density(lam))
    return torch.sqrt(y0[:, 0] ** 2 - V_B2) / (math.sqrt(12 * math.pi * G) * p_phi)


def classical_volume(t: torch.Tensor, V0: torch.Tensor, p_phi: torch.Tensor) -> torch.Tensor:
    """GR: V̇ = −√(12πG) p_φ exactly, so V falls linearly and hits 0 (clamped there): the singularity."""
    return (V0 - math.sqrt(12 * math.pi * G) * p_phi * t).clamp(min=0)


def classical_singularity_time(V0, p_phi) -> float:
    """Time for classical contraction from V0 to V = 0 (ρ = ∞)."""
    return float(V0 / (math.sqrt(12 * math.pi * G) * p_phi))


def wdw_packet(x: torch.Tensor, phi: torch.Tensor, p_phi: float, x_b: float, spread: float, n_k: int = 401):
    r"""Ψ(φ, x) = Σ_k A(k) e^{ik(x − x_b + ln 2 + √(12πG) φ)}: a contracting Wheeler–DeWitt packet.   → (P, X) complex

    Positive frequency ω = √(12πG) k with k > 0 peaked at k₀ = p_φ/√(12πG) (ħ = 1), Gaussian of relative width
    spread. Each plane wave solves ∂²_φ Ψ = 12πG ∂²_x Ψ, so the packet moves rigidly along the classical
    trajectory x = x_b − ln 2 − √(12πG) φ, straight into x → −∞ (V → 0). The summed index is k.
    """
    k0 = p_phi / math.sqrt(12 * math.pi * G)
    k = torch.linspace(k0 * (1 - 6 * spread), k0 * (1 + 6 * spread), n_k, dtype=x.dtype, device=x.device)
    A = torch.exp(-((k - k0) ** 2) / (2 * (spread * k0) ** 2))  # (K,)
    arg = x[None, :, None] - x_b + math.log(2) + math.sqrt(12 * math.pi * G) * phi[:, None, None]  # (P, X, 1)
    waves = torch.polar(torch.ones_like(arg * k), arg * k)  # (P, X, K)
    return torch.einsum("k,pxk->px", A.to(waves.dtype), waves)
