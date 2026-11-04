"""Build the concrete tensors: PDG constants of the neutral kaons (in units of τ_S and ħ = 1) and the time grid.

PDG 2024 (Review of Particle Physics, Phys. Rev. D 110, 030001):
    τ_S = 0.8954 × 10⁻¹⁰ s,  τ_L = 5.116 × 10⁻⁸ s,  Δm = m_L − m_S = 0.5293 × 10¹⁰ ħ s⁻¹,
    |ε| = 2.228 × 10⁻³, φ_ε = 43.52°,  |η₊₋| = 2.232 × 10⁻³, φ₊₋ = 43.4°,  m_K = 497.611 MeV.
CPLEAR 1998 (Phys. Lett. B 444, 43): Kabir asymmetry ⟨A_T⟩ = (6.6 ± 1.3 stat ± 1.0 syst) × 10⁻³.
"""

import cmath
import math

import torch

TAU_S, TAU_L = 0.8954e-10, 5.116e-8  # s
DM_PER_S = 0.5293e10  # Δm in ħ/s
GAMMA_S = 1.0  # time unit τ_S, so Γ_S = 1
GAMMA_L = TAU_S / TAU_L  # ≈ 1.75 × 10⁻³
DM = DM_PER_S * TAU_S  # ≈ 0.474: Δm in units of ħ/τ_S
EPS_ABS, EPS_PHASE = 2.228e-3, 43.52  # |ε|, φ_ε in degrees
ETA_PM_ABS, ETA_PM_PHASE = 2.232e-3, 43.4  # |η₊₋|, φ₊₋ in degrees
M_K = 497.611  # MeV, the common mass (only differences enter the evolution)
CPLEAR_AT = (6.6e-3, math.hypot(1.3e-3, 1.0e-3))  # value, total error


def epsilon(modulus: float, phase_deg: float) -> complex:
    """A complex CP parameter from its modulus and phase in degrees."""
    return cmath.rect(modulus, math.radians(phase_deg))


def times(t_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Proper times 0 … t_max in units of τ_S: shape (T,)."""
    return torch.linspace(0.0, t_max, n, dtype=torch.float64, device=device)


def scenario_parameters(eps: complex, delta: complex, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Three kaon worlds side by side: the chosen (ε, δ), CPT conserved (δ = 0) and no CP violation at all.

    Returns ε and δ, each (3,) complex, so every later tensor carries a leading batch axis B = 3.
    """
    e = torch.tensor([eps, eps, 0.0], dtype=torch.complex128, device=device)
    d = torch.tensor([delta, 0.0, 0.0], dtype=torch.complex128, device=device)
    return e, d
