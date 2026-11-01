"""Build the concrete tensors: the acids (their pKa values), titrant volumes and total concentrations.

Units: concentrations in mol/L (M), volumes in mL. Every acid is padded to three protons; a missing proton gets
pKa = +∞ (Kₐ = 0), so it never comes off and the same formulas cover mono- and triprotic acids.
"""

import math

import torch

KW = 1e-14  # ion product of water at 25 °C


def acids(weak_pKa: float) -> tuple[list[str], list[list[float]]]:
    """Names and pKa lists. pKa values of H₃PO₄ are approximate textbook values at 25 °C."""
    return (
        ["HCl (strong)", f"weak acid, pKa {weak_pKa:g}", "H₃PO₄ (triprotic)"],
        [[-6.0], [weak_pKa], [2.15, 7.20, 12.35]],  # HCl: pKa ≈ -6, effectively fully dissociated
    )


def pka_table(pkas: list[list[float]], device: torch.device) -> torch.Tensor:
    """pKa_{aj} of shape (A, 3), padded with +∞ for protons an acid does not have."""
    rows = [p + [math.inf] * (3 - len(p)) for p in pkas]
    return torch.tensor(rows, dtype=torch.float64, device=device)


def log_beta(pka: torch.Tensor) -> torch.Tensor:
    """log₁₀ β_{aj} = log₁₀(K_{a1} ⋯ K_{aj}) = -(pK_{a1} + … + pK_{aj}),  β_{a0} = 1.   (A, 3) → (A, 4).

    A missing proton (pKa = +∞) gives log β = -∞, i.e. that species never forms.
    """
    zero = torch.zeros_like(pka[:, :1])
    return torch.cat([zero, -pka.cumsum(-1)], dim=-1)


def volumes(v_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Titrant volumes V_n in mL, shape (N,)."""
    return torch.linspace(0.0, v_max, n, dtype=torch.float64, device=device)


def totals(V: torch.Tensor, c_acid: float, v0: float, c_base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Total acid C_A(V) = c_a V₀/(V₀+V) and sodium [Na⁺](V) = c_b V/(V₀+V) after mixing, both (N,) in M."""
    total_volume = v0 + V
    return c_acid * v0 / total_volume, c_base * V / total_volume
