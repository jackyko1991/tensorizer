"""Build the tensors: physical constants, the Daniell cell, concentration and temperature batches, radial grids.

Units: potentials in V, temperatures in K, molalities in mol/kg (≈ mol/L for dilute water), lengths in nm.
"""

import torch

F = 96485.332  # Faraday constant, C/mol
R_GAS = 8.314462  # gas constant, J/(mol K)
N_A = 6.02214076e23  # Avogadro, 1/mol
E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J/K
EPS0 = 8.8541878128e-12  # F/m

# Daniell cell  Zn | Zn²⁺(aq) || Cu²⁺(aq) | Cu:   Cu²⁺ + Zn → Cu + Zn²⁺,  two electrons, E° ≈ 1.10 V
N_ELECTRONS = 2
Z_ION = 2  # Zn²⁺ and Cu²⁺, each with SO₄²⁻: a 2:2 salt, ionic strength I = ½ Σ m z² = 4 m


def temperatures(device: torch.device) -> torch.Tensor:
    """A batch of cell temperatures, 0–80 °C   (T,) in K."""
    return torch.tensor([0.0, 20.0, 40.0, 60.0, 80.0], device=device) + 273.15


def log_ratios(device: torch.device) -> torch.Tensor:
    """log₁₀ Q of the reaction quotient Q = a(Zn²⁺)/a(Cu²⁺), from 10⁻⁴ to 10⁴   (Q,)."""
    return torch.linspace(-4.0, 4.0, 81, device=device)


def zinc_molalities(n: int, device: torch.device) -> torch.Tensor:
    """The swept Zn²⁺ molality m (C,): 10⁻⁴ … 10⁻¹ mol/kg (one animation frame each)."""
    return torch.logspace(-4.0, -1.0, n, device=device)


def ionic_strength(m: torch.Tensor, z: int = Z_ION) -> torch.Tensor:
    """I = ½ (m z² + m z²) = m z² for a symmetric z:z salt   (same shape as m)."""
    return m * z**2


def radial_grids(
    kappa: torch.Tensor, a: float, n_points: int, reach: float = 12.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """r_ci (C, M) from the ion surface r = a out to a + reach/κ_c, and the spacing h_c (C,), in nm.

    Each concentration gets its own grid measured in Debye lengths, so a thick cloud and a thin cloud are
    resolved equally well. Point M (not stored) sits at the outer edge, where φ = 0.
    """
    h = reach / kappa / n_points  # (C,)
    i = torch.arange(n_points, device=kappa.device, dtype=kappa.dtype)  # (M,)
    return a + torch.einsum("c,i->ci", h, i), h
