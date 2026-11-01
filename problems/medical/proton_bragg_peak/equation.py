r"""Physics → tensor translation.

A proton loses energy to atomic electrons at the rate given by the Bethe–Bloch formula
(no shell or density corrections), for water of density ρ:

    -\frac{dE}{dz} = K \rho \frac{Z}{A} \frac{1}{\beta^2}
                     \left[\tfrac12 \ln\frac{2 m_e c^2 \beta^2\gamma^2 T_{max}}{I^2} - \beta^2\right]

It grows like 1/β² as the proton slows down, so most energy is dumped just before it stops: the Bragg peak.
For a batch b of initial energies the ODE is integrated together, E_b(z) of shape (B,).

The depth-dose of pencil b, blurred by range straggling (Gaussian kernel G_{bzk}), and a spread-out
Bragg peak as a weighted sum of energy layers k:

    D_{bz} = G_{bzk}\, d_{bk}                (k = depth bin, summed)
    D^{SOBP}_z = D_{kz}\, w_k                (k = energy layer, summed)
"""

import torch

M_E = 0.51099895  # electron rest energy (MeV)
M_P = 938.272  # proton rest energy (MeV)
K = 0.307075  # 4π N_A r_e² m_e c² (MeV cm²/mol)
Z_OVER_A = 0.5551  # water: 10 electrons per 18.015 g/mol
RHO = 1.0  # g/cm³


def kinematics(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(β², β²γ², γ) for a proton of kinetic energy E (MeV); any shape."""
    gamma = 1.0 + E / M_P
    bg2 = gamma**2 - 1.0
    return bg2 / gamma**2, bg2, gamma


def stopping_power(E: torch.Tensor, I_mev: float) -> torch.Tensor:
    r"""-dE/dz in MeV/cm from the Bethe–Bloch formula (no shell or density correction)."""
    beta2, bg2, gamma = kinematics(E)
    t_max = 2 * M_E * bg2 / (1 + 2 * gamma * M_E / M_P + (M_E / M_P) ** 2)  # max energy to one electron
    log_term = 0.5 * torch.log(2 * M_E * bg2 * t_max / I_mev**2)
    return K * RHO * Z_OVER_A / beta2 * (log_term - beta2)


def rhs(E: torch.Tensor, I_mev: float, e_cut: float) -> torch.Tensor:
    """dE/dz for the batch (B,): the proton stops (rate 0) once E falls below e_cut, where Bethe fails."""
    moving = E > e_cut
    return torch.where(moving, -stopping_power(E.clamp_min(e_cut), I_mev), torch.zeros_like(E))


def bragg_kleeman(E: torch.Tensor, alpha: float, p: float) -> torch.Tensor:
    r"""Empirical range–energy rule R = \alpha E^p (cm, E in MeV)."""
    return alpha * E**p


def straggling_width(R: torch.Tensor) -> torch.Tensor:
    """Range straggling σ ≈ 0.012 R^0.951 (cm) in water — about 1.2 % of the range."""
    return 0.012 * R.clamp_min(1e-3) ** 0.951


def primary_fluence(z: torch.Tensor, kappa: float = 0.011) -> torch.Tensor:
    """Fraction of protons not yet removed by nuclear reactions, ≈ 1 % per cm (crude)."""
    return torch.exp(-kappa * z)


def straggle(d: torch.Tensor, z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""D_{bz} = G_{bzk} d_{bk}: blur each pristine curve with a Gaussian of width σ_b (cm).

    d: (B, Z) pristine dose, z: (Z,), sigma: (B,). Each row of G sums to 1 (a local weighted average),
    so a flat curve stays flat right up to the surface.
    """
    diff = z[None, :, None] - z[None, None, :]  # (1, Z, Z)  z_z − z_k
    G = torch.exp(-0.5 * (diff / sigma[:, None, None]) ** 2)  # (B, Z, Z)
    G = G / G.sum(dim=2, keepdim=True)
    return torch.einsum("bzk,bk->bz", G, d)


def sobp(layers: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""D^{SOBP}_z = D_{kz} w_k.   layers: (K, Z), w: (K,) → (Z,)"""
    return torch.einsum("kz,k->z", layers, w)


def xray_depth_dose(z: torch.Tensor, mu: float = 0.049, buildup: float = 2.5) -> torch.Tensor:
    r"""Megavoltage photon beam: D(z) ∝ (1 - e^{-b z})\, e^{-\mu z} (electron build-up, then exponential)."""
    return (1 - torch.exp(-buildup * z)) * torch.exp(-mu * z)
