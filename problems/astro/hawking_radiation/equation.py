r"""Physics → tensor translation.

Quantum fields near a horizon make a black hole glow like a black body (Hawking 1974) at the temperature

    T_H = \frac{\hbar c^3}{8\pi G M k_B}           (smaller hole ⇒ hotter)

Its spectrum is Planck's law, evaluated for a batch of masses m and wavelengths l at once by broadcasting:

    B_{ml} = \frac{2 h c^2}{λ_l^5}\,\frac{1}{e^{hc/(λ_l k_B T_m)} - 1}

and peaks at Wien's λ_max = b / T_H. Treating the horizon (area 4π r_s², r_s = 2GM/c²) as a black body
radiating photons, the Stefan–Boltzmann law gives the power and the mass loss:

    P = σ T_H^4 \, 4π r_s^2 = \frac{\hbar c^6}{15360\,π G^2 M^2},     \frac{dM}{dt} = -\frac{P}{c^2} = -\frac{α}{M^2}

Integrating, M^3 falls linearly in time and the hole vanishes after

    t_{life} = \frac{M_0^3}{3α} = \frac{5120\,π G^2 M_0^3}{\hbar c^4}.

The last moments are a burst: P ∝ 1/M² grows without bound as M → 0.
"""

import math

import torch

# SI constants
HBAR = 1.054571817e-34  # J s
H = 2 * math.pi * HBAR
C = 2.99792458e8  # m/s
G = 6.67430e-11  # m³/(kg s²)
KB = 1.380649e-23  # J/K
SIGMA = 5.670374419e-8  # W/(m² K⁴)
WIEN_B = 2.897771955e-3  # m K
M_SUN = 1.98847e30  # kg


def hawking_temperature(M: torch.Tensor) -> torch.Tensor:
    r"""T_H = ħc³ / (8π G M k_B) for every mass, (...) → (...) in kelvin."""
    return HBAR * C**3 / (8 * math.pi * G * M * KB)


def schwarzschild_radius(M: torch.Tensor) -> torch.Tensor:
    return 2 * G * M / C**2


def planck(lam: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    r"""Spectral radiance B_{ml} for masses m (temperatures T_m) and wavelengths λ_l.   (L,), (M,) → (M, L)"""
    x = H * C / (lam[None, :] * KB * T[:, None])  # (M, L) broadcast: one row per black hole
    return 2 * H * C**2 / lam[None, :] ** 5 / torch.expm1(x)


def wien_peak(T: torch.Tensor) -> torch.Tensor:
    """λ_max = b / T."""
    return WIEN_B / T


def power(M: torch.Tensor) -> torch.Tensor:
    r"""P = σ T_H⁴ · 4π r_s² (photons only), in watts."""
    return SIGMA * hawking_temperature(M) ** 4 * 4 * math.pi * schwarzschild_radius(M) ** 2


def mass_loss_rate(M: torch.Tensor) -> torch.Tensor:
    r"""dM/dt = −P/c² = −α/M²."""
    return -power(M) / C**2


def lifetime(M0: torch.Tensor) -> torch.Tensor:
    r"""t_life = 5120 π G² M_0³ / (ħ c⁴), in seconds."""
    return 5120 * math.pi * G**2 * M0**3 / (HBAR * C**4)
