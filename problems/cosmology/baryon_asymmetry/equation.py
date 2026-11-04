r"""Physics → tensor translation: the Boltzmann equations of thermal leptogenesis (decays and inverse decays).

A heavy Majorana neutrino N of mass M decays into a lepton and a Higgs, slightly more often into one than into the
CP conjugate (CP asymmetry ε). With z = M/T as the clock and abundances per comoving volume that holds one photon
at z ≪ 1 (Buchmüller, Di Bari & Plümacher 2005), the two equations are

    dN_N/dz      = −D(z)\,(N_N − N_N^{eq}),
    dN_{B−L}/dz  = ε\,D(z)\,(N_N − N_N^{eq}) − W(z)\,N_{B−L},

    D = K z \frac{K_1(z)}{K_2(z)},     W = \frac14 K z^3 K_1(z),     N_N^{eq} = \frac38 z^2 K_2(z),

with K_1, K_2 modified Bessel functions. The decay parameter K = Γ_D / H(T = M) compares the decay rate to the
expansion rate: K ≪ 1 is weak washout, K ≫ 1 strong washout. Sphalerons turn the final B − L into baryons:

    η_B = \frac{a_{sph}}{f}\,N_{B−L}^f,   a_{sph} = 28/79,   f = 2387/86 (photon production after leptogenesis),

i.e. η_B ≈ 0.96 × 10^{-2} ε κ_f, where κ_f = N_{B−L}^f / (\tfrac34 ε) is the efficiency factor
(≤ 1 for a thermal start).
Here ε is defined as the B − L produced per decay, so ε > 0 makes matter.
"""

import torch

A_SPH = 28 / 79  # fraction of B − L that sphalerons leave as B (Standard Model, one Higgs doublet)
DILUTION = 2387 / 86  # f: photons per comoving volume grow by this factor until recombination


def k1(z):
    return torch.special.modified_bessel_k1(z)


def k2(z):
    """K_2(z) = K_0(z) + (2/z) K_1(z) (recurrence; torch has K_0 and K_1)."""
    return torch.special.modified_bessel_k0(z) + 2 / z * k1(z)


def n_eq(z: torch.Tensor) -> torch.Tensor:
    r"""N_N^{eq} = (3/8) z² K₂(z): 3/4 while relativistic (z ≪ 1), Boltzmann-suppressed e^{−z} later."""
    return 3 / 8 * z**2 * k2(z)


def dn_eq_dz(z: torch.Tensor) -> torch.Tensor:
    r"""dN^{eq}/dz = −(3/8) z² K₁(z), from d(z²K₂)/dz = −z²K₁."""
    return -3 / 8 * z**2 * k1(z)


def decay_rate(z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    r"""D = K z K₁/K₂: decays per Hubble time (K₁/K₂ is the average time-dilation factor ⟨1/γ⟩).   → broadcast"""
    return K * z * k1(z) / k2(z)


def washout(z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    r"""W = (1/4) K z³ K₁(z): inverse decays ℓ H → N erase the asymmetry; switched off by e^{−z} once T < M."""
    return K / 4 * z**3 * k1(z)


def phi(x: torch.Tensor) -> torch.Tensor:
    r"""(1 − e^{−x})/x, equal to 1 at x = 0 (the exact one-step factor for dy/dz = −a y + s with constant a, s)."""
    return torch.where(x > 1e-12, -torch.expm1(-x) / x.clamp_min(1e-300), 1 - x / 2)


def baryon_to_photon(n_bl: torch.Tensor) -> torch.Tensor:
    r"""η_B = (a_{sph}/f) N_{B−L}^f ≈ 0.0128 N_{B−L}^f."""
    return A_SPH / DILUTION * n_bl


def kappa_strong_washout(K: torch.Tensor) -> torch.Tensor:
    r"""Analytic fit of the efficiency (decays + inverse decays, any start): κ_f = (2/(z_B K))(1 − e^{−z_B K/2})."""
    z_b = 2 + 4 * K**0.13 * torch.exp(-2.5 / K)
    return 2 / (z_b * K) * (1 - torch.exp(-z_b * K / 2))


def effective_mass(K: float) -> float:
    r"""m̃₁ = K m* in eV, with the equilibrium neutrino mass m* ≈ 1.08 × 10⁻³ eV."""
    return K * M_STAR_EV


M_STAR_EV = 1.08e-3
