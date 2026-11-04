r"""Physics → tensor translation: the Planck law and its two small distortions.

FIRAS measured the sky intensity I_ν against wavenumber ν̃ = ν/c in cm⁻¹. With x = hν/kT = c₂ ν̃ / T
(c₂ = hc/k = 1.4388 cm K) a blackbody has the photon occupation n = 1/(eˣ − 1) and

    I_ν = 2hν³/c² · n = 2hc ν̃³ · n          (per Hz per steradian)

Two ways energy injected into the early universe would bend this curve:

    μ-distortion (z ≳ 2×10⁵, Bose–Einstein equilibrium):  n = 1 / (e^{x + μ} − 1)
    y-distortion (z ≲ 5×10⁴, Compton scattering off hot electrons, Kompaneets):
        Δn = y · x eˣ/(eˣ − 1)² · (x coth(x/2) − 4)

so the full model, for a batch of B parameter rows θ_b = (T_b, μ_b, y_b) and F frequencies, is

    I_{bf} = 2hc ν̃_f³ [ 1/(e^{x_{bf} + μ_b} − 1) + y_b Y(x_{bf}) ],     x_{bf} = c₂ ν̃_f / T_b

No index is summed: the batch is pure broadcasting (B, 1) against (F,).
"""

import math

import torch

H_PLANCK = 6.62607015e-34  # J s
K_B = 1.380649e-23  # J/K
C_LIGHT = 299792458.0  # m/s
C2 = H_PLANCK * C_LIGHT / K_B * 100  # cm K, second radiation constant: x = C2 ν̃ / T
PREF = 2 * H_PLANCK * C_LIGHT * 100**3 * 1e20  # 2hc ν̃³ in MJy/sr for ν̃ in cm⁻¹ (1 MJy = 1e-20 W m⁻² Hz⁻¹)
ZETA3 = 1.2020569031595942
EV = 1.602176634e-19  # J


def planck(nu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Blackbody I_ν in MJy/sr at wavenumber nu (cm⁻¹) and temperature T (K); any broadcastable shapes."""
    x = C2 * nu / T
    return PREF * nu**3 / torch.expm1(x)


def y_shape(x: torch.Tensor) -> torch.Tensor:
    r"""Y(x) = Δn / y = x eˣ/(eˣ − 1)² · (x coth(x/2) − 4), the Compton (Sunyaev–Zel'dovich) distortion."""
    return x * torch.exp(x) / torch.expm1(x) ** 2 * (x / torch.tanh(x / 2) - 4)


def spectrum(nu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """I_{bf} for parameter rows theta (B, 3) = (T, μ, y) and wavenumbers nu (F,) → (B, F) in MJy/sr."""
    T, mu, y = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]  # (B, 1) each, broadcast against (F,)
    x = C2 * nu / T  # (B, F)
    occupation = 1 / torch.expm1(x + mu) + y * y_shape(x)  # (B, F) Bose–Einstein + Kompaneets
    return PREF * nu**3 * occupation


def chi2(model: torch.Tensor, data: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""χ²_b = Σ_f ((d_f − I_{bf}) / σ_f)²: repeated f summed.   (B, F) → (B,)"""
    r = (data - model) / sigma
    return torch.einsum("bf,bf->b", r, r)


def photon_number_density(T: torch.Tensor) -> torch.Tensor:
    r"""n_γ = (2ζ(3)/π²) (kT/ħc)³ in cm⁻³ (≈ 411 for the CMB)."""
    hbar = H_PLANCK / (2 * math.pi)
    return 2 * ZETA3 / math.pi**2 * (K_B * T / (hbar * C_LIGHT)) ** 3 * 1e-6


def energy_density(T: torch.Tensor) -> torch.Tensor:
    r"""u_γ = (π²/15) (kT)⁴/(ħc)³ = a T⁴ in eV cm⁻³ (≈ 0.26 for the CMB)."""
    hbar = H_PLANCK / (2 * math.pi)
    return math.pi**2 / 15 * (K_B * T) ** 4 / (hbar * C_LIGHT) ** 3 / EV * 1e-6
