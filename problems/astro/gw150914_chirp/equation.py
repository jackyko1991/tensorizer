r"""Physics → tensor translation for a compact-binary chirp and its matched filter.

Leading (Newtonian, quadrupole) order: two masses of chirp mass M_c = (m_1 m_2)^{3/5}/(m_1 + m_2)^{1/5} spiral in,
and the gravitational-wave frequency f (twice the orbital frequency) rises as

    df/dt = (96/5) π^{8/3} M_c^{5/3} f^{11/3}         ⇒   f(τ) = (1/π) (5/256 τ)^{3/8} M_c^{−5/8}

with τ = t_c − t the time left to coalescence and M_c in seconds (G M_c / c³; 1 M☉ = 4.925 μs).

Stationary-phase (SPA) frequency-domain template, post-Newtonian phase up to 2PN (η = m_1 m_2/M², v = (π M f)^{1/3}):

    h̃(f) = f^{−7/6} exp(−i[2π f t_c − π/4 + Ψ(f)]),
    Ψ(f) = 3/(128 η v⁵) [1 + (3715/756 + 55η/9) v² − 16π v³ + (15293365/508032 + 27145η/504 + 3085η²/72) v⁴]

Noise-weighted inner product (one-sided PSD S_n(f)) and the matched-filter SNR time series:

    ⟨a|b⟩ = 4 Re Σ_k ã(f_k) b̃*(f_k) / S_n(f_k) Δf,        z(t) = 4 Σ_k d̃(f_k) h̃*(f_k) / S_n(f_k) e^{2π i f_k t} Δf

with ⟨h|h⟩ = 1, so ρ(t) = |z(t)| is the SNR maximised over the template's phase. Index k (frequency) is summed.
"""

import math

import torch

G = 6.67430e-11  # m³ kg⁻¹ s⁻²
C = 299_792_458.0  # m/s
M_SUN = 1.98847e30  # kg
T_SUN = G * M_SUN / C**3  # 4.925 μs: one solar mass expressed as a time


def chirp_frequency(tau: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
    r"""f(τ) = (1/π)(5/(256 τ))^{3/8} M_c^{−5/8}   τ (…,) seconds to merger, mc in seconds (broadcast) → Hz."""
    return (5.0 / (256.0 * tau)) ** 0.375 * mc ** (-0.625) / math.pi


def chirp_rate(f: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
    r"""df/dt = (96/5) π^{8/3} M_c^{5/3} f^{11/3}   (Hz/s)."""
    return 96.0 / 5.0 * math.pi ** (8 / 3) * mc ** (5 / 3) * f ** (11 / 3)


def time_to_coalescence(f: torch.Tensor, mc) -> torch.Tensor:
    r"""τ(f) = (5/256) (π f)^{−8/3} M_c^{−5/3}, the inverse of chirp_frequency."""
    return 5.0 / 256.0 * (math.pi * f) ** (-8 / 3) * mc ** (-5 / 3)


def spa_phase(f: torch.Tensor, mc: torch.Tensor, eta: float = 0.25, pn_order: int = 2) -> torch.Tensor:
    r"""Ψ(f) up to pn_order 0 (Newtonian), 1 (1PN) or 2 (1.5PN tail + 2PN): f (F,), mc (K, 1) seconds → (K, F)."""
    total = mc * eta ** (-0.6)  # M = M_c η^{−3/5}
    v = (math.pi * total * f) ** (1 / 3)
    terms = torch.ones_like(v)
    if pn_order >= 1:
        terms = terms + (3715 / 756 + 55 * eta / 9) * v**2
    if pn_order >= 2:
        terms = terms - 16 * math.pi * v**3 + (15293365 / 508032 + 27145 * eta / 504 + 3085 * eta**2 / 72) * v**4
    return 3.0 / (128.0 * eta * v**5) * terms


def peak_frequency(mc: torch.Tensor, eta: float = 0.25) -> torch.Tensor:
    r"""f_peak = 0.36 / (2π M): GW frequency where an equal-mass, non-spinning merger peaks (M ω₂₂ ≈ 0.36, NR)."""
    return 0.36 / (2 * math.pi * mc * eta ** (-0.6))


def isco_frequency(mc: torch.Tensor, eta: float = 0.25) -> torch.Tensor:
    r"""f_ISCO = 1 / (6^{3/2} π M): twice the orbital frequency at the innermost stable circular orbit."""
    return 1.0 / (6**1.5 * math.pi * mc * eta ** (-0.6))


def spa_template(f: torch.Tensor, mc: torch.Tensor, f_low: float, f_high: torch.Tensor, pn_order: int = 2):
    r"""h̃(f) = f^{−7/6} e^{−i(Ψ − π/4)} for f_low ≤ f ≤ f_high (t_c = 0).   f (F,), mc and f_high (K, 1) → (K, F)"""
    fs = f.clamp(min=1.0)  # avoid f = 0 (masked anyway)
    band = (f >= f_low) & (f <= f_high)
    phase = spa_phase(fs, mc, pn_order=pn_order) - math.pi / 4
    return torch.where(band, fs ** (-7 / 6) * torch.exp(-1j * phase), torch.zeros((), dtype=torch.complex128))


def inner(a: torch.Tensor, b: torch.Tensor, psd: torch.Tensor, df) -> torch.Tensor:
    r"""⟨a|b⟩ = 4 Re Σ_k a_k b*_k / S_k Δf — k summed.   (…, F) → (…,)"""
    return 4.0 * df * (a * b.conj() / psd).real.sum(-1)


def snr_series(data_f: torch.Tensor, h: torch.Tensor, psd: torch.Tensor, df, n: int) -> torch.Tensor:
    r"""z(t) = 4 Σ_k d̃_k h̃*_k / S_k e^{2πi f_k t} Δf for every template.   d̃, S (D, F), h̃ (K, F) → (D, K, n)

    Only positive frequencies enter, so the inverse FFT returns a complex series whose modulus is the SNR
    maximised over the template phase (the real and imaginary parts are the cos and sin templates).
    """
    integrand = 4.0 * df * data_f[:, None, :] * h.conj()[None] / psd[:, None, :]  # (D, K, F)
    return torch.fft.ifft(integrand, n=n, dim=-1) * n  # ifft has a 1/n; Σ_k e^{2πi k j/n}
