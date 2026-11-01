r"""Physics → tensor translation.

One Feynman diagram: e⁻(p) e⁺(p') annihilate into a virtual photon, q = p + p', which makes ℓ⁻(k) ℓ⁺(k').
The Feynman rules (vertex −ieγ^μ, propagator −iη_{μν}/q², spinors u, v) give

    \mathcal M = \frac{e^2}{s}\,[\bar v(p')\gamma^\mu u(p)]\,[\bar u(k)\gamma_\mu v(k')],     s = q^2.

Summing |M|² over all spins turns the spinors into traces of 4×4 Dirac matrices (Σ u ū = p̸ + m, Σ v v̄ = p̸ − m,
with p̸ = γ^μ η_{μν} p^ν), and averaging over the 4 initial spin states:

    \overline{|\mathcal M|^2} = \frac{e^4}{4 s^2}\,L^{\mu\nu} M_{\mu\nu},
    L^{\mu\nu} = \mathrm{Tr}[(\not p' - m_e)\gamma^\mu(\not p + m_e)\gamma^\nu],
    M_{\mu\nu} = \mathrm{Tr}[(\not k + m)\gamma_\mu(\not k' - m)\gamma_\nu].

Each trace is one einsum over the spinor indices a, b, c, d, and L^{μν}M_{μν} is one more over μ, ν. In the
centre-of-mass frame the cross section per solid angle is

    \frac{d\sigma}{d\Omega} = \frac{1}{64\pi^2 s}\,\frac{|\mathbf k|}{|\mathbf p|}\,\overline{|\mathcal M|^2}.
"""

import math

import torch

GEV2_TO_NB = 0.3893794e6  # (ħc)² in GeV² nb: converts GeV⁻² to nanobarn


def slash(p: torch.Tensor, gamma: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""p̸_{ab} = γ^μ_{ab} η_{μν} p^ν   (μ, ν summed).   p: (..., 4) real → (..., 4, 4) complex"""
    return torch.einsum("mab,mn,...n->...ab", gamma, eta, p.to(gamma.dtype))


def trace_tensor(a: torch.Tensor, b: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""T^{μν} = Tr[A γ^μ B γ^ν] = A_{ab} γ^μ_{bc} B_{cd} γ^ν_{da}   (a, b, c, d summed).
    a, b: (..., 4, 4) → (..., 4, 4)"""
    return torch.einsum("...ab,mbc,...cd,nda->...mn", a, gamma, b, gamma)


def amplitude_squared(p, pb, k, kb, m_e: float, m_f: float, alpha: float, gamma, eta) -> torch.Tensor:
    r"""Spin-averaged |M|² = e⁴/(4s²) L^{μν} η_{μρ} η_{νσ} M^{ρσ}.   momenta: (..., 4) → (...)"""
    one = torch.eye(4, dtype=gamma.dtype, device=gamma.device)
    lepton = trace_tensor(slash(pb, gamma, eta) - m_e * one, slash(p, gamma, eta) + m_e * one, gamma)  # L^{μν}
    muon = trace_tensor(slash(k, gamma, eta) + m_f * one, slash(kb, gamma, eta) - m_f * one, gamma)  # M^{ρσ}
    contracted = torch.einsum("...mn,mr,ns,...rs->...", lepton, eta, eta, muon)  # L^{μν} M_{μν}
    q = (p + pb).to(gamma.dtype)
    s = torch.einsum("...m,mn,...n->...", q, eta, q).real
    e2 = 4 * math.pi * alpha
    return (e2**2 / (4 * s**2) * contracted).real


def dsigma_domega(msq: torch.Tensor, sqrt_s: torch.Tensor, m_e: float, m_f: float) -> torch.Tensor:
    r"""dσ/dΩ = |k|/|p| · |M|²/(64π² s) in nb/sr.   msq: (E, B), sqrt_s: (E,) → (E, B)"""
    s = sqrt_s**2
    ratio = ((s / 4 - m_f**2).clamp(min=0) / (s / 4 - m_e**2)).sqrt()
    return (ratio / (64 * math.pi**2 * s))[:, None] * msq * GEV2_TO_NB


def sigma_point_like(sqrt_s: torch.Tensor, m_f: float, alpha: float) -> torch.Tensor:
    r"""σ = (4πα²/3s) β (1 + 2m²/s), β = √(1 − 4m²/s), m_e → 0, in nb.   (E,) → (E,)"""
    s = sqrt_s**2
    x = 4 * m_f**2 / s
    beta = (1 - x).clamp(min=0).sqrt()
    return 4 * math.pi * alpha**2 / (3 * s) * beta * (1 + x / 2) * GEV2_TO_NB
