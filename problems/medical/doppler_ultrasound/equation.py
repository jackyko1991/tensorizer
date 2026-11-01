r"""Physics → tensor translation.

An ultrasound wave of frequency f_0 hits a red blood cell moving at speed v. The cell first receives a
shifted frequency (moving observer) and then re-emits it (moving source), so the echo is shifted twice:

    f_D = \frac{2 f_0 v \cos\theta}{c},      θ = angle between the beam and the flow,  c ≈ 1540 m/s in tissue.

In a vessel with laminar (Poiseuille) flow every scatterer n at radius r_n moves with

    v_{tn} = v_{\max}(t) \, (1 - r_n^2 / R^2)          (outer product: no index summed)

and the received (demodulated, complex IQ) signal is the sum of one rotating phasor per scatterer:

    s_t = a_n \, e^{i \phi_{tn}},     \phi_{tn} = \phi_n + 2\pi \sum_{t' \le t} f_{D,t'n} \, \Delta t

The repeated index n is summed — that is einsum('n,tn->t').  The real part of s_t is a sum of cosines
with random phases φ_n.
"""

import math

import torch

C_TISSUE = 1540.0  # m/s, average speed of sound in soft tissue


def doppler_shift(v: torch.Tensor, f0: float, theta: float, c: float = C_TISSUE) -> torch.Tensor:
    r"""f_D = 2 f_0 v \cos\theta / c   (any shape, Hz).   θ in radians."""
    return 2.0 * f0 * v * math.cos(theta) / c


def velocity_field(v_max: torch.Tensor, r: torch.Tensor, R: float) -> torch.Tensor:
    r"""v_{tn} = v_max(t) (1 - r_n²/R²).   v_max: (T,), r: (N,) → (T, N)"""
    profile = 1.0 - (r / R) ** 2  # (N,) parabolic profile: 1 on the axis, 0 at the wall
    return torch.einsum("t,n->tn", v_max, profile)


def received_signal(amplitude: torch.Tensor, phase0: torch.Tensor, f_D: torch.Tensor, dt: float) -> torch.Tensor:
    r"""s_t = a_n exp(i φ_{tn}),  φ_{tn} = φ_n + 2π Σ_{t'≤t} f_{D,t'n} Δt.   f_D: (T, N) → s: (T,) complex.

    The phase is the running integral of the frequency, so a scatterer whose speed changes over the cardiac
    cycle keeps a continuous phase.  Sampling it every Δt = 1/PRF is what makes pulsed Doppler alias.
    """
    phase = phase0 + 2 * math.pi * torch.cumsum(f_D, dim=0) * dt  # (T, N)
    phasors = torch.polar(torch.ones_like(phase), phase)  # (T, N) complex: e^{iφ}
    return torch.einsum("n,tn->t", amplitude.to(phasors.dtype), phasors)  # repeated n ⇒ summed


def alias(f: torch.Tensor, prf: float) -> torch.Tensor:
    """Frequency seen after sampling at the pulse repetition frequency: wrapped into [-PRF/2, PRF/2)."""
    return torch.remainder(f + prf / 2, prf) - prf / 2


def velocity_from_frequency(f: torch.Tensor, f0: float, theta: float, c: float = C_TISSUE) -> torch.Tensor:
    r"""Invert the Doppler equation: v = f_D c / (2 f_0 \cos\theta)."""
    return f * c / (2.0 * f0 * math.cos(theta))
