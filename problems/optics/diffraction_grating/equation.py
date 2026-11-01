r"""Physics → tensor translation.

Far from N equal slits (Fraunhofer limit) the wave from slit n, at height n d, travels an extra
path n d \sin\theta to reach direction θ, i.e. it picks up a phase k d n \sin\theta (k = 2π/λ).
The total complex amplitude is the sum of N unit phasors,

    E(\theta) = \frac{1}{N} \sum_{n=0}^{N-1} e^{\,i k d n \sin\theta}

In index notation with n = slit and t = sample of the angle:

    P_{nt} = e^{\,i k d\, n \sin\theta_t},    E_t = A_n P_{nt}        (repeated n ⇒ summed)

which is einsum('n,nt->t') on complex tensors. Each slit has a width a, which multiplies the
intensity by the single-slit envelope S_t = sinc²(a \sin\theta_t / λ):

    I_t = S_t\, E^*_t E_t

Summing the geometric series by hand gives the closed form (φ = k d \sin\theta)

    |E|^2 = \frac{\sin^2(N\phi/2)}{N^2 \sin^2(\phi/2)}
"""

import math

import torch


def phasors(n: torch.Tensor, sin_theta: torch.Tensor, k: float, d: float) -> torch.Tensor:
    r"""P_{nt} = exp(i k d n sin θ_t).   n: (N,), sin_theta: (T,) → (N, T) complex"""
    phase = k * d * torch.einsum("n,t->nt", n.to(sin_theta.dtype), sin_theta)  # (N, T) outer product
    return torch.exp(1j * phase)


def array_factor(n_slits: int, sin_theta: torch.Tensor, k: float, d: float) -> torch.Tensor:
    r"""E_t = A_n P_{nt}  with A_n = 1/N for every slit.   → (T,) complex"""
    n = torch.arange(n_slits)
    p = phasors(n, sin_theta, k, d)  # (N, T)
    amplitude = torch.full((n_slits,), 1.0 / n_slits, dtype=p.dtype)  # (N,) complex
    return torch.einsum("n,nt->t", amplitude, p)


def envelope(sin_theta: torch.Tensor, a: float, wavelength: float) -> torch.Tensor:
    """Single-slit factor S_t = sinc²(a sin θ / λ), with torch.sinc(x) = sin(πx)/(πx).   (T,) → (T,)"""
    return torch.sinc(a * sin_theta / wavelength) ** 2


def intensity(e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """I_t = S_t E*_t E_t.   (T,) complex, (T,) → (T,) real"""
    return s * (e.conj() * e).real


def closed_form(sin_theta: torch.Tensor, n_slits: int, k: float, d: float) -> torch.Tensor:
    """sin²(Nφ/2) / (N² sin²(φ/2)) with φ = k d sin θ; equal to 1 where sin(φ/2) = 0.   (T,) → (T,)"""
    half_phi = 0.5 * k * d * sin_theta
    num, den = torch.sin(n_slits * half_phi) ** 2, (n_slits * torch.sin(half_phi)) ** 2
    return torch.where(den < 1e-12, torch.ones_like(den), num / den.clamp(min=1e-30))


def wavenumber(wavelength: float) -> float:
    """k = 2π / λ."""
    return 2 * math.pi / wavelength
