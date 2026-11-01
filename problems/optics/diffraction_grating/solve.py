"""Compute the grating pattern for every slit count, and the phasor chains drawn in the animation."""

import torch

from .equation import array_factor, envelope, intensity, wavenumber
from .tensors import phasor_angles


def solve(counts: list[int], sin_theta: torch.Tensor, wavelength: float, d: float, a: float) -> torch.Tensor:
    """Intensity I_t for every N in counts: shape (len(counts), T), normalised to 1 at θ = 0."""
    k = wavenumber(wavelength)
    s = envelope(sin_theta, a, wavelength)  # (T,) the same for every N
    return torch.stack([intensity(array_factor(n, sin_theta, k, d), s) for n in counts])


def phasor_chain(n_slits: int, phi: float) -> torch.Tensor:
    """Tips of N unit phasors e^{i n φ} drawn head to tail, starting at 0: (N + 1,) complex.

    The last point is the (un-normalised) sum N·E; the chain closes into a polygon when Nφ = 2π.
    """
    steps = torch.exp(1j * phi * torch.arange(n_slits))  # (N,)
    return torch.cat([torch.zeros(1, dtype=steps.dtype), torch.cumsum(steps, dim=0)])


def frames(counts: list[int]) -> list[tuple[int, float, torch.Tensor]]:
    """One (index into counts, φ, phasor chain) entry per animation frame."""
    out = []
    for i, n in enumerate(counts):
        for phi in phasor_angles(n).tolist():
            out.append((i, phi, phasor_chain(n, phi)))
    return out


def principal_maxima(wavelength: float, d: float, sin_max: float) -> list[float]:
    """sin θ_m = m λ / d for every order m with |sin θ_m| ≤ sin_max."""
    m_max = int(sin_max * d / wavelength)
    return [m * wavelength / d for m in range(-m_max, m_max + 1)]
