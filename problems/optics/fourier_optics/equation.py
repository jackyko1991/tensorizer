r"""Physics → tensor translation.

Far behind an aperture (Fraunhofer limit, or in the focal plane of a lens) the field in the direction
(\sin\theta_x, \sin\theta_y) is the Fourier transform of the aperture's transmission A(x, y):

    U(q_x, q_y) \propto \iint A(x, y)\, e^{-i (q_x x + q_y y)}\, dx\, dy,    q = k \sin\theta

On an N×N pixel grid (row a = y, column b = x) this is the 2D discrete Fourier transform

    U_{kl} = \sum_{a,b} A_{ab}\, e^{-2\pi i (k a + l b)/N} = F_{ka} A_{ab} F_{lb},    F_{ka} = e^{-2\pi i k a / N}

where the repeated indices a and b are summed: one 1D transform along y, one along x.
torch.fft.fft2 computes exactly this sum, fast. The intensity is I_{kl} = U^*_{kl} U_{kl}.
"""

import math

import torch


def far_field(aperture: torch.Tensor) -> torch.Tensor:
    """U = fft2(A), with fftshift moving zero frequency (straight ahead) to the centre.   (N, N) → (N, N) complex"""
    return torch.fft.fftshift(torch.fft.fft2(aperture.to(torch.complex64)))


def dft_matrix(n: int) -> torch.Tensor:
    """F_{ka} = exp(−2πi k a / N): shape (N, N) complex."""
    k = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.einsum("k,a->ka", k, k) / n)


def far_field_einsum(aperture: torch.Tensor) -> torch.Tensor:
    r"""U_{kl} = F_{ka} A_{ab} F_{lb} written out with einsum (same as fft2, then fftshift).   (N, N) → (N, N)"""
    f = dft_matrix(aperture.shape[0])
    u = torch.einsum("ka,ab,lb->kl", f, aperture.to(f.dtype), f)
    return torch.fft.fftshift(u)


def intensity(u: torch.Tensor) -> torch.Tensor:
    """I_{kl} = U*_{kl} U_{kl}.   (N, N) complex → (N, N) real"""
    return (u.conj() * u).real


def log_intensity(i: torch.Tensor, floor: float = 1e-5) -> torch.Tensor:
    """log₁₀(I / I_max), clipped at log₁₀(floor) so the dark zeros stay finite."""
    return torch.log10((i / i.max()).clamp(min=floor))


def airy_radius(n: int, diameter: float) -> float:
    """First dark ring of a circular hole: sin θ = 1.22 λ / D  ⇒  1.22 N / D frequency pixels."""
    return 1.22 * n / diameter
