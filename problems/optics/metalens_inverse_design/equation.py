r"""Physics → tensor translation: scalar angular-spectrum propagation and the focusing figure of merit.

A 1D metasurface at z = 0 imprints a phase φ_l(x) on a plane wave of wavelength λ_l (l = wavelength index):

    E_{l}(x, 0) = A(x) e^{i φ_l(x)},        φ_l(x) = 2π \frac{λ_0}{λ_l} (θ(x) \bmod 1)

θ(x) ∈ [0, 1) is the design variable: the phase delay of the meta-atom at x, in waves at the design wavelength λ_0.
(Toy dispersion model: a fixed optical path, wrapped into one wave. Real meta-atoms have their own dispersion.)

Angular spectrum: each plane-wave component e^{i k_x x} picks up the phase e^{i k_z z}, k_z = \sqrt{k^2 - k_x^2}:

    \hat E_{lq}(z) = e^{i k_{z,lq} z} \hat E_{lq}(0),     \hat E_{lq} = F_{qx} E_{lx}   (F = discrete Fourier transform)

Components with |k_x| > k are evanescent: k_z is imaginary and they decay. The FFT does the sum over x.

Figure of merit: the fraction of the incident power that lands in the focal spot |x| < w at z = f,

    η_l = \sum_{|x| < w} |E_l(x, f)|^2 / \sum_x |E_l(x, 0)|^2
"""

import math

import torch


def kz(kx: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
    """k_{z,lq} = sqrt(k_l² - k_{x,q}²) as a complex number (imaginary for evanescent waves): (L, Q)."""
    k = 2 * math.pi / wavelengths  # (L,)
    return torch.sqrt((k[:, None] ** 2 - kx[None, :] ** 2).to(torch.complex64))


def propagate(E0: torch.Tensor, kz_lq: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    r"""E_l(x, z) for every wavelength l and distance z at once.

    E0 (L, X) complex, kz (L, Q), z (Z,)  →  (L, Z, X).   \hat E_{lzq} = e^{i k_{z,lq} z_z} \hat E_{lq}
    """
    spectrum = torch.fft.fft(E0, dim=-1)  # (L, Q)
    H = torch.exp(1j * kz_lq[:, None, :] * z[None, :, None])  # (L, Z, Q) transfer function
    return torch.fft.ifft(spectrum[:, None, :] * H, dim=-1)


def metasurface_phase(theta: torch.Tensor, wavelengths: torch.Tensor, lam0: float) -> torch.Tensor:
    """φ_l(x) = 2π (λ0 / λ_l) (θ(x) mod 1): theta (X,) → (L, X). The wrap is what makes a metalens chromatic."""
    return 2 * math.pi * (lam0 / wavelengths)[:, None] * torch.remainder(theta, 1.0)[None, :]


def hyperbolic_theta(x: torch.Tensor, f: float, lam0: float) -> torch.Tensor:
    r"""Analytic lens: every path from x to the focus is in phase, φ(x) = -k_0 (\sqrt{x^2 + f^2} - f).  In waves."""
    return -(torch.sqrt(x**2 + f**2) - f) / lam0


def field_at_surface(aperture: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """E_l(x, 0) = A(x) e^{i φ_l(x)}: (L, X) complex."""
    return aperture[None, :] * torch.exp(1j * phase)


def focal_efficiency(E_f: torch.Tensor, E0: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
    """η_l = Σ_x spot_{lx} |E_l(x, f)|² / Σ_x |E_l(x, 0)|²  (repeated x ⇒ summed): (L,)."""
    return torch.einsum("lx,lx->l", spot, E_f.abs() ** 2) / torch.einsum("lx->l", E0.abs() ** 2)
