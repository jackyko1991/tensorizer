r"""Physics → tensor translation.

Linear waves on deep water obey the dispersion relation ω(k) = √(g |k|). Each Fourier mode of the surface
evolves on its own, it only turns its phase:

    \hat a_{nk} = \hat a_k \, e^{-i \omega_k t_n}

and the surface at time t_n is the inverse Fourier transform (k summed):

    a_{nj} = F^{-1}_{jk} \hat a_{nk},        \eta_{nj} = \mathrm{Re}\, a_{nj}

Phase velocity v_p = ω/k, group velocity v_g = dω/dk = ½ √(g/k) = ½ v_p.
"""

import math

import torch


def omega(k: torch.Tensor, g: float) -> torch.Tensor:
    """ω_k = √(g |k|).   (K,) → (K,)"""
    return torch.sqrt(g * k.abs())


def evolve(a0: torch.Tensor, k: torch.Tensor, t: torch.Tensor, g: float) -> torch.Tensor:
    r"""a_{nj} = F^{-1}_{jk} [\hat a_k e^{−iω_k t_n}] for every time at once.   a0: (J,), t: (N,) → (N, J) complex"""
    a_hat = torch.fft.fft(a0)  # (K,)
    phase = torch.exp(-1j * omega(k, g)[None, :] * t[:, None])  # (N, K)
    return torch.fft.ifft(a_hat[None, :] * phase, dim=-1)  # (N, J)


def phase_velocity(k0: float, g: float) -> float:
    """v_p = ω/k = √(g/k)."""
    return math.sqrt(g / k0)


def group_velocity(k0: float, g: float) -> float:
    """v_g = dω/dk = ½ √(g/k)."""
    return 0.5 * math.sqrt(g / k0)


def wave_energy(a: torch.Tensor) -> torch.Tensor:
    r"""E_n ∝ Σ_j |a_{nj}|² (= Σ_k |\hat a_{nk}|² / K by Parseval).   (N, J) → (N,)"""
    return torch.einsum("nj,nj->n", a.real, a.real) + torch.einsum("nj,nj->n", a.imag, a.imag)
