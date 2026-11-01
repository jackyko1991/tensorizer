r"""Physics → tensor translation.

Free-particle Schrödinger equation (units ħ = m = 1):

    i \partial_t \psi = -\tfrac12 \partial_x^2 \psi.

A plane wave e^{ikx} is an eigenfunction of \partial_x^2 with eigenvalue -k^2, so in the Fourier
(plane-wave) basis the equation decouples into one ODE per mode, i \partial_t \tilde\psi_k = \tfrac12 k^2 \tilde\psi_k,
with the exact solution

    \tilde\psi_{tk} = K_{tk} \tilde\psi_k,      K_{tk} = e^{-i k_k^2 t_t / 2}       (no sum: elementwise)

The changes of basis are the discrete Fourier transform and its inverse (x and k summed):

    \tilde\psi_k = F_{kx} \psi_x(0),            \psi_{tx} = F^{-1}_{xk} \tilde\psi_{tk}

Centre and width of the packet, with \rho_{tx} = |\psi_{tx}|^2 dx:

    \langle x \rangle_t = \rho_{tx} x_x,         \sigma_t^2 = \rho_{tx} (x_x - \langle x \rangle_t)^2     (x summed)
"""

import torch


def kinetic_phase(times: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""K_{tk} = e^{-i k_k^2 t_t / 2}.   times: (T,), k: (N,) → (T, N) complex"""
    return torch.exp(-0.5j * torch.einsum("t,k->tk", times, k**2))


def evolve(psi0: torch.Tensor, times: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""\psi_{tx} = F^{-1}_{xk} K_{tk} F_{ky} \psi_y(0).   psi0: (N,) → (T, N) complex"""
    psi_k = torch.fft.fft(psi0)  # (N,)    x → k
    psi_tk = kinetic_phase(times, k) * psi_k  # (T, N)  broadcast over t: each mode gets its own phase
    return torch.fft.ifft(psi_tk, dim=-1)  # (T, N)  k → x for every t at once


def mean_and_width(psi: torch.Tensor, x: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""(\langle x \rangle_t, \sigma_t) of |\psi|^2.   psi: (T, N), x: (N,) → (T,), (T,)"""
    rho = psi.abs() ** 2 * dx  # (T, N) probability per grid cell
    mean = torch.einsum("tx,x->t", rho, x)  # (T,)
    spread = x[None, :] - mean[:, None]  # (T, N)
    width = torch.sqrt(torch.einsum("tx,tx->t", rho, spread**2))  # (T,)
    return mean, width


def norm(psi: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\int |\psi|^2 dx \approx \psi^*_x \psi_x dx  (x summed).   (..., N) → (...)"""
    return torch.einsum("...x,...x->...", psi.conj(), psi).real * dx
