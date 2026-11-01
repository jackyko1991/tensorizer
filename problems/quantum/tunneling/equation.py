r"""Physics → tensor translation.

Time-dependent Schrödinger equation in 1D (units ħ = m = 1):

    i \partial_t \psi = (\hat T + \hat V) \psi,      \hat T = -\tfrac12 \partial_x^2,   \hat V = V(x).

The exact step \psi(t + \Delta t) = e^{-i(\hat T + \hat V)\Delta t} \psi(t) is hard because T and V do not
commute. Strang splitting approximates it (error O(\Delta t^3) per step) by

    e^{-i\hat V \Delta t/2} \; e^{-i\hat T \Delta t} \; e^{-i\hat V \Delta t/2}.

Each factor is DIAGONAL in its own basis, and a diagonal operator D_{xy} = d_x \delta_{xy} acts as
D_{xy}\psi_y = d_x \psi_x (no sum left): an elementwise multiply.

    potential, in x-space:   \psi_x \leftarrow P_x \psi_x,            P_x = e^{-i V_x \Delta t / 2}
    kinetic,   in k-space:   \tilde\psi_k \leftarrow K_k \tilde\psi_k,  K_k = e^{-i k^2 \Delta t / 2}

The basis change \tilde\psi_k = F_{kx}\psi_x (x summed) is the discrete Fourier transform, done by torch.fft.
"""

import torch


def potential_phase(V: torch.Tensor, dt: float) -> torch.Tensor:
    r"""P_x = e^{-i V_x \Delta t/2}.   V: (N,) → (N,) complex"""
    return torch.exp(-0.5j * V * dt)


def kinetic_phase(k: torch.Tensor, dt: float) -> torch.Tensor:
    r"""K_k = e^{-i (k^2/2) \Delta t}.   k: (N,) → (N,) complex"""
    return torch.exp(-0.5j * k**2 * dt)


def split_step(psi: torch.Tensor, P: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    r"""One Strang step  \psi \to P \, F^{-1} K F \, P \, \psi.   All tensors (N,) complex."""
    psi = P * psi  # half potential step, diagonal in x
    psi_k = torch.fft.fft(psi)  # x → k:  \tilde\psi_k = F_{kx} \psi_x
    psi_k = K * psi_k  # full kinetic step, diagonal in k
    psi = torch.fft.ifft(psi_k)  # k → x
    return P * psi  # second half potential step


def probability(psi: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\int |\psi|^2 dx \approx \psi^*_x \psi_x dx  (x summed).   (..., N) → (...)"""
    return torch.einsum("...x,...x->...", psi.conj(), psi).real * dx
