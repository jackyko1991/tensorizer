r"""Physics → tensor translation.

Time-dependent Schrödinger equation in 2D (units ħ = m = 1):

    i \partial_t \psi = \Big(-\tfrac12 (\partial_x^2 + \partial_y^2) + V(x, y)\Big) \psi.

Exactly as in 1D, Strang splitting turns one time step into three diagonal operators:

    potential (x-space):  \psi_{xy} \leftarrow P_{xy} \psi_{xy},          P_{xy} = e^{-i V_{xy} \Delta t/2}
    kinetic (k-space):    \tilde\psi_{pq} \leftarrow K_{pq} \tilde\psi_{pq},  K_{pq} = e^{-i (k_p^2 + k_q^2) \Delta t/2}

No index is summed in either line: a diagonal operator D_{xy,x'y'} = d_{xy} \delta_{xx'} \delta_{yy'} is just an
elementwise multiply. The 2D Fourier transform sums over both grid indices,

    \tilde\psi_{pq} = F_{px} F_{qy} \psi_{xy}          (x and y summed: torch.fft.fft2)

and factorises into one 1D transform per axis. Finally an absorbing mask M_{xy} \le 1 damps ψ near the
grid edges after every step.
"""

import torch


def potential_phase(V: torch.Tensor, dt: float) -> torch.Tensor:
    r"""P_{xy} = e^{-i V_{xy} \Delta t/2}.   V: (N, N) → (N, N) complex"""
    return torch.exp(-0.5j * V * dt)


def kinetic_phase(KX: torch.Tensor, KY: torch.Tensor, dt: float) -> torch.Tensor:
    r"""K_{pq} = e^{-i (k_p^2 + k_q^2) \Delta t/2}.   KX, KY: (N, N) → (N, N) complex"""
    return torch.exp(-0.5j * (KX**2 + KY**2) * dt)


def split_step(psi: torch.Tensor, P: torch.Tensor, K: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    r"""One Strang step  \psi \to M P F^{-1} K F P \psi.   All tensors (N, N)."""
    psi = P * psi  # half potential step, diagonal in x-space
    psi_k = torch.fft.fft2(psi)  # (x, y) → (k_x, k_y)
    psi_k = K * psi_k  # full kinetic step, diagonal in k-space
    psi = torch.fft.ifft2(psi_k)  # back to (x, y)
    return M * (P * psi)  # second half potential step, then absorb at the edges


def probability(psi: torch.Tensor, dA: float) -> torch.Tensor:
    r"""\int |\psi|^2 dA \approx \psi^*_{xy} \psi_{xy} dA  (x and y summed).   (..., N, N) → (...)"""
    return torch.einsum("...xy,...xy->...", psi.conj(), psi).real * dA
