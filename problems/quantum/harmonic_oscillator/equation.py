r"""Physics → tensor translation.

Time-independent Schrödinger equation for a mass on a spring (units ħ = m = 1):

    \hat H \varphi = -\tfrac12 \varphi'' + \tfrac12 \omega^2 x^2 \varphi = E \varphi.

On N grid points the operator becomes a matrix and the wavefunction a vector:

    H_{xy} = -\tfrac12 D2_{xy} + V_x \delta_{xy},      V_x = \tfrac12 \omega^2 x_x^2,

and collecting all eigenvectors as the columns of \varphi_{xn} (grid point x, state n):

    H_{xy} \varphi_{yn} = E_n \varphi_{xn}             (y summed).

Time evolution: each eigenstate only rotates its phase, so

    c_n = \varphi_{xn} \psi_x(0) \, dx                  (x summed: projection)
    \psi_x(t) = \varphi_{xn} \, c_n e^{-i E_n t}        (n summed: superposition)
    \langle x \rangle = \psi^*_x \, x_x \, \psi_x \, dx    (x summed)
"""

import torch


def project(phi: torch.Tensor, psi0: torch.Tensor, dx: float) -> torch.Tensor:
    r"""c_n = \varphi_{xn} \psi_x dx.   phi: (N, M), psi0: (N,) complex → (M,) complex"""
    return torch.einsum("xn,x->n", phi.to(psi0.dtype), psi0) * dx


def evolve(phi: torch.Tensor, energies: torch.Tensor, c: torch.Tensor, t: float) -> torch.Tensor:
    r"""\psi_x(t) = \varphi_{xn} c_n e^{-i E_n t}.   phi: (N, M), energies: (M,), c: (M,) → (N,) complex"""
    phase = torch.exp(-1j * energies * t)  # (M,)  e^{-i E_n t}
    return torch.einsum("xn,n->x", phi.to(c.dtype), c * phase)


def mean_position(psi: torch.Tensor, x: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\langle x \rangle = \psi^*_x x_x \psi_x dx  (x summed).   psi: (..., N), x: (N,) → (...)"""
    return torch.einsum("...x,x,...x->...", psi.conj(), x.to(psi.dtype), psi).real * dx


def norm(psi: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\int |\psi|^2 dx \approx \psi^*_x \psi_x dx  (x summed).   (..., N) → (...)"""
    return torch.einsum("...x,...x->...", psi.conj(), psi).real * dx
