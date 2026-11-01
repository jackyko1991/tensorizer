r"""Physics → tensor translation.

Time-independent Schrödinger equation for a particle in a box 0 < x < L (units ħ = m = 1):

    \hat H \varphi = -\tfrac12 \varphi'' = E \varphi,      \varphi(0) = \varphi(L) = 0.

On N grid points the operator becomes a matrix and the wavefunction a vector:

    H_{xy} \varphi_y = E \varphi_x,      H_{xy} = -\tfrac12 D2_{xy}      (y summed),

where D2 is the finite-difference second derivative. Collecting all N eigenvectors as the
columns of \varphi_{xn} (grid point x, state n) the whole eigenproblem is one line:

    H_{xy} \varphi_{yn} = E_n \varphi_{xn}.

Time evolution: each eigenstate only rotates its phase, so

    c_n = \varphi_{xn} \psi_x(0) \, dx                      (x summed: projection)
    \psi_x(t) = \varphi_{xn} \, c_n e^{-i E_n t}            (n summed: superposition)
"""

import torch


def project(phi: torch.Tensor, psi0: torch.Tensor, dx: float) -> torch.Tensor:
    r"""c_n = \varphi_{xn} \psi_x dx.   phi: (N, M), psi0: (N,) complex → (M,) complex"""
    return torch.einsum("xn,x->n", phi.to(psi0.dtype), psi0) * dx


def evolve(phi: torch.Tensor, energies: torch.Tensor, c: torch.Tensor, t: float) -> torch.Tensor:
    r"""\psi_x(t) = \varphi_{xn} c_n e^{-i E_n t}.   phi: (N, M), energies: (M,), c: (M,) → (N,) complex"""
    phase = torch.exp(-1j * energies * t)  # (M,)  e^{-i E_n t}
    return torch.einsum("xn,n->x", phi.to(c.dtype), c * phase)


def norm(psi: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\int |\psi|^2 dx \approx \psi^*_x \psi_x dx  (x summed).   (..., N) → (...)"""
    return torch.einsum("...x,...x->...", psi.conj(), psi).real * dx
