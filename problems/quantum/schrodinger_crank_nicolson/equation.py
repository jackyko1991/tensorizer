r"""Physics → tensor translation.

Time-dependent Schrödinger equation (units ħ = m = 1) on N grid points:

    i \partial_t \psi_x = H_{xy} \psi_y,        H_{xy} = -\tfrac12 D2_{xy} + V_x \delta_{xy}      (y summed).

Several wave packets are evolved at once as the columns b of \psi_{xb}.

Crank–Nicolson averages the right-hand side over the old and the new time level:

    i (\psi^{n+1} - \psi^n)/\Delta t = \tfrac12 H (\psi^{n+1} + \psi^n)
    ⇔  A_{xy} \psi^{n+1}_{yb} = B_{xy} \psi^n_{yb},
        A = I + \tfrac{i\Delta t}{2} H,   B = I - \tfrac{i\Delta t}{2} H       (y summed).

U = A^{-1} B is the Cayley transform of the Hermitian H, which is exactly unitary, so the norm never drifts.
For comparison, explicit Euler takes \psi^{n+1} = (I - i\Delta t H)\psi^n, whose eigenvalues
1 - iE\Delta t all have modulus \sqrt{1 + E^2\Delta t^2} > 1: the norm grows every step.

    norm_b   = \psi^*_{xb} \psi_{xb} dx               (x summed)
    \langle H\rangle_b = \psi^*_{xb} H_{xy} \psi_{yb} dx     (x, y summed)
"""

import torch


def cayley_matrices(H: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""A = I + i\Delta t H/2,  B = I - i\Delta t H/2.   H: (N, N) real → two (N, N) complex matrices."""
    eye = torch.eye(len(H), dtype=torch.complex128, device=H.device)
    Hc = H.to(torch.complex128)
    return eye + 0.5j * dt * Hc, eye - 0.5j * dt * Hc


def apply(M: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""M_{xy} \psi_{yb}  (y summed).   M: (N, N), psi: (N, B) → (N, B)"""
    return torch.einsum("xy,yb->xb", M, psi)


def cn_step(lu: torch.Tensor, pivots: torch.Tensor, B: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""Solve A \psi^{n+1} = B \psi^n with the LU factors of A (computed once).   (N, B) → (N, B)"""
    return torch.linalg.lu_solve(lu, pivots, apply(B, psi))


def cn_step_solve(A: torch.Tensor, B: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""The same step with a fresh torch.linalg.solve (factorises A again every call).   (N, B) → (N, B)"""
    return torch.linalg.solve(A, apply(B, psi))


def euler_step(H: torch.Tensor, psi: torch.Tensor, dt: float) -> torch.Tensor:
    r"""Explicit Euler: \psi^{n+1} = \psi^n - i\Delta t H \psi^n.   (N, B) → (N, B)"""
    return psi - 1j * dt * apply(H.to(psi.dtype), psi)


def norm(psi: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\psi^*_{xb} \psi_{xb} dx  (x summed).   psi: (..., N, B) → (..., B)"""
    return torch.einsum("...xb,...xb->...b", psi.conj(), psi).real * dx


def energy(psi: torch.Tensor, H: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\langle H\rangle_b = \psi^*_{xb} H_{xy} \psi_{yb} dx  (x, y summed).   psi: (..., N, B) → (..., B)"""
    return torch.einsum("...xb,xy,...yb->...b", psi.conj(), H.to(psi.dtype), psi).real * dx


def mean_position(psi: torch.Tensor, x: torch.Tensor, dx: float) -> torch.Tensor:
    r"""\langle x\rangle_b = \psi^*_{xb} x_x \psi_{xb} dx  (x summed).   psi: (..., N, B) → (..., B)"""
    return torch.einsum("...xb,x,...xb->...b", psi.conj(), x.to(psi.dtype), psi).real * dx
