r"""Physics → tensor translation.

A clamped membrane obeys the 2D wave equation ∂_t² u = c² ∇² u. Standing waves u = φ(x, y) cos(ωt) turn it
into an eigenvalue problem, −∇²φ = (ω/c)² φ.

On an n × n grid the point (a, b) gets the single index p = a·n + b, and the 2D Laplacian is a Kronecker sum
of the 1D second-difference matrix D:

    \Delta_{(ab)(cd)} = D_{ac} \delta_{bd} + \delta_{ac} D_{bd}        i.e.   Δ = D ⊗ I + I ⊗ D

The modes are the eigenvectors of −Δ (q summed):

    -\Delta_{pq} \phi_{qm} = (\omega_m / c)^2 \phi_{pm},        \phi_{pm} \phi_{pn} = \delta_{mn}    (p summed)
"""

import torch


def laplacian_2d(D: torch.Tensor) -> torch.Tensor:
    """Δ = D ⊗ I + I ⊗ D.   D: (n, n) → (n², n²)"""
    eye = torch.eye(D.shape[0], dtype=D.dtype, device=D.device)
    return torch.kron(D, eye) + torch.kron(eye, D)


def restrict(lap: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    """Δ_{pq} for the points inside the drum only (outside points are clamped to 0).   inside: (n²,) bool → (P, P)"""
    return lap[inside][:, inside]


def modes(lap: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Angular frequencies ω_m (P,) lowest first and orthonormal mode shapes φ_{pm} (P, P) from eigh(−Δ)."""
    lam, phi = torch.linalg.eigh(-lap)  # eigh sorts the eigenvalues in ascending order
    return c * torch.sqrt(lam), phi


def overlap(phi: torch.Tensor) -> torch.Tensor:
    """φ_{pm} φ_{pn} — p summed. Orthonormal modes give the identity.   (P, M) → (M, M)"""
    return torch.einsum("pm,pn->mn", phi, phi)


def displacement(shapes: torch.Tensor, omega: torch.Tensor, t: float) -> torch.Tensor:
    """u_m(x, y, t) = φ_m(x, y) cos(ω_m t) for every shown mode.   shapes: (M, n, n), omega: (M,) → (M, n, n)"""
    return shapes * torch.cos(omega * t)[:, None, None]
