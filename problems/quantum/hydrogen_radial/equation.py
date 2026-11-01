r"""Physics → tensor translation.

The hydrogen atom in atomic units, -\tfrac12 \nabla^2 \psi - \psi / r = E \psi, separates as
\psi_{nlm} = R_{nl}(r) Y_{lm}(\theta, \varphi). With u(r) = r R(r) the radial part is a 1D problem

    -\tfrac12 u'' + V_l(r) u = E u,      V_l(r) = \frac{l(l+1)}{2 r^2} - \frac{1}{r},      u(0) = u(\infty) = 0.

On N radial grid points, and for L values of l at once, it is a stack of matrix eigenproblems:

    V_{li} = \frac{l(l+1)}{2 r_i^2} - \frac{1}{r_i}
    H_{lij} = -\tfrac12 D2_{ij} + V_{li} \delta_{ij}
    H_{lij} u_{ljn} = E_{ln} u_{lin}                     (j summed; l is a batch index, never summed)

The radial probability is P(r) = r^2 |R|^2 = u^2, and for m = 0 the angular part is
Y_{l0}(\theta) = \sqrt{(2l+1)/4\pi} \, P_l(\cos\theta) with the Legendre polynomial P_l.
"""

import math

import torch


def effective_potential(r: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    r"""V_{li} = l(l+1)/(2 r_i^2) - 1/r_i.   r: (N,), ls: (L,) → (L, N)"""
    centrifugal = torch.einsum("l,i->li", ls * (ls + 1) / 2, 1 / r**2)  # (L, N) outer product
    return centrifugal - 1 / r  # Coulomb part broadcast over l


def hamiltonian(D2: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    r"""H_{lij} = -\tfrac12 D2_{ij} + V_{li} \delta_{ij}.   D2: (N, N), V: (L, N) → (L, N, N)"""
    return -0.5 * D2 + torch.diag_embed(V)  # diag_embed puts each row V_l on the diagonal of its own matrix


def radial_probability(u: torch.Tensor) -> torch.Tensor:
    r"""P(r) = r^2 |R(r)|^2 = u(r)^2.   (..., N) → (..., N)"""
    return u**2


def spherical_harmonic_m0(ell: int, cos_theta: torch.Tensor) -> torch.Tensor:
    r"""Y_{l0} = \sqrt{(2l+1)/4\pi} P_l(\cos\theta).   cos_theta: any shape → same shape"""
    return math.sqrt((2 * ell + 1) / (4 * math.pi)) * torch.special.legendre_polynomial_p(cos_theta, ell)


def orbital_density(R: torch.Tensor, ell: int, cos_theta: torch.Tensor) -> torch.Tensor:
    r"""|\psi_{nl0}|^2 = |R_{nl}(r) Y_{l0}(\theta)|^2.   R, cos_theta: (P, P) → (P, P)"""
    return (R * spherical_harmonic_m0(ell, cos_theta)) ** 2
