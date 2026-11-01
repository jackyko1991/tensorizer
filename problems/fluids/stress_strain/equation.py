r"""Physics → tensor translation.

Linear elasticity of a solid. The displacement field u_i(x) gives the (small) strain tensor

    ε_{ij} = ½ (∂_i u_j + ∂_j u_i)

and Hooke's law links strain to stress through the rank-4 stiffness tensor (k, l summed):

    σ_{ij} = C_{ijkl} ε_{kl},      C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})

For an isotropic material this collapses to σ_{ij} = λ ε_{kk} δ_{ij} + 2 μ ε_{ij}.
The principal stresses are the eigenvalues of the symmetric matrix σ_{ij} (torch.linalg.eigh);
Mohr's circle draws the in-plane stress state as a circle of centre (σ_xx+σ_yy)/2.
"""

import torch


def lame(E: float, nu: float) -> tuple[float, float]:
    """Lamé constants from Young's modulus E and Poisson's ratio ν."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def stiffness(lam: float, mu: float, delta: torch.Tensor) -> torch.Tensor:
    r"""C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk}).   delta: (3, 3) → (3, 3, 3, 3)"""
    return (
        lam * torch.einsum("ij,kl->ijkl", delta, delta)
        + mu * torch.einsum("ik,jl->ijkl", delta, delta)
        + mu * torch.einsum("il,jk->ijkl", delta, delta)
    )


def strain(grad_u: torch.Tensor) -> torch.Tensor:
    r"""ε_{ij} = ½ (∂_i u_j + ∂_j u_i).   grad_u[..., i, j] = ∂_i u_j: (..., 3, 3) → (..., 3, 3)"""
    return 0.5 * (grad_u + grad_u.transpose(-1, -2))


def stress(C: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""σ_{ij} = C_{ijkl} ε_{kl}   (k, l summed).   (3,3,3,3), (..., 3, 3) → (..., 3, 3)"""
    return torch.einsum("ijkl,...kl->...ij", C, eps)


def lame_stress(eps: torch.Tensor, lam: float, mu: float, delta: torch.Tensor) -> torch.Tensor:
    r"""The same law written out: σ_{ij} = λ ε_{kk} δ_{ij} + 2 μ ε_{ij}."""
    trace = torch.einsum("...kk->...", eps)  # ε_kk, repeated k ⇒ summed
    return lam * torch.einsum("...,ij->...ij", trace, delta) + 2 * mu * eps


def principal_stresses(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (ascending) and eigenvectors (columns) of the symmetric stress tensor."""
    return torch.linalg.eigh(sigma)


def mohr_circle(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""In-plane Mohr circle: centre (σ_xx + σ_yy)/2 and radius \sqrt{((σ_xx - σ_yy)/2)^2 + σ_xy^2}."""
    sxx, syy, sxy = sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 0, 1]
    return 0.5 * (sxx + syy), torch.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
