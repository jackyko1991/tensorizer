r"""Physics → tensor translation.

Viscous Burgers in conservation form, u_t + ∂_x F(u) = 0 with flux F = ½u² - ν u_x, on a periodic
domain. A finite-volume scheme stores cell averages u_i and updates them with the flux through each face:

    du_i/dt = -(F_{i+½} - F_{i-½}) / Δx

Whatever F_{i+½} is, the same number leaves cell i and enters cell i+1, so Σ_i u_i Δx is conserved exactly.
The plain scheme uses the Rusanov (local Lax–Friedrichs) flux plus a centred viscous flux:

    F_{i+½} = ½(½u_i² + ½u_{i+1}²) - ½ α_{i+½} (u_{i+1} - u_i) - ν (u_{i+1} - u_i)/Δx,   α = max(|u_i|, |u_{i+1}|)

The learned scheme adds a correction from a small convolutional network (in the spirit of Kochkov et al.,
PNAS 2021, who learned interpolation stencils for Navier–Stokes):

    F_{i+½} → F_{i+½} + δF_{i+½}[u],      δF = CNN(u):   h^{(1)}_{c,i} = σ(K^{(1)}_{c,k} u_{i+k} + β_c),   …

Each convolution sums over the input channel and the stencil offset k (the circular padding is the periodic
boundary). Because δF is a flux, the learned scheme conserves Σ_i u_i as exactly as the plain one does.
"""

import torch


def rusanov_flux(u: torch.Tensor) -> torch.Tensor:
    r"""F^{adv}_{i+½} = ¼(u_i² + u_{i+1}²) - ½ max(|u_i|, |u_{i+1}|)(u_{i+1} - u_i).

    u: (B, N) → (B, N); entry i is the flux through face i+½."""
    right = torch.roll(u, -1, dims=-1)  # u_{i+1} (periodic)
    alpha = torch.maximum(u.abs(), right.abs())
    return 0.25 * (u**2 + right**2) - 0.5 * alpha * (right - u)


def viscous_flux(u: torch.Tensor, nu: float, dx: float) -> torch.Tensor:
    r"""F^{visc}_{i+½} = -ν (u_{i+1} - u_i) / Δx.   (B, N) → (B, N)"""
    return -nu * (torch.roll(u, -1, dims=-1) - u) / dx


def rhs(u: torch.Tensor, nu: float, dx: float, correction=None) -> torch.Tensor:
    r"""du_i/dt = -(F_{i+½} - F_{i-½})/Δx with F = Rusanov + viscous (+ learned δF).   (B, N) → (B, N)"""
    F = rusanov_flux(u) + viscous_flux(u, nu, dx)
    if correction is not None:
        F = F + correction(u)
    return -(F - torch.roll(F, 1, dims=-1)) / dx


def step(u: torch.Tensor, dt: float, nu: float, dx: float, correction=None) -> torch.Tensor:
    """One Heun (2nd-order Runge–Kutta) step; with a correction it stays differentiable for training."""
    k1 = rhs(u, nu, dx, correction)
    k2 = rhs(u + dt * k1, nu, dx, correction)
    return u + 0.5 * dt * (k1 + k2)


def cell_average(u_fine: torch.Tensor, factor: int) -> torch.Tensor:
    r"""ū_I = (1/f) Σ_{i ∈ I} u_i: fine cells → coarse cells (f fine cells each).   (..., N_f) → (..., N_f / f)"""
    return u_fine.unflatten(-1, (-1, factor)).mean(-1)


def mean(u: torch.Tensor) -> torch.Tensor:
    r"""⟨u⟩ = (1/N) Σ_i u_i — conserved on a periodic domain.   (..., N) → (...)"""
    return u.mean(-1)
