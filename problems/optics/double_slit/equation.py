r"""Physics → tensor translation.

The scalar wave equation in two dimensions,

    \partial_t^2 u = c^2 \nabla^2 u,     index form   \partial_t^2 u = c^2\, \partial_i \partial_i u

(repeated i ⇒ summed over i ∈ {x, y}). On a grid with spacing h the Laplacian becomes a 3×3 stencil
S_{kl} slid over the field — exactly what a 2D convolution does:

    (\partial_i\partial_i u)_{ab} \approx \frac{1}{h^2} \sum_{k,l} S_{kl}\, u_{a+k,\,b+l},
    S = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

Time uses the centred second difference (leapfrog). With a damping rate σ_{ab} (non-zero only in a
"sponge" frame round the edge, so outgoing waves are absorbed instead of reflected), the equation
\partial_t^2 u + σ \partial_t u = c^2 \nabla^2 u becomes

    (1 + σΔt/2)\, u^{n+1} = 2u^n - (1 - σΔt/2)\, u^{n-1} + (cΔt)^2 \nabla^2 u^n

This explicit scheme is stable only if the Courant number cΔt/h ≤ 1/\sqrt{2} (the CFL condition in 2D).
"""

import math

import torch
import torch.nn.functional as F

# S_{kl}: shape (3, 3). Centre weight −4, the four nearest neighbours +1.
LAPLACIAN_STENCIL = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)

CFL_LIMIT = 1 / math.sqrt(2)  # largest stable cΔt/h for the 5-point stencil in 2D


def laplacian(u: torch.Tensor, h: float) -> torch.Tensor:
    r"""(\partial_i\partial_i u)_{ab} = \sum_{kl} S_{kl} u_{a+k,b+l} / h^2.   u: (Ny, Nx) → (Ny, Nx)

    conv2d wants (batch, channel, H, W) tensors, hence the [None, None]; padding=1 treats the
    field outside the grid as zero.
    """
    kernel = LAPLACIAN_STENCIL.to(u)[None, None]  # (1, 1, 3, 3)
    return F.conv2d(u[None, None], kernel, padding=1)[0, 0] / h**2


def courant(c: float, dt: float, h: float) -> float:
    """Courant number cΔt/h: how many cells a wave crosses in one time step."""
    return c * dt / h


def step(u: torch.Tensor, u_prev: torch.Tensor, sigma: torch.Tensor, c: float, dt: float, h: float) -> torch.Tensor:
    r"""One leapfrog step  u^{n-1}, u^n → u^{n+1}  with sponge damping σ_{ab}.   All tensors (Ny, Nx)."""
    damp = 0.5 * sigma * dt  # (Ny, Nx)
    return (2 * u - (1 - damp) * u_prev + (c * dt) ** 2 * laplacian(u, h)) / (1 + damp)
