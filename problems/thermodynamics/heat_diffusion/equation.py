r"""Physics → tensor translation.

Fourier's law says heat flows down the temperature gradient, and energy conservation says
the temperature of a cell changes by the net heat flowing into it:

    \mathbf q = -k \nabla T,        \rho c \, \partial_t T = -\nabla \cdot \mathbf q

In index notation, with i ∈ {x, y} the spatial direction:

    q_i = -k \, \partial_i T,        \partial_t T = -\frac{1}{\rho c} \partial_i q_i = \alpha \, \partial_i \partial_i T

The repeated index i in \partial_i q_i is summed: \partial_x q_x + \partial_y q_y, the divergence.
Putting the two together gives \partial_i \partial_i T = \nabla^2 T, the Laplacian, with α = k / (ρc).

On a grid the derivatives become finite-difference stencils, i.e. small convolution kernels:

    gradient    ∂_i T  →  conv2d(T, GRAD)   1 input channel  → 2 output channels (one per i)
    divergence  ∂_i q_i →  conv2d(q, DIV)    2 input channels → 1 output channel

conv2d sums over its input channels, so the second convolution performs the sum over i for us.
"""

import torch
import torch.nn.functional as F


def gradient_kernel(dx: float) -> torch.Tensor:
    """Forward differences  ∂_x T ≈ (T[y, x+1] − T[y, x]) / dx  (same for y).  Shape (2, 1, 3, 3)."""
    k = torch.zeros(2, 1, 3, 3)
    k[0, 0, 1, 1], k[0, 0, 1, 2] = -1.0, 1.0  # i = x: neighbour to the right
    k[1, 0, 1, 1], k[1, 0, 2, 1] = -1.0, 1.0  # i = y: neighbour below (next row)
    return k / dx


def divergence_kernel(dx: float) -> torch.Tensor:
    """Backward differences  ∂_x q_x ≈ (q_x[y, x] − q_x[y, x−1]) / dx.  Shape (1, 2, 3, 3)."""
    k = torch.zeros(1, 2, 3, 3)
    k[0, 0, 1, 0], k[0, 0, 1, 1] = -1.0, 1.0  # i = x: neighbour to the left
    k[0, 1, 0, 1], k[0, 1, 1, 1] = -1.0, 1.0  # i = y: neighbour above
    return k / dx


def pad_boundary(T: torch.Tensor, boundary: str, T_edge: float) -> torch.Tensor:
    """Add two layers of ghost cells around T (1, 1, H, W) → (1, 1, H+4, W+4).

    dirichlet: ghost cells held at T_edge (the edges are a cold bath).
    neumann:   ghost cells copy the edge cells, so ∂_i T = 0 there and no heat leaks out.
    """
    if boundary == "dirichlet":
        return F.pad(T, (2, 2, 2, 2), mode="constant", value=T_edge)
    return F.pad(T, (2, 2, 2, 2), mode="replicate")


def heat_flux(T_padded: torch.Tensor, k: float, grad: torch.Tensor) -> torch.Tensor:
    r"""q_i = -k ∂_i T.   (1, 1, H+4, W+4) → (1, 2, H+2, W+2): channel i is the flux component."""
    return -k * F.conv2d(T_padded, grad)


def divergence(q: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    r"""∂_i q_i, summed over i by conv2d.   (1, 2, H+2, W+2) → (1, 1, H, W)."""
    return F.conv2d(q, div)


def dT_dt(T: torch.Tensor, k: float, rho_c: float, grad, div, boundary: str, T_edge: float) -> torch.Tensor:
    r"""∂_t T = -∂_i q_i / (ρc)   (for constant k this is α ∂_i ∂_i T, the 5-point Laplacian)."""
    q = heat_flux(pad_boundary(T, boundary, T_edge), k, grad)
    return -divergence(q, div) / rho_c
