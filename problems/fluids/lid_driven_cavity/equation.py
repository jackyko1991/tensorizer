r"""Physics → tensor translation.

2D incompressible Navier–Stokes in streamfunction–vorticity form. With a streamfunction ψ,

    u = ∂_y ψ,   v = -∂_x ψ     ⇒   ∂_i u_i = 0 automatically,

and the vorticity ω = ∂_x v - ∂_y u obeys a transport equation plus a Poisson equation for ψ:

    ∂_t ω + u_j ∂_j ω = ν ∂_j ∂_j ω          (j summed: advection and diffusion)
    ∂_j ∂_j ψ = -ω

On the grid f[i, j] = f(x_i, y_j) the derivatives are central differences with spacing h.
The Poisson equation is solved with the eigenvectors of the 1D Laplacian matrix T
(T = V Λ V^T, see solve.py): ∇²ψ = T ψ + ψ T^T becomes diagonal, (Λ_a + Λ_b) \hat ψ_{ab} = \hat f_{ab}.
"""

import torch


def ddx(f: torch.Tensor, h: float) -> torch.Tensor:
    """Central ∂_x on interior points: (N, N) → (N-2, N-2)."""
    return (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * h)


def ddy(f: torch.Tensor, h: float) -> torch.Tensor:
    """Central ∂_y on interior points: (N, N) → (N-2, N-2)."""
    return (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * h)


def laplacian(f: torch.Tensor, h: float) -> torch.Tensor:
    r"""∂_j ∂_j f, five-point stencil on interior points: (N, N) → (N-2, N-2)."""
    return (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]) / h**2


def velocity(psi: torch.Tensor, h: float) -> torch.Tensor:
    """u_i = (∂_y ψ, -∂_x ψ) on interior points: (N, N) → (2, N-2, N-2)."""
    return torch.stack([ddy(psi, h), -ddx(psi, h)])


def divergence(u: torch.Tensor, h: float) -> torch.Tensor:
    """∂_i u_i by central differences: (2, M, M) → (M-2, M-2). Zero to round-off for u from a streamfunction."""
    return (u[0, 2:, 1:-1] - u[0, :-2, 1:-1] + u[1, 1:-1, 2:] - u[1, 1:-1, :-2]) / (2 * h)


def vorticity_rhs(omega: torch.Tensor, psi: torch.Tensor, h: float, nu: float) -> torch.Tensor:
    r"""∂_t ω = -u_j ∂_j ω + ν ∂_j ∂_j ω  (j summed by einsum).   (N, N) fields → (N-2, N-2)."""
    u = velocity(psi, h)  # (2, N-2, N-2)
    grad_omega = torch.stack([ddx(omega, h), ddy(omega, h)])  # (2, N-2, N-2)
    advection = torch.einsum("jxy,jxy->xy", u, grad_omega)  # u_j ∂_j ω
    return -advection + nu * laplacian(omega, h)


def wall_vorticity(omega: torch.Tensor, psi: torch.Tensor, h: float, lid: float) -> torch.Tensor:
    """Thom's formula: vorticity on the walls from ψ next to them (ψ = 0 on the walls). Returns a new ω.

    Bottom, left, right walls are at rest: ω_wall = -2 ψ_next / h².  The top lid moves at speed U:
    ω_top = -2 ψ_next / h² - 2 U / h.
    """
    omega = omega.clone()
    omega[:, 0] = -2 * psi[:, 1] / h**2  # bottom
    omega[0, :] = -2 * psi[1, :] / h**2  # left
    omega[-1, :] = -2 * psi[-2, :] / h**2  # right
    omega[:, -1] = -2 * psi[:, -2] / h**2 - 2 * lid / h  # top (moving lid)
    return omega
