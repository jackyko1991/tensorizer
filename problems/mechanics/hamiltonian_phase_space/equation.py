r"""Physics → tensor translation.

Hamilton's equations for coordinates q_i and momenta p_i,

    \dot q_i = \frac{\partial H}{\partial p_i}, \qquad \dot p_i = -\frac{\partial H}{\partial q_i},

become one equation for the phase-space point z = (q, p) with the symplectic matrix J = [[0, I], [−I, 0]]:

    \dot z_a = J_{ab} \frac{\partial H}{\partial z_b}        (repeated b ⇒ summed)

H is typed in once as a function; torch.autograd supplies ∂H/∂z_b. For the pendulum (q = angle from the
downward vertical, p = angular momentum)

    H(q, p) = \frac{p^2}{2 m L^2} - m g L \cos q ,

and the separatrix H = mgL divides swinging back and forth (libration) from going over the top (rotation).
Liouville's theorem: the flow keeps phase-space area, measured here with the shoelace formula

    A = \tfrac12 \sum_k (q_k p_{k+1} - q_{k+1} p_k)        (k runs around the blob's boundary).
"""

from typing import Callable

import torch

Hamiltonian = Callable[[torch.Tensor], torch.Tensor]  # H(z (..., 2n)) → (...)


def pendulum(m: float, L: float, g: float) -> Hamiltonian:
    """H(q, p) = p²/(2mL²) − mgL cos q, with z[..., 0] = q and z[..., 1] = p."""

    def H(z: torch.Tensor) -> torch.Tensor:
        q, p = z[..., 0], z[..., 1]
        return p**2 / (2 * m * L**2) - m * g * L * torch.cos(q)

    return H


def harmonic(m: float, L: float, g: float) -> Hamiltonian:
    """The small-angle pendulum H = p²/(2mL²) + ½ mgL q² (constant −mgL dropped): a harmonic oscillator."""

    def H(z: torch.Tensor) -> torch.Tensor:
        q, p = z[..., 0], z[..., 1]
        return p**2 / (2 * m * L**2) + 0.5 * m * g * L * q**2

    return H


def symplectic_matrix(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """J = [[0, I], [−I, 0]] of shape (2n, 2n)."""
    eye = torch.eye(n, dtype=dtype, device=device)
    zero = torch.zeros(n, n, dtype=dtype, device=device)
    return torch.cat([torch.cat([zero, eye], dim=1), torch.cat([-eye, zero], dim=1)], dim=0)


def grad_H(H: Hamiltonian, z: torch.Tensor) -> torch.Tensor:
    """∂H/∂z_b by autograd for a whole batch of points at once.   (..., 2n) → (..., 2n)

    The points are independent, so the gradient of the sum ΣH gives every point's own gradient.
    """
    z = z.detach().requires_grad_(True)
    (dH,) = torch.autograd.grad(H(z).sum(), z)
    return dH


def hamilton_rhs(H: Hamiltonian, z: torch.Tensor) -> torch.Tensor:
    r"""\dot z_a = J_{ab} ∂H/∂z_b (b summed).   (..., 2n) → (..., 2n)"""
    J = symplectic_matrix(z.shape[-1] // 2, z.dtype, z.device)  # (2n, 2n)
    return torch.einsum("ab,...b->...a", J, grad_H(H, z))


def pendulum_rhs_by_hand(z: torch.Tensor, m: float, L: float, g: float) -> torch.Tensor:
    """The same equations derived on paper: q̇ = p/(mL²), ṗ = −mgL sin q (for checking autograd)."""
    q, p = z[..., 0], z[..., 1]
    return torch.stack([p / (m * L**2), -m * g * L * torch.sin(q)], dim=-1)


def separatrix_momentum(q: torch.Tensor, m: float, L: float, g: float) -> torch.Tensor:
    """Upper branch of H = mgL:  p = 2 m L √(gL) cos(q/2).   (K,) → (K,)"""
    return 2 * m * L * (g * L) ** 0.5 * torch.cos(q / 2)


def shoelace_area(z: torch.Tensor) -> torch.Tensor:
    r"""Area enclosed by a closed polygon of phase-space points: (..., K, 2) → (...).

    A = ½ Σ_k (q_k p_{k+1} − q_{k+1} p_k); torch.roll gives the next point k + 1 (wrapping K → 0).
    """
    q, p = z[..., 0], z[..., 1]
    q_next, p_next = torch.roll(q, -1, dims=-1), torch.roll(p, -1, dims=-1)
    return 0.5 * (q * p_next - q_next * p).sum(dim=-1)
