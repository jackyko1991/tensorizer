r"""Physics → tensor translation.

Give PyTorch a Lagrangian L(q, \dot q) = T − V and let automatic differentiation derive the equations of motion.
The Euler–Lagrange equations

    \frac{d}{dt}\frac{\partial L}{\partial \dot q_i} = \frac{\partial L}{\partial q_i}

expand by the chain rule (repeated j ⇒ summed) into

    H_{ij} \ddot q_j + C_{ij} \dot q_j = \frac{\partial L}{\partial q_i},
    \qquad H_{ij} = \frac{\partial^2 L}{\partial \dot q_i \partial \dot q_j},
    \qquad C_{ij} = \frac{\partial^2 L}{\partial \dot q_i \partial q_j}

so the acceleration is a linear solve:

    \ddot q_i = H^{-1}_{ij} \left( \frac{\partial L}{\partial q_j} - C_{jk} \dot q_k \right)

H is the (generalised) mass matrix. torch.autograd.grad gives every derivative: one call for ∂L/∂q and
p_i = ∂L/∂\dot q_i (keeping the graph), then one more per component p_i for row i of H and of C.
The conserved energy is

    E = \dot q_i \frac{\partial L}{\partial \dot q_i} - L .

Example: a spring (elastic) pendulum with q = (r, θ), mass m, spring k, natural length ℓ_0:

    L = \tfrac12 m (\dot r^2 + r^2 \dot\theta^2) + m g r \cos\theta - \tfrac12 k (r - \ell_0)^2
"""

from typing import Callable

import torch

Lagrangian = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # L(q (..., n), q_dot (..., n)) → (...)


def euler_lagrange(L: Lagrangian, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
    r"""\ddot q_i = H^{-1}_{ij}(∂L/∂q_j − C_{jk} \dot q_k), all derivatives by autograd.   (n,), (n,) → (n,)"""
    q = q.detach().requires_grad_(True)
    q_dot = q_dot.detach().requires_grad_(True)
    lagrangian = L(q, q_dot)  # scalar
    # first derivatives; create_graph=True so that we can differentiate them once more.
    # materialize_grads=True returns zeros (instead of an error) when L does not depend on a variable.
    dL_dq, p = torch.autograd.grad(lagrangian, (q, q_dot), create_graph=True, materialize_grads=True)  # (n,) each
    # second derivatives: row i of H and C is the gradient of p_i = ∂L/∂q̇_i
    rows = [torch.autograd.grad(p[i], (q_dot, q), retain_graph=True, materialize_grads=True) for i in range(len(p))]
    H = torch.stack([d_dq_dot for d_dq_dot, _ in rows])  # (n, n)  ∂²L/∂q̇_i∂q̇_j
    C = torch.stack([d_dq for _, d_dq in rows])  # (n, n)  ∂²L/∂q̇_i∂q_j
    q_ddot = torch.linalg.solve(H, dL_dq - torch.einsum("jk,k->j", C, q_dot))
    return q_ddot.detach()


def rhs(y: torch.Tensor, L: Lagrangian) -> torch.Tensor:
    """dy/dt for the state y of shape (2, n): y[0] = q, y[1] = q̇."""
    q, q_dot = y[0], y[1]
    return torch.stack([q_dot, euler_lagrange(L, q, q_dot)])


def energy(L: Lagrangian, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
    r"""E = \dot q_i ∂L/∂\dot q_i − L (i summed), by autograd, for many states at once.   (T, n), (T, n) → (T,)"""
    q_dot = q_dot.detach().requires_grad_(True)
    lagrangian = L(q.detach(), q_dot)  # (T,)
    (p,) = torch.autograd.grad(lagrangian.sum(), q_dot)  # (T, n) p_i = ∂L/∂q̇_i (states are independent)
    return (torch.einsum("ti,ti->t", q_dot, p) - lagrangian).detach()


def spring_pendulum(m: float, k: float, rest_length: float, g: float) -> Lagrangian:
    """L(q, q̇) for a mass on a spring swinging in a vertical plane; q = (r, θ), θ from the downward vertical."""

    def L(q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        r, theta = q[..., 0], q[..., 1]
        r_dot, theta_dot = q_dot[..., 0], q_dot[..., 1]
        kinetic = 0.5 * m * (r_dot**2 + r**2 * theta_dot**2)
        potential = -m * g * r * torch.cos(theta) + 0.5 * k * (r - rest_length) ** 2
        return kinetic - potential

    return L
