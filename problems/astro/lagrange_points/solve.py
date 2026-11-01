"""Find the Lagrange points with Newton's method, then integrate test particles near L4 with RK4."""

import torch

from problems._common.integrators import integrate

from .equation import grad_potential, hessian_potential, levi_civita, rhs


def lagrange_points(guess: torch.Tensor, mu: float, iterations: int = 30) -> torch.Tensor:
    """Newton iterations x ← x − H⁻¹∇Φ_eff for all five points at once: (5, 3) → (5, 3)."""
    x = guess.clone()
    for _ in range(iterations):
        g = grad_potential(x, mu)  # (5, 3)
        H = hessian_potential(x, mu)  # (5, 3, 3)
        x = x - torch.linalg.solve(H, g)  # one batched linear solve per point
    return x


def solve(y0: torch.Tensor, mu: float, t_max: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2, 3)) in the rotating frame."""
    eps = levi_civita(y0.dtype, y0.device)
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, mu, eps), y0, dt, n_steps)
    times = torch.arange(n_steps + 1, dtype=y0.dtype, device=y0.device) * dt
    return times, traj
