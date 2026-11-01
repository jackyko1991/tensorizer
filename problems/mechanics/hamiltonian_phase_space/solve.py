"""Three time steppers for Hamilton's equations: symplectic leapfrog, explicit Euler and RK4.

All take a batch of phase-space points z of shape (..., 2n) and get ∂H/∂z from autograd.
"""

from typing import Callable

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import Hamiltonian, grad_H, hamilton_rhs

Stepper = Callable[[Hamiltonian, torch.Tensor, float], torch.Tensor]


def leapfrog_step(H: Hamiltonian, z: torch.Tensor, dt: float) -> torch.Tensor:
    """Störmer–Verlet (kick–drift–kick) for a separable H = T(p) + V(q):

    p ← p − ½Δt ∂H/∂q,   q ← q + Δt ∂H/∂p,   p ← p − ½Δt ∂H/∂q.
    Each sub-step is a shear of phase space, so the whole step keeps area exactly (it is symplectic).
    """
    n = z.shape[-1] // 2
    q, p = z[..., :n], z[..., n:]
    p = p - 0.5 * dt * grad_H(H, torch.cat([q, p], -1))[..., :n]  # kick
    q = q + dt * grad_H(H, torch.cat([q, p], -1))[..., n:]  # drift
    p = p - 0.5 * dt * grad_H(H, torch.cat([q, p], -1))[..., :n]  # kick
    return torch.cat([q, p], -1)


def euler_step(H: Hamiltonian, z: torch.Tensor, dt: float) -> torch.Tensor:
    """Explicit Euler z ← z + Δt ż: stretches phase space a little every step, so energy grows."""
    return z + dt * hamilton_rhs(H, z)


def rk4_hamilton_step(H: Hamiltonian, z: torch.Tensor, dt: float) -> torch.Tensor:
    """Classical 4th-order Runge–Kutta: very accurate per step, but not symplectic (slow energy drift)."""
    return rk4_step(lambda y, t: hamilton_rhs(H, y), z, 0.0, dt)


STEPPERS: dict[str, Stepper] = {"leapfrog": leapfrog_step, "RK4": rk4_hamilton_step, "Euler": euler_step}


def integrate(stepper: Stepper, H: Hamiltonian, z0: torch.Tensor, dt: float, n_steps: int, label: str = "steps"):
    """Whole trajectory, shape (n_steps + 1, *z0.shape)."""
    zs = [z0]
    z = z0
    for _ in steps(n_steps, label):
        z = stepper(H, z, dt)
        zs.append(z)
    return torch.stack(zs)


def solve(H: Hamiltonian, z0: dict[str, torch.Tensor], dt: float, t_max: float) -> dict:
    """Run every initial-condition set with every integrator, all points of a set as one batch.

    z0 maps a name ("orbits", "test", "blob") to a (B, 2) tensor; each set is stacked into one batch
    per integrator so a single step moves all B·(sets) points. Returns times (T,) and
    traj[integrator][name] of shape (T, B, 2).
    """
    n_steps = int(round(t_max / dt))
    names = list(z0)
    sizes = [z0[k].shape[0] for k in names]
    batch = torch.cat([z0[k] for k in names])  # (ΣB, 2) every point in one tensor
    traj = {}
    for label, stepper in STEPPERS.items():
        path = integrate(stepper, H, batch, dt, n_steps, f"{label} steps")  # (T, ΣB, 2)
        traj[label] = dict(zip(names, torch.split(path, sizes, dim=1)))
    times = torch.arange(n_steps + 1, dtype=batch.dtype, device=batch.device) * dt
    return {"times": times, "traj": traj}


def small_oscillation_period(H: Hamiltonian, amplitude: float, t_max: float, dt: float = 0.01, device=None) -> float:
    """Release one pendulum from rest at a small angle (radians), leapfrog it and measure its period."""
    z0 = torch.tensor([[amplitude, 0.0]], dtype=torch.float64, device=device)
    n_steps = int(round(t_max / dt))
    q = integrate(leapfrog_step, H, z0, dt, n_steps, "small swing")[:, 0, 0]  # (T,)
    return period(torch.arange(n_steps + 1, dtype=q.dtype, device=q.device) * dt, q)


def period(times: torch.Tensor, q: torch.Tensor) -> float:
    """Mean period of q(t) from its downward zero crossings (linear interpolation between steps)."""
    down = torch.nonzero((q[:-1] > 0) & (q[1:] <= 0)).flatten()  # steps just before a crossing
    t_cross = times[down] + (times[1] - times[0]) * q[down] / (q[down] - q[down + 1])
    return float((t_cross[-1] - t_cross[0]) / (len(t_cross) - 1))
