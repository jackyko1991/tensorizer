"""Time stepping (RK4): a batch of couplings at once, a slow ramp of K, and the chimera ring."""

import math

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from . import tensors
from .equation import order_parameter, ring_kernel, ring_velocity, velocity


def integrate(
    theta0: torch.Tensor, omega: torch.Tensor, k: torch.Tensor, t_max: float, dt: float, save_every: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """RK4 for every coupling K_b at once. Phases are not wrapped (so the mean phase can be followed).
    theta0 (B, N), omega (N,), k (B, 1) → saved phases (S, B, N), order parameter at every step (B, T + 1)."""
    n_steps = round(t_max / dt)
    theta = theta0.to(omega.dtype)
    saved, rs = [theta], [order_parameter(theta)[0]]
    for i in steps(n_steps, "RK4 steps"):
        theta = rk4_step(lambda th, t: velocity(th, omega, k), theta, i * dt, dt)
        rs.append(order_parameter(theta)[0])
        if (i + 1) % save_every == 0:
            saved.append(theta)
    return torch.stack(saved), torch.stack(rs, dim=1)


def sweep(
    omega: torch.Tensor, k: torch.Tensor, t_max: float, dt: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One independent population per coupling K_b, all from random phases.   (N,), (B, 1) → final phases (B, N),
    r(t) (B, T + 1)"""
    theta0 = tensors.initial_phases(len(k), len(omega), omega.device, seed)
    saved, r = integrate(theta0, omega, k.to(omega.dtype), t_max, dt, save_every=round(t_max / dt))
    return saved[-1], r


def ramp(
    omega: torch.Tensor, k_max: float, t_max: float, dt: float, n_frames: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One population while K grows slowly from 0 to k_max: K(t) = k_max t / t_max.
    → K at the saved frames (F,), phases (F, N), r at the frames (F,)."""
    n_steps = round(t_max / dt)
    every = max(1, n_steps // n_frames)
    theta = tensors.initial_phases(1, len(omega), omega.device, seed).to(omega.dtype)
    ks, thetas = [0.0], [theta[0]]
    for i in steps(n_steps, "ramp"):
        theta = rk4_step(lambda th, t: velocity(th, omega, torch.tensor(k_max * t / t_max)), theta, i * dt, dt)
        if (i + 1) % every == 0:
            ks.append(k_max * (i + 1) / n_steps)
            thetas.append(theta[0])
    thetas = torch.stack(thetas)
    return torch.tensor(ks), thetas, order_parameter(thetas)[0]


def chimera(
    n: int = 256, t_max: float = 300.0, dt: float = 0.05, seed: int = 0, a: float = 0.995, beta: float = 0.18
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical oscillators on a ring with kernel (1 + A cos x)/2π and phase lag α = π/2 − β (Abrams and Strogatz
    2004). Returns positions (N,) and the final phases wrapped to (−π, π] (N,), in the frame of the locked domain."""
    x = tensors.ring(n, torch.device("cpu"))
    g = ring_kernel(x, a)
    alpha = math.pi / 2 - beta
    theta = tensors.chimera_start(x, seed)
    for i in steps(round(t_max / dt), "ring"):
        theta = rk4_step(lambda th, t: ring_velocity(th, g, alpha, 0.0), theta, i * dt, dt)
    z = torch.exp(1j * theta)
    local = torch.stack([torch.roll(z, s) for s in range(-5, 6)]).mean(0)  # (N,) neighbourhood mean field
    locked = local[local.abs().argmax()].angle()  # phase of the most coherent neighbourhood
    return x, torch.remainder(theta - locked + math.pi, 2 * math.pi) - math.pi
