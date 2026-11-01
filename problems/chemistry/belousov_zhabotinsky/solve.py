"""Positive, stable Patankar–Euler time stepping of the stiff Oregonator: a batch of f values, and a 2D medium."""

import torch

from problems._common.progress import steps

from .equation import jacobian, laplacian, production_loss, steady_state, trace


def step(x, z, f, eps: float, q: float, dt: float, diffusion: torch.Tensor | float = 0.0):
    """One step: x ← (x + Δt (P/ε + D∇²x)) / (1 + Δt L/ε), then z ← (z + Δt x) / (1 + Δt).

    Every loss term is taken at the new time (implicit) and every gain at the old one, so the stiff x relaxes
    without overshooting below zero. The z step is backward Euler, exact for its linear decay."""
    P, L = production_loss(x, z, f, q)
    x = ((x + dt * (P / eps + diffusion)) / (1 + dt * L / eps)).clamp(min=1e-12)  # clamp guards the D∇²x < 0 corner
    z = (z + dt * x) / (1 + dt)
    return x, z


def integrate(x0, z0, f, eps: float, q: float, dt: float, n_steps: int, n_frames: int = 300):
    """Batch of well-mixed reactions. Returns times (T,), x (T, B), z (T, B) at n_frames evenly spaced times."""
    every = max(1, n_steps // n_frames)
    x, z = x0.clone(), z0.clone()
    t, xs, zs = [0.0], [x], [z]
    for k in steps(n_steps, "Oregonator steps"):
        x, z = step(x, z, f, eps, q, dt)
        if (k + 1) % every == 0:
            t.append((k + 1) * dt)
            xs.append(x)
            zs.append(z)
    return torch.tensor(t), torch.stack(xs), torch.stack(zs)


def spiral(x0, z0, f: float, eps: float, q: float, dt: float, h: float, n_steps: int, n_frames: int = 90):
    """2D medium where only x diffuses (D_x = 1; the catalyst z is held in a gel). Returns z snapshots (T, n, n)."""
    every = max(1, n_steps // n_frames)
    x, z = x0.clone(), z0.clone()
    frames = [z.clone()]
    for k in steps(n_steps, "spiral-wave steps"):
        x, z = step(x, z, f, eps, q, dt, diffusion=laplacian(x, h))
        if (k + 1) % every == 0:
            frames.append(z.clone())
    return torch.stack(frames)


def hopf_points(f: torch.Tensor, eps: float, q: float) -> torch.Tensor:
    """Values of f where tr J at the steady state changes sign (det J > 0 there): the Hopf bifurcations.

    f (F,) a fine grid → the crossings, linearly interpolated."""
    xs = steady_state(f, q)
    tr = trace(jacobian(xs, xs, f, eps, q))  # (F,)
    i = torch.nonzero(torch.sign(tr[1:]) != torch.sign(tr[:-1])).flatten()
    return f[i] - tr[i] * (f[i + 1] - f[i]) / (tr[i + 1] - tr[i])


def period(t: torch.Tensor, x: torch.Tensor) -> float | None:
    """Mean time between upward crossings of the geometric mean of x's range, in the second half of the run."""
    half = len(t) // 2
    t, x = t[half:], x[half:].log()
    level = (x.min() + x.max()) / 2
    up = torch.nonzero((x[1:] > level) & (x[:-1] <= level)).flatten()
    if x.max() - x.min() < 0.5 or len(up) < 2:
        return None
    return (t[up[-1]] - t[up[0]]).item() / (len(up) - 1)


def amplitude(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum and maximum of x over the second half of the run (after the transient).   x (T, B) → (B,), (B,)"""
    late = x[x.shape[0] // 2 :]
    return late.min(0).values, late.max(0).values
