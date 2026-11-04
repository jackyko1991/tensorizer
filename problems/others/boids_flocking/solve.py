"""Time stepping: the Vicsek model for a batch of noise levels, and Reynolds boids."""

import torch

from problems._common.progress import steps

from .equation import polar_order, reynolds_acceleration, vicsek_step


def vicsek(
    x: torch.Tensor,
    theta: torch.Tensor,
    eta: torch.Tensor,
    n_steps: int,
    box: float,
    radius: float,
    v0: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run every noise level η_b (B, 1) at once from x (B, N, 2), θ (B, N).
    → positions (T + 1, B, N, 2), headings (T + 1, B, N), polar order φ (B, T + 1)."""
    gen = torch.Generator().manual_seed(seed)
    xs, thetas = [x], [theta]
    for _ in steps(n_steps, "Vicsek steps"):
        xi = (torch.rand(theta.shape, generator=gen) - 0.5).to(theta.device)  # uniform in [−½, ½]
        x, theta = vicsek_step(x, theta, eta, xi, box, radius, v0)
        xs.append(x)
        thetas.append(theta)
    thetas = torch.stack(thetas)
    unit = torch.stack([torch.cos(thetas), torch.sin(thetas)], -1)  # (T + 1, B, N, 2)
    phi = polar_order(unit.flatten(0, 1)).reshape(len(thetas), -1).T  # (B, T + 1)
    return torch.stack(xs), thetas, phi


def boids(
    x: torch.Tensor,
    v: torch.Tensor,
    n_steps: int,
    dt: float,
    box: float,
    radius: float,
    weights: dict,
    speed: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler: v ← v + a dt (then |v| ← v0 if speed is given), x ← x + v dt in the periodic box.
    (B, N, 2), (B, N, 2) → positions (T + 1, B, N, 2), velocities (T + 1, B, N, 2)."""
    xs, vs = [x], [v]
    for _ in steps(n_steps, "boids steps"):
        v = v + dt * reynolds_acceleration(x, v, box, radius, weights)
        if speed is not None:
            v = speed * v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        x = torch.remainder(x + dt * v, box)
        xs.append(x)
        vs.append(v)
    return torch.stack(xs), torch.stack(vs)
