"""Build the concrete tensors: field points, the charges and the path of the moving charge."""

import math

import torch

L = 2.0  # the plot shows the square −L ≤ x, y ≤ L


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coordinates (N,), field points (N·N, 2)).

    The points are listed row by row (y outer, x inner), so reshape(N, N, 2) gives rows = y, columns = x.
    """
    xs = torch.linspace(-L, L, n, device=device)  # (N,)
    Y, X = torch.meshgrid(xs, xs, indexing="ij")  # (N, N) each
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # (N·N, 2)
    return xs, points


def x_axis(n: int, device: torch.device) -> torch.Tensor:
    """n points along the x axis (y = 0), shape (n, 2): the field is drawn along this line too."""
    xs = torch.linspace(-L, L, n, device=device)
    return torch.stack([xs, torch.zeros_like(xs)], dim=-1)


def charges(q_fixed: float, q_moving: float, device: torch.device) -> torch.Tensor:
    """Q_q of shape (2,): charge 0 sits still, charge 1 moves."""
    return torch.tensor([q_fixed, q_moving], device=device)


def charge_path(n_steps: int, device: torch.device) -> torch.Tensor:
    """x_{tqi} of shape (T, 2, 2): positions of both charges at each of T time steps.

    Charge 0 stays at (−0.6, 0.03); charge 1 goes once round a circle of radius 1.1 about the origin.
    (The small offset keeps charge 0 just off the x axis, where the field is sampled.)
    """
    angle = torch.linspace(0.0, 2 * math.pi, n_steps, device=device)  # (T,)
    path = torch.zeros(n_steps, 2, 2, device=device)
    path[:, 0, 0] = -0.6  # fixed charge
    path[:, 0, 1] = 0.03
    path[:, 1, 0] = 1.1 * torch.cos(angle)  # moving charge, x
    path[:, 1, 1] = 1.1 * torch.sin(angle)  # moving charge, y
    return path
