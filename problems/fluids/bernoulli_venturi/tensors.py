"""Build the concrete tensors: the axis grid, the pipe radius r(x) and the tracer particles."""

import torch

LENGTH = 1.0  # pipe length (m); the profile repeats beyond it so tracers can loop around


def smoothstep(s: torch.Tensor) -> torch.Tensor:
    """0 for s ≤ 0, 1 for s ≥ 1, a smooth S-curve in between."""
    s = s.clamp(0.0, 1.0)
    return 3 * s**2 - 2 * s**3


def radius(x: torch.Tensor, r1: float, r2: float) -> torch.Tensor:
    """Pipe radius at any x: r1 at the inlet, narrowing to r2 in the throat, slowly opening again."""
    x = x % LENGTH
    narrowing = smoothstep((x - 0.15) / 0.22) * (1 - smoothstep((x - 0.55) / 0.3))  # 0 … 1 … 0
    return r1 - (r1 - r2) * narrowing


def axis(n: int, device: torch.device) -> torch.Tensor:
    """N points along the pipe axis: shape (N,)."""
    return torch.linspace(0.0, LENGTH, n, device=device)


def tracers(n_rows: int, n_cols: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracer start positions x (n,) and their fixed fractional heights η ∈ (−1, 1) (n,).

    A tracer at height η always sits at y = η r(x): in this one-dimensional picture it follows the wall shape.
    """
    eta = torch.linspace(-0.8, 0.8, n_rows, device=device)
    x = torch.arange(n_cols, device=device) * LENGTH / n_cols
    xx, ee = torch.meshgrid(x, eta, indexing="ij")  # (n_cols, n_rows)
    xx = xx + 0.5 * LENGTH / n_cols * (torch.arange(n_rows, device=device) % 2)  # stagger the rows
    return xx.flatten(), ee.flatten()
