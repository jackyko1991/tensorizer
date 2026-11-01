"""Build the concrete tensors: the batch of f values, the starting points and the broken wave that seeds a spiral.

The scaled Oregonator has two small constants, ε (set from the page, default 0.04) and q below. Both are ratios of
Field–Kőrös–Noyes rate constants and reactant concentrations; the values used here (Tyson 1985) are approximate, and
published fits range over ε ≈ 0.01–0.1 and q ≈ 2×10⁻⁴–8×10⁻⁴ depending on the recipe.
"""

import torch

from .equation import steady_state

Q = 8e-4  # q: a small ratio of rate constants (approximate)
DT = 2e-3  # time step of the Patankar–Euler scheme, in units of the z relaxation time
H = 0.5  # grid spacing of the 2D medium, in units where D_x = 1


def f_values(n: int, device: torch.device) -> torch.Tensor:
    """A batch of stoichiometric factors f_b spanning both Hopf bifurcations.   (B,)"""
    return torch.linspace(0.3, 3.0, n, dtype=torch.float64, device=device)


def initial_state(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Start every member of the batch 50 % above its steady state in x.   (B,), (B,)"""
    xs = steady_state(f, Q)
    return 1.5 * xs, xs.clone()


def broken_wave(n: int, f: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """x, z on an n × n grid (n, n): rest everywhere, an excited band (high x) along half of the middle row,
    and a refractory band (high z) just above it, so the wave can only travel down and its free end curls up."""
    xs = steady_state(torch.tensor(f, dtype=torch.float64), Q).item()
    x = torch.full((n, n), xs, dtype=torch.float64, device=device)
    z = torch.full((n, n), xs, dtype=torch.float64, device=device)
    w = max(2, n // 20)
    x[n // 2 : n // 2 + w, : n // 2] = 0.8
    z[n // 2 - 2 * w : n // 2, : n // 2] = 0.3
    return x, z
