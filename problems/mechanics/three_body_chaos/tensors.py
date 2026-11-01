"""Build the concrete tensors for Burrau's Pythagorean three-body problem.

Masses 3, 4, 5 start at rest on the corners of a 3-4-5 right triangle, each opposite the side of the
same length. Units: G = 1. float64, because the twin run differs from the reference by only δ ~ 1e-6.
"""

import torch

G = 1.0
DTYPE = torch.float64


def masses(device: torch.device | None = None) -> torch.Tensor:
    """m_a, shape (3,)."""
    return torch.tensor([3.0, 4.0, 5.0], dtype=DTYPE, device=device)


def pythagorean_positions(device: torch.device | None = None) -> torch.Tensor:
    """x_ai, shape (3, 2): the right angle sits at mass 5; the centre of mass is at the origin.

    side opposite mass 3: |x_4 - x_5| = 3,  opposite mass 4: |x_3 - x_5| = 4,  opposite mass 5: |x_3 - x_4| = 5.
    """
    return torch.tensor([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], dtype=DTYPE, device=device)


def initial_state(delta: float, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(m (3,), x (K=2, 3, 2), v (K=2, 3, 2)): run 0 = reference, run 1 = twin with mass 3 moved by δ along x."""
    m = masses(device)
    x = pythagorean_positions(device).expand(2, 3, 2).clone()  # (K, N, 2) two copies
    x[1, 0, 0] += delta  # the twin: one coordinate nudged by δ
    v = torch.zeros_like(x)  # everything starts at rest
    return m, x, v
