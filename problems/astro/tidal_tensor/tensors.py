"""Build the concrete tensors: a ring of test particles dropped from rest toward the black hole.

Units: G = c = 1 and lengths in GM/c², so the black hole has GM = 1 and its horizon (Schwarzschild radius)
is at r_s = 2. For a 10-solar-mass black hole one length unit is 15 km and one time unit 49 µs.
"""

import math

import torch

GM = 1.0


def initial_state(r0: float, ring_radius: float, n: int, device: torch.device) -> torch.Tensor:
    """y0 of shape (1 + 2n, 2, 3), everything at rest.

    row 0          centre of the ring, at (r0, 0, 0)
    rows 1 … n     ring particles at the centre + ξ_b, followed exactly
    rows n+1 … 2n  the separations ξ_b themselves, evolved with the linear tidal law
    """
    phi = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    xi = ring_radius * torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=-1)  # (n, 3)
    centre = torch.tensor([[r0, 0.0, 0.0]], dtype=torch.float64)
    x = torch.cat([centre, centre + xi, xi])
    return torch.stack([x, torch.zeros_like(x)], dim=1).to(device)
