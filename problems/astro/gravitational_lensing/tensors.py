"""Build the concrete tensors: a fan of light rays, the sky pixel grid and the path of the background source.

Rays: units G = c = M = 1 (lengths in GM/c²). Sky: angles in units of the Einstein radius θ_E.
"""

import torch


def ray_fan(b_min: float, b_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Impact parameters b (B,) and initial states y0 (B, 2): u = 0 (at infinity), du/dφ = 1/b."""
    b = torch.linspace(b_min, b_max, n, dtype=torch.float64, device=device)
    return b, torch.stack([torch.zeros_like(b), 1 / b], dim=1)


def sky_grid(n: int, half_width: float, device: torch.device) -> torch.Tensor:
    """Angles θ_{pi} of an n × n pixel grid, shape (n, n, 2), offset by half a pixel so no pixel sits on the lens."""
    s = torch.linspace(-half_width, half_width, n + 1, device=device)
    s = 0.5 * (s[1:] + s[:-1])
    X, Y = torch.meshgrid(s, s, indexing="xy")
    return torch.stack([X, Y], dim=-1)


def source_path(n: int, impact: float, half_length: float, device: torch.device) -> torch.Tensor:
    """Source centre (F, 2) moving along x at height `impact` behind the lens."""
    x = torch.linspace(-half_length, half_length, n, device=device)
    return torch.stack([x, torch.full_like(x, impact)], dim=-1)
