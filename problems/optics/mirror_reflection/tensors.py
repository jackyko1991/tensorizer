"""Build the concrete tensors: the lamp's fan of rays for the flat mirror, parallel rays for the curved one.

Both mirrors sit at x = 0 and the light arrives from the left (x < 0).
"""

import torch


def lamp_rays(radius: float, n_rays: int, aperture: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A lamp at (−R/2, R/4) shines at points spread along the flat mirror x = 0.

    p[r, :] = lamp position                 start point p_{ri}       (R, 2)
    d[r, :] = unit vector lamp → mirror     direction d_{ri}         (R, 2)
    """
    lamp = torch.tensor([-0.5 * radius, 0.25 * radius])  # (2,)
    y_target = torch.linspace(-aperture, aperture, n_rays) * radius  # (R,) where each ray aims
    target = torch.stack([torch.zeros_like(y_target), y_target], dim=-1)  # (R, 2)
    d = target - lamp
    d = d / d.norm(dim=-1, keepdim=True)  # make every direction a unit vector
    return lamp.expand_as(d), d


def parallel_rays(radius: float, n_rays: int, aperture: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays parallel to the axis, at heights between −aperture·R and +aperture·R, starting at x = −1.2 R.

    p[r, :] = (−1.2 R, y_r)        (R, 2)
    d[r, :] = (1, 0)               (R, 2)
    """
    y = torch.linspace(-aperture, aperture, n_rays) * radius  # (R,)
    p = torch.stack([torch.full_like(y, -1.2 * radius), y], dim=-1)
    d = torch.tensor([1.0, 0.0]).expand_as(p)
    return p, d


def flat_mirror() -> tuple[torch.Tensor, torch.Tensor]:
    """A point on the flat mirror and its unit normal (facing the light): two (2,) tensors."""
    return torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 0.0])


def mirror_centre(radius: float) -> torch.Tensor:
    """Centre of curvature of the concave mirror whose vertex is at the origin: (2,) = (−R, 0)."""
    return torch.tensor([-radius, 0.0])


def times(radius: float, n_steps: int = 120) -> torch.Tensor:
    """Time samples (light speed 1), long enough for rays to come back past the focus: (T,)."""
    return torch.linspace(0.0, 2.2 * radius, n_steps)
