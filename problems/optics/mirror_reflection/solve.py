"""Find where each ray meets its mirror, reflect it, and record the ray tip at every time."""

import torch

from .equation import dot, reflect, sphere_normal


def hit_flat(p: torch.Tensor, d: torch.Tensor, point: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Distance t along each ray to the plane through `point` with unit `normal`: (R,).

    Solve (p + t d − a)·n = 0  for t.
    """
    n = normal.expand_as(d)
    return dot(point - p, n) / dot(d, n)


def hit_sphere(p: torch.Tensor, d: torch.Tensor, centre: torch.Tensor, radius: float) -> torch.Tensor:
    """Distance t along each ray to the sphere |p + t d − c| = R: (R,).

    With q = p − c this is the quadratic t² + 2 (q·d) t + (q·q − R²) = 0. The rays start inside
    the sphere, so the + root is the one in front of them (the mirror on the right).
    """
    q = p - centre  # (R, 2)
    b = dot(q, d)  # (R,)
    c = dot(q, q) - radius**2  # (R,)
    return -b + torch.sqrt(b**2 - c)


def bounce(p: torch.Tensor, d: torch.Tensor, t_hit: torch.Tensor, normal: torch.Tensor):
    """Return (hit points (R, 2), reflected directions (R, 2))."""
    p_hit = p + t_hit[:, None] * d
    return p_hit, reflect(d, normal)


def tips(times: torch.Tensor, p, d, t_hit, p_hit, r) -> torch.Tensor:
    """Position of every ray tip at every time: (T,) → (T, R, 2).

    Before the bounce the tip is at p + t d; afterwards at p_hit + (t − t_hit) r.
    """
    t = times[:, None, None]  # (T, 1, 1) so it broadcasts against (R, 2)
    before = p + t * d
    after = p_hit + (t - t_hit[:, None]) * r
    return torch.where(t < t_hit[:, None], before, after)


def axis_crossing(p_hit: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """x where each reflected ray crosses the axis y = 0: (R,).

    Walk back along r from the hit point until y = 0:  x = p_x − p_y r_x / r_y.
    (A ray travelling exactly on the axis gives 0/0 = NaN.)
    """
    return p_hit[:, 0] - p_hit[:, 1] * r[:, 0] / r[:, 1]


def solve(radius: float, lamp: tuple, parallel: tuple, flat: tuple, centre: torch.Tensor, times: torch.Tensor):
    """Trace both scenes. Returns a dict of tensors for the plot."""
    p1, d1 = lamp
    t1 = hit_flat(p1, d1, *flat)
    hit1, r1 = bounce(p1, d1, t1, flat[1].expand_as(d1))

    p2, d2 = parallel
    t2 = hit_sphere(p2, d2, centre, radius)
    hit2 = p2 + t2[:, None] * d2
    hit2, r2 = bounce(p2, d2, t2, sphere_normal(hit2, centre, radius))

    return {
        "flat": tips(times, p1, d1, t1, hit1, r1),  # (T, R, 2) ray tips
        "flat_start": p1,  # (R, 2)
        "flat_hits": hit1,  # (R, 2)
        "flat_t_hit": t1,  # (R,)
        "flat_dirs": r1,  # (R, 2)
        "curved": tips(times, p2, d2, t2, hit2, r2),  # (T, R, 2)
        "curved_start": p2,
        "curved_hits": hit2,
        "curved_t_hit": t2,
        "crossings": axis_crossing(hit2, r2),  # (R,)
    }
