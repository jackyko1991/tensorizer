"""Physics checks, written first (TDD): what any correct mirror code must satisfy.

Run with `pytest problems/optics/mirror_reflection/test.py`. The law of reflection (angle in = angle out), paraxial rays
focusing at R/2 on a spherical mirror, and a flat mirror's virtual image behind it.
"""

import torch

from . import equation, solve, tensors


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def test_reflection_angle_equals_incidence_angle():
    angles = torch.deg2rad(torch.linspace(-80.0, 80.0, 17))
    d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # heading towards +x
    tilt = torch.deg2rad(torch.tensor(20.0))
    n = torch.stack([-torch.cos(tilt), torch.sin(tilt)]).expand_as(d)  # a tilted mirror facing −x
    r = equation.reflect(d, n)
    assert torch.allclose(r.norm(dim=-1), torch.ones(len(r)), atol=1e-6)
    assert torch.allclose(-equation.dot(d, n), equation.dot(r, n), atol=1e-6)  # same angle with the normal
    assert torch.allclose(_cross(d, n), _cross(r, n), atol=1e-6)  # the part along the mirror is unchanged


def _curved_crossings(radius: float, n_rays: int, aperture: float) -> tuple[torch.Tensor, torch.Tensor]:
    p, d = tensors.parallel_rays(radius, n_rays, aperture)
    centre = tensors.mirror_centre(radius)
    t = solve.hit_sphere(p, d, centre, radius)
    hit = p + t[:, None] * d
    assert torch.allclose((hit - centre).norm(dim=-1), torch.full((n_rays,), radius), atol=1e-4)
    r = equation.reflect(d, equation.sphere_normal(hit, centre, radius))
    return p[:, 1], solve.axis_crossing(hit, r)


def test_paraxial_rays_cross_near_half_radius():
    radius = 4.0
    y, x = _curved_crossings(radius, 11, 0.05)  # heights up to 0.05 R: paraxial
    off_axis = y != 0
    assert torch.allclose(x[off_axis], torch.full_like(x[off_axis], -radius / 2), atol=0.005 * radius)
    y, x = _curved_crossings(radius, 3, 0.6)  # marginal rays cross closer to the mirror (spherical aberration)
    assert x[0] > -radius / 2 + 0.05 * radius


def test_flat_mirror_rays_meet_at_virtual_image():
    radius = 4.0
    p, d = tensors.lamp_rays(radius, 7, 0.6)
    point, normal = tensors.flat_mirror()
    t = solve.hit_flat(p, d, point, normal)
    hit, r = solve.bounce(p, d, t, normal.expand_as(d))
    assert torch.allclose(hit[:, 0], torch.zeros(7), atol=1e-6)
    image = torch.tensor([0.5 * radius, 0.25 * radius])  # lamp mirrored to the other side of x = 0
    assert torch.allclose(_cross(r, image - hit), torch.zeros(7), atol=1e-5)  # image lies on every reflected line
