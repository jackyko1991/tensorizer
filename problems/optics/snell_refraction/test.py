"""Physics checks, written first (TDD): what any correct refraction code must satisfy.

Run with `pytest problems/optics/snell_refraction/test.py`. Snell's law n₁ sin θ₁ = n₂ sin θ₂ holds at every interface,
and above the critical angle the ray is totally internally reflected.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _incoming(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.stack([torch.sin(angles), -torch.cos(angles)], dim=-1)  # going down
    n = torch.tensor([0.0, 1.0]).expand_as(d)  # normal facing the incoming ray
    return d, n


def test_snell_law_holds_below_critical_angle():
    n1, n2 = 1.0, 1.5
    angles = torch.deg2rad(torch.linspace(0.0, 85.0, 18))
    d, n = _incoming(angles)
    t, tir = equation.refract(d, n, torch.full_like(angles, n1 / n2))
    assert not tir.any()
    assert torch.allclose(t.norm(dim=-1), torch.ones(len(t)), atol=1e-6)  # still unit vectors
    assert torch.allclose(n1 * equation.sin_angle(d, n), n2 * equation.sin_angle(t, n), atol=1e-6)
    assert (t[:, 1] < 0).all()  # transmitted, keeps going down


def test_total_internal_reflection_above_critical_angle():
    n1, n2 = 1.5, 1.0
    theta_c = solve.critical_angle(n1, n2)
    assert math.isclose(theta_c, math.degrees(math.asin(n2 / n1)))
    angles = torch.deg2rad(torch.tensor([theta_c - 5, theta_c - 1, theta_c + 1, theta_c + 10]))
    d, n = _incoming(angles)
    t, tir = equation.refract(d, n, torch.full_like(angles, n1 / n2))
    assert tir.tolist() == [False, False, True, True]
    # reflected rays: angle preserved, vertical component flipped
    assert torch.allclose(t[tir, 0], d[tir, 0]) and torch.allclose(t[tir, 1], -d[tir, 1])
    assert torch.allclose(n1 * equation.sin_angle(d[~tir], n[~tir]), n2 * equation.sin_angle(t[~tir], n[~tir]))


def test_traced_rays_obey_snell_between_layers():
    n_layer = tensors.indices(1.6, 1.33, 1.0, CPU)
    angles = tensors.fan_angles(4, 30.0)  # all below every critical angle
    p0, d0 = tensors.initial_rays(angles, CPU)
    _, rays = solve.solve(p0, d0, n_layer, t_max=6.0, dt=0.005)
    # x where each ray crosses y = -1 and y = -2 gives the slope in the middle layer
    mid = rays[:, :, 1]
    x1 = torch.stack([rays[(mid[:, r] <= -1.0).nonzero()[0, 0], r, 0] for r in range(len(angles))])
    x2 = torch.stack([rays[(mid[:, r] <= -2.0).nonzero()[0, 0], r, 0] for r in range(len(angles))])
    theta2 = torch.atan2(x2 - x1, torch.ones_like(x1))
    assert torch.allclose(1.6 * torch.sin(angles), 1.33 * torch.sin(theta2), atol=2e-2)
