"""Physics checks, written first (TDD): what any correct inclined-plane code must satisfy.

Run with `pytest problems/mechanics/inclined_plane/test.py`. Newton's second law with Coulomb friction: a = g(sin θ −
μ_k cos θ) when sliding, no motion below the friction angle, and d = ½ g sin θ t² without friction.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_sliding_acceleration():
    g, mu_s, mu_k = 9.81, 0.3, 0.2
    theta = torch.deg2rad(torch.tensor([20.0, 35.0, 60.0]))  # tan θ = 0.36, 0.70, 1.73 > μ_s
    s, n = tensors.unit_vectors(theta, CPU)
    a = equation.acceleration(tensors.gravity(g, CPU), s, n, mu_s, mu_k)
    expected = g * (torch.sin(theta) - mu_k * torch.cos(theta))
    assert torch.allclose(torch.einsum("bi,bi->b", a, s), expected, atol=1e-5)
    assert torch.allclose(torch.einsum("bi,bi->b", a, n), torch.zeros(3), atol=1e-5)  # stays on the surface


def test_stays_at_rest_below_friction_angle():
    g, mu_s, mu_k = 9.81, 0.6, 0.4
    theta = torch.deg2rad(torch.tensor([5.0, 20.0, 30.0]))  # all tan θ < 0.6
    s, n = tensors.unit_vectors(theta, CPU)
    a = equation.acceleration(tensors.gravity(g, CPU), s, n, mu_s, mu_k)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-5)
    sol = solve.solve(a, s, ramp_length=3.0, t_max=2.0)
    assert sol["d"].abs().max() < 1e-4


def test_frictionless_distance():
    theta = torch.deg2rad(torch.tensor([30.0]))
    s, n = tensors.unit_vectors(theta, CPU)
    a = equation.acceleration(tensors.gravity(9.81, CPU), s, n, 0.0, 0.0)
    sol = solve.solve(a, s, ramp_length=100.0, t_max=2.0)
    assert sol["d"][-1, 0].item() == pytest.approx(0.5 * 9.81 * 0.5 * 4.0, rel=1e-5)  # ½ g sin30° t²
