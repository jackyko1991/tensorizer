"""Physics checks, written first (TDD): what any correct Mandelbrot/Julia code must satisfy.

Run with `pytest problems/maths/mandelbrot_julia/test.py`. Each test is a mathematical invariant, not an implementation
detail: the main cardioid never escapes, |c| > 2 escapes at once, the distance estimate is bounded by the true distance,
and the Julia set of c = 0 is the unit circle.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_points_inside_the_main_cardioid_never_escape():
    c = tensors.pixel_grid(complex(-0.6, 0.0), 3.2, 120, CPU)
    inside = equation.in_cardioid_or_bulb(c)
    assert inside.sum() > 1000
    result = solve.mandelbrot(c[inside], 2000)
    assert (result["n"] == 0).all()
    # centres of the cardioid's family: 0 (fixed point), −1 (2-cycle), and points just inside the edge
    w = 0.97 * torch.exp(1j * torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64))
    edge = w / 2 - w * w / 4
    assert equation.in_cardioid_or_bulb(edge).all()
    assert (solve.mandelbrot(edge, 5000)["n"] == 0).all()


def test_points_with_modulus_above_two_escape_at_the_first_step():
    angle = torch.linspace(0, 2 * math.pi, 200, dtype=torch.float64)
    radius = torch.linspace(2.01, 5.0, 200, dtype=torch.float64)
    c = radius * torch.exp(1j * angle)
    result = solve.mandelbrot(c, 100, radius=2.0)
    assert (result["n"] == 1).all()
    assert not equation.in_cardioid_or_bulb(c).any()


def test_distance_estimate_is_positive_outside_and_bounded_by_true_distance():
    c = tensors.pixel_grid(complex(-0.6, 0.0), 3.2, 100, CPU)
    result = solve.mandelbrot(c, 500)
    out = result["n"] > 0
    assert (result["d"][out] > 0).all()
    # on the real axis right of M (which ends at c = 1/4) the true distance is c − 1/4
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64).to(torch.complex128)
    d = solve.mandelbrot(x, 500)["d"]
    true = x.real - 0.25
    assert ((d > true / 4) & (d < 4 * true)).all()


def test_cardioid_boundary_has_neutral_fixed_point_and_julia_is_connected_there():
    theta = torch.linspace(0.1, 6.0, 40, dtype=torch.float64)
    c = equation.cardioid_boundary(theta)
    multiplier = (1 - torch.sqrt(1 - 4 * c)).abs()  # |2 z*| for the attracting-branch fixed point
    assert torch.allclose(multiplier, torch.ones_like(multiplier), atol=1e-12)
    # c = 0: the Julia set is the unit circle, so |z0| < 1 stays bounded and |z0| > 1 escapes
    z0 = torch.tensor([0.5, 0.9j, -0.99, 1.01, 1.5j], dtype=torch.complex128)
    n = solve.julia(z0, torch.zeros(1, dtype=torch.complex128), 400)["n"]
    assert (n[:3] == 0).all() and (n[3:] > 0).all()
