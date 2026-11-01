"""Physics checks, written first (TDD): what any correct CT reconstruction code must satisfy.

Run with `pytest problems/medical/radon_ct/test.py`. Each test is a physical law: projections of a disk are chord
lengths, every projection conserves the total mass, and filtered backprojection converges as the number of angles
grows.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_projection_of_centred_disk_matches_chord_length():
    x = tensors.grid(256, CPU)
    R = 0.5
    theta = tensors.angles(12, CPU)
    p = solve.scan(tensors.disk(x, R), theta, x)
    chord = 2 * torch.sqrt((R**2 - x**2).clamp_min(0.0))
    assert (p - chord).abs().max() < 0.02  # ~ one pixel width (2/256) of edge error
    assert (p - chord).abs().mean() < 0.003


def test_projection_conserves_mass_for_every_angle():
    x = tensors.grid(128, CPU)
    f = tensors.shepp_logan(x, tensors.ellipse_params(CPU))
    dx = float(x[1] - x[0])
    p = solve.scan(f, tensors.angles(45, CPU), x)
    mass = p.sum(1) * dx
    assert torch.allclose(mass, (f.sum() * dx * dx).expand_as(mass), rtol=5e-3)


def test_analytic_ellipse_projection_matches_sampled_one():
    x = tensors.grid(128, CPU)
    E = tensors.ellipse_params(CPU)
    theta = tensors.angles(30, CPU)
    analytic = sum(equation.ellipse_projection(theta, x, *e) for e in E)
    sampled = solve.scan(tensors.shepp_logan(x, E), theta, x)
    assert (analytic - sampled).abs().mean() < 0.01


def test_fbp_error_small_and_decreasing_with_angles():
    x = tensors.grid(128, CPU)
    f = tensors.shepp_logan(x, tensors.ellipse_params(CPU))
    errors = solve.error_vs_angles(f, x, [8, 16, 32, 64, 128], "ram-lak")
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.3  # limited by the sharp skull edges on a 128² grid
    assert errors[0] > 3 * errors[-1]


def test_plain_backprojection_of_point_falls_like_one_over_r():
    n = 129  # odd: a pixel sits exactly at the centre
    x = tensors.grid(n, CPU)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    point = torch.exp(-(xx**2 + yy**2) / (2 * (1.5 * 2 / n) ** 2))
    theta = tensors.angles(720, CPU)
    b = equation.backproject(equation.smear(solve.scan(point, theta, x), theta, x, x), theta)
    c = n // 2
    r_times_b = torch.tensor([float(b[c, c + k] * x[c + k]) for k in (10, 20, 40, 60)])
    assert (r_times_b.max() - r_times_b.min()) / r_times_b.mean() < 0.05
