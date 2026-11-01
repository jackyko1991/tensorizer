"""Physics checks, written first (TDD): what any correct Jones-calculus code must satisfy.

Run with `pytest problems/optics/polarization_jones/test.py`. Malus's law I = I₀ cos² θ, a quarter-wave plate at 45°
making circular light, and the I₀/8 three-polarizer result.
"""

import math

import torch

from . import equation, solve, tensors


def test_malus_law():
    e_in = tensors.linear_input(0.0)
    for theta in torch.linspace(0.0, math.pi, 13).tolist():
        out = equation.apply(equation.polarizer(theta), e_in)
        assert math.isclose(equation.intensity(out).item(), math.cos(theta) ** 2, abs_tol=1e-6)
    angles = tensors.analyzer_angles(19)
    result = solve.solve(e_in, tensors.plate("none", 0.0), angles, tensors.phases())
    assert torch.allclose(result["intensity"], torch.cos(angles) ** 2, atol=1e-6)


def test_quarter_wave_plate_makes_circular_light():
    e_in, qwp = tensors.linear_input(0.0), tensors.plate("quarter", 45.0)
    ex, ey, phase = solve.describe(equation.apply(qwp, e_in))
    assert math.isclose(ex, ey, rel_tol=1e-5) and math.isclose(ex, 1 / math.sqrt(2), rel_tol=1e-5)
    assert math.isclose(abs(phase), 90.0, abs_tol=1e-3)
    result = solve.solve(e_in, qwp, tensors.analyzer_angles(19), tensors.phases())
    assert torch.allclose(result["intensity"], torch.full((19,), 0.5), atol=1e-6)  # every analyzer passes I₀/2
    radius = result["ellipse_plate"].norm(dim=-1)
    assert torch.allclose(radius, torch.full_like(radius, 1 / math.sqrt(2)), atol=1e-5)  # the ellipse is a circle


def test_crossed_polarizers_and_the_45_degree_trick():
    # unpolarized light with I₀ = 1: the first polarizer passes I₀/2, crossed ones block,
    # and a 45° polarizer between them lets I₀/2 · cos²45° · cos²45° = I₀/8 through
    p0, p45, p90 = (equation.polarizer(math.radians(a)) for a in (0, 45, 90))
    assert math.isclose(equation.unpolarized_intensity(p0).item(), 0.5, abs_tol=1e-6)
    assert equation.unpolarized_intensity(equation.chain(p0, p90)).item() < 1e-10
    assert math.isclose(equation.unpolarized_intensity(equation.chain(p0, p45, p90)).item(), 1 / 8, abs_tol=1e-6)
    # a half-wave plate at α rotates linear light by 2α
    e = equation.apply(tensors.plate("half", 30.0), tensors.linear_input(0.0))
    assert torch.allclose(e.abs(), torch.tensor([0.5, math.sqrt(3) / 2]), atol=1e-6)
