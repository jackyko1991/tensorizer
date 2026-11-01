"""Physics checks, written first (TDD): what any correct drum-membrane code must satisfy.

Run with `pytest problems/waves/membrane_modes/test.py`. Square-drum frequencies scale as √(m² + n²) with orthonormal
modes, and the circular drum's fundamental matches the first Bessel zero.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_square_membrane_frequencies_and_orthonormal_modes():
    n = 30
    X, Y, h = tensors.grid(n, CPU)
    D = tensors.second_difference(n, h, CPU)
    inside = tensors.drum_mask(X, Y, "square")
    omega, shapes = solve.solve(D, inside, tensors.C)
    ratios = (omega[:6] / omega[0]).tolist()
    exact = [math.sqrt((m * m + q * q) / 2) for m, q in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]]
    assert ratios == pytest.approx(exact, rel=5e-3)  # ω_mn ∝ √(m² + n²)
    assert omega[0].item() == pytest.approx(math.pi * math.sqrt(2) * tensors.C / tensors.SIDE, rel=2e-3)
    lap = equation.restrict(equation.laplacian_2d(D), inside)
    _, phi = equation.modes(lap, tensors.C)
    assert torch.allclose(equation.overlap(phi), torch.eye(phi.shape[1], dtype=phi.dtype), atol=1e-10)


def test_circular_membrane_fundamental_close_to_bessel_zero():
    n = 40
    X, Y, h = tensors.grid(n, CPU)
    omega, _ = solve.solve(tensors.second_difference(n, h, CPU), tensors.drum_mask(X, Y, "circle"), tensors.C)
    assert omega[0].item() == pytest.approx(2.40483 * tensors.C / (tensors.SIDE / 2), rel=3e-2)  # staircase rim
    assert (omega[1] / omega[0]).item() == pytest.approx(3.83171 / 2.40483, rel=1e-2)
