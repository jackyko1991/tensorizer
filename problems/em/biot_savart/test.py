"""Physics checks, written first (TDD): what any correct Biot–Savart code must satisfy.

Run with `pytest problems/em/biot_savart/test.py`. The integrated field of a current loop matches the on-axis closed
form μ₀IR²/2(R²+z²)^{3/2}, and a long straight wire gives Ampère's μ₀I/2πr circling the wire.
"""

import torch

from . import equation, tensors

CPU = torch.device("cpu")


def test_loop_field_on_axis_matches_formula():
    eps = tensors.levi_civita(CPU).double()
    R, current = 0.4, 3.0
    mid, dl = tensors.circle(R, 0.0, 400, +1, CPU)
    z = torch.linspace(-1, 1, 21, dtype=torch.float64)
    axis = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    B = equation.biot_savart(axis, mid.double(), dl.double(), eps, current)
    exact = equation.MU0 * current * R**2 / (2 * (R**2 + z**2) ** 1.5)
    assert torch.allclose(B[:, 2], exact, rtol=1e-4)  # on-axis field of a current loop
    assert B[:, :2].abs().max() < 1e-6 * exact.max()  # along the axis by symmetry
    assert torch.allclose(equation.loop_axis_field(z, R, [0.0], [current]), exact)


def test_long_straight_wire():
    eps = tensors.levi_civita(CPU).double()
    mid, dl = tensors.straight_wire(200.0, 20000, CPU)
    r = torch.tensor([0.1, 0.3, 1.0], dtype=torch.float64)
    points = torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], dim=-1)  # on the x axis, wire along z
    B = equation.biot_savart(points, mid.double(), dl.double(), eps, 5.0)
    assert torch.allclose(B[:, 1], equation.wire_field(r, 5.0), rtol=1e-3)  # Ampère: μ₀I/2πr, +y at +x
    assert B[:, [0, 2]].abs().max() < 1e-6 * B.abs().max()
