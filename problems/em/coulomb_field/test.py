"""Physics checks, written first (TDD): what any correct Coulomb-field code must satisfy.

Run with `pytest problems/em/coulomb_field/test.py`. A point charge gives a radial inverse-square field (Coulomb's
law) and, by superposition, a dipole's field on its bisector matches the closed form.
"""

import torch

from . import equation


def test_single_charge_field_is_inverse_square():
    points = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 4.0]])
    E = equation.electric_field(points, torch.tensor([[0.0, 0.0]]), torch.tensor([2.5]), k=1.0)
    r = points.norm(dim=-1)
    assert torch.allclose(E.norm(dim=-1), 2.5 / r**2)  # Coulomb's law: |E| = kq/r²
    assert torch.allclose(E / E.norm(dim=-1, keepdim=True), points / r[:, None])  # radially outward


def test_dipole_field_on_bisector():
    d, q = 0.5, 1.5
    charge_pos = torch.tensor([[d, 0.0], [-d, 0.0]], dtype=torch.float64)
    y = torch.linspace(-3, 3, 13, dtype=torch.float64)
    points = torch.stack([torch.zeros_like(y), y], dim=-1)
    E = equation.electric_field(points, charge_pos, torch.tensor([q, -q], dtype=torch.float64))
    expected = -2 * q * d / (d**2 + y**2) ** 1.5  # superposition: from + (at x = +d) towards − (at x = −d)
    assert torch.allclose(E[:, 0], expected)
    assert torch.allclose(E[:, 1], torch.zeros_like(y), atol=1e-12)  # mirror symmetry cancels E_y
