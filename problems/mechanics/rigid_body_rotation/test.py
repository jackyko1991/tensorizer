"""Physics checks, written first (TDD): what any correct rigid-body code must satisfy.

Run with `pytest problems/mechanics/rigid_body_rotation/test.py`. Torque-free motion conserves kinetic energy and
angular momentum (fixed in space), and the tennis-racket theorem holds: spin about the major and minor axes is stable,
about the intermediate axis it flips.
"""

import pytest
import torch

from . import equation, solve, tensors


def test_energy_and_angular_momentum_are_conserved():
    inertia = tensors.box_inertia(2.0, 3.0, 2.0, 1.0)
    y0 = tensors.initial_state("intermediate", 5.0, 0.01)
    sol = solve.solve(y0, inertia, t_max=10.0)
    T = equation.kinetic_energy(sol["omega"], inertia)
    L_body = equation.angular_momentum(sol["omega"], inertia)
    L_sq = torch.einsum("ti,ti->t", L_body, L_body)
    assert torch.allclose(T, T[0].expand_as(T), rtol=1e-8)
    assert torch.allclose(L_sq, L_sq[0].expand_as(L_sq), rtol=1e-8)
    L_world = equation.world_angular_momentum(sol["R"], sol["omega"], inertia)  # (T, 3) fixed in space
    assert torch.allclose(L_world, L_world[0].expand_as(L_world), atol=1e-6)


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_major_and_minor_axes_stable_intermediate_flips():
    inertia = tensors.box_inertia(1.0, 3.0, 2.0, 1.0)
    for axis in ("major", "minor"):
        w = solve.solve(tensors.initial_state(axis, 5.0, 0.01), inertia, t_max=15.0)["omega"]
        main = tensors.AXES[axis]
        others = [i for i in range(3) if i != main]
        assert w[:, main].min() > 0.99 * 5.0, axis  # spin barely changes
        assert w[:, others].abs().max() < 0.2, axis  # wobble stays small
    w = solve.solve(tensors.initial_state("intermediate", 5.0, 0.01), inertia, t_max=15.0)["omega"]
    assert w[:, 1].min() < -4.5  # the intermediate-axis spin reverses: the box flipped over


def test_euler_equations_match_component_form():
    """The ε-einsum equals the textbook (I_2 − I_3) ω_2 ω_3 / I_1 etc. for a diagonal inertia tensor."""
    inertia = tensors.box_inertia(1.0, 3.0, 2.0, 1.0)
    I1, I2, I3 = inertia.diagonal().tolist()
    w = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    w_dot = torch.linalg.solve(inertia, equation.gyroscopic_torque(w, inertia, equation.levi_civita()))
    expected = torch.stack([(I2 - I3) * w[1] * w[2] / I1, (I3 - I1) * w[2] * w[0] / I2, (I1 - I2) * w[0] * w[1] / I3])
    assert torch.allclose(w_dot, expected, atol=1e-12)
