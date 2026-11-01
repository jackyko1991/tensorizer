"""Physics checks, written first (TDD): what any correct Hubble-expansion code must satisfy.

Run with `pytest problems/astro/hubble_expansion/test.py`. Each test is a physical law, not an implementation
detail: every observer sees v = H d, H equals ȧ/a and redshift is the ratio of scale factors.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("model", ["matter", "de Sitter"])
def test_hubble_law_for_every_observer(model):
    H0 = tensors.hubble_constant(70.0)
    q = tensors.galaxy_grid(5, 500.0, CPU).double()
    t = torch.linspace(equation.time_of(0.5, model, H0), equation.time_of(1.5, model, H0), 50, dtype=torch.float64)
    for observer in range(len(q)):
        sol = solve.solve(q, t, model, H0, observer)
        # v_rel = H d holds galaxy by galaxy, not just on average
        assert torch.allclose(sol["v_rel"], sol["H"][:, None, None] * sol["x_rel"], atol=1e-9)
        assert torch.allclose(solve.fitted_hubble_rate(sol["x_rel"], sol["v_rel"]), sol["H"], rtol=1e-9)


@pytest.mark.parametrize("model", ["matter", "de Sitter"])
def test_hubble_rate_is_adot_over_a(model):
    H0 = tensors.hubble_constant(70.0)
    t = torch.linspace(5.0, 20.0, 2001, dtype=torch.float64)
    a = equation.scale_factor(t, model, H0)
    a_dot = torch.gradient(a, spacing=(t,))[0]
    H = equation.hubble_rate(t, model, H0)
    assert torch.allclose((a_dot / a)[1:-1], H[1:-1], rtol=1e-5)  # Friedmann: H = ȧ/a
    assert equation.scale_factor(torch.tensor(equation.today(model, H0)), model, H0) == pytest.approx(1.0)


def test_redshift_is_ratio_of_scale_factors():
    # 1 + z = a_obs / a_emit
    assert equation.redshift(torch.tensor(1.5), torch.tensor(0.5)).item() == pytest.approx(2.0)
