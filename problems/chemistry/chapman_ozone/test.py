"""Physics checks, written first (TDD): what any correct Chapman ozone-layer code must satisfy.

Run with `pytest problems/chemistry/chapman_ozone/test.py`. Each test is a physical law, not an implementation
detail: every reaction conserves oxygen atoms, the long-time state matches the analytic steady state, O ⇄ O3 is fast.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_stoichiometry_conserves_oxygen():
    assert torch.equal(equation.ATOMS @ equation.S, torch.zeros(4, dtype=torch.float64))  # each reaction conserves O
    # and dc/dt conserves atoms for any state
    c = torch.rand(5, 3, dtype=torch.float64) * torch.tensor([1e8, 1e17, 1e12], dtype=torch.float64)
    co = [torch.full((5,), v, dtype=torch.float64) for v in (1e-11, 1e-3, 1e-33, 1e-15, 1e17)]
    rate = equation.dcdt(c, *co) @ equation.ATOMS
    assert (rate.abs() / (equation.rates(c, *co).abs().sum(-1) * 3)).max() < 1e-14
    # Jacobian matches finite differences
    J = equation.jacobian(c, *co)
    h = 1e-6 * c
    for s in range(3):
        cp = c.clone()
        cp[:, s] += h[:, s]
        fd = (equation.dcdt(cp, *co) - equation.dcdt(c, *co)) / h[:, s : s + 1]
        assert torch.allclose(J[:, :, s], fd, rtol=1e-4, atol=1e-9 * fd.abs().max())


def test_steady_state_matches_analytic():
    z = torch.tensor([25.0, 30.0, 35.0, 40.0, 45.0, 50.0], dtype=torch.float64)
    times = tensors.time_grid(1e11, 200)
    c, (j1, j3, k2, k4, M) = solve.solve(z, times, 1.0)
    atoms = c @ equation.ATOMS
    assert ((atoms - atoms[0]) / atoms[0]).abs().max() < 1e-10  # total oxygen atoms conserved in time
    o, O3 = equation.steady_state(j1, j3, k2, k4, M, c[-1, :, 1])
    assert torch.allclose(c[-1, :, 2], O3, rtol=1e-3)
    assert torch.allclose(c[-1, :, 0], o, rtol=1e-3)
    # the textbook square-root formula is close where k4 [O] << j3
    approx = equation.chapman_approx(j1, j3, k2, k4, M, c[-1, :, 1])
    assert torch.allclose(approx, O3, rtol=5e-3)
    # time scales: O ⇄ O3 cycling is far faster than odd-oxygen relaxation
    fast, slow = solve.lifetimes(c[-1], j1, j3, k2, k4, M)
    assert (slow / fast > 1e3).all()
    # the layer: the maximum lies inside the range, not at an edge
    peak = int(c[-1, :, 2].argmax())
    assert 0 < peak < len(z) - 1
