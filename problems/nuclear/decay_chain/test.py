"""Physics checks, written first (TDD): what any correct decay-chain code must satisfy.

Run with `pytest problems/nuclear/decay_chain/test.py`. Each test is a physical law, not an implementation detail:
the populations match the Bateman solution, the total number of nuclei is conserved and the chain reaches transient
and secular equilibrium.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def chain(t1: float = 10.0, t2: float = 1.0):
    lam = tensors.decay_constants([t1, t2], CPU)
    return lam, equation.bateman_matrix(lam), tensors.initial_populations(1000.0, 3, CPU)


def test_matches_two_member_bateman_solution():
    lam, a, n0 = chain()
    t = tensors.time_axis(50.0, 101, CPU)
    n = solve.exact(a, n0, t)
    l1, l2 = lam[0], lam[1]
    n1 = 1000.0 * torch.exp(-l1 * t)
    n2 = 1000.0 * l1 / (l2 - l1) * (torch.exp(-l1 * t) - torch.exp(-l2 * t))
    assert torch.allclose(n[:, 0], n1, rtol=1e-10)
    assert torch.allclose(n[:, 1], n2, rtol=1e-9, atol=1e-9)  # Bateman solution for the daughter


def test_conserves_nuclei_and_rk4_agrees():
    _, a, n0 = chain()
    t_rk, n_rk = solve.rk4(a, n0, 50.0, 500)
    n = solve.exact(a, n0, t_rk)
    # stable end member: total nuclei conserved
    assert torch.allclose(n.sum(dim=1), torch.full_like(t_rk, 1000.0), rtol=1e-12)
    assert torch.allclose(n_rk, n, atol=1e-6)


def test_transient_and_secular_equilibrium():
    lam, a, n0 = chain(10.0, 1.0)
    t = torch.tensor([150.0], dtype=torch.float64)
    act = equation.activity(solve.exact(a, n0, t), lam)[0]
    expected = lam[1] / (lam[1] - lam[0])
    assert math.isclose(act[1] / act[0], expected, rel_tol=1e-6)  # transient equilibrium: a2/a1 = λ2/(λ2 − λ1)
    lam, a, n0 = chain(10000.0, 1.0)  # very long-lived parent → secular: a2 = a1
    act = equation.activity(solve.exact(a, n0, torch.tensor([30.0], dtype=torch.float64)), lam)[0]
    assert math.isclose(act[1] / act[0], 1.0, rel_tol=2e-4)
