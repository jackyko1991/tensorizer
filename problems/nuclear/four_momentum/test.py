"""Physics checks, written first (TDD): what any correct four-momentum code must satisfy.

Run with `pytest problems/nuclear/four_momentum/test.py`. Each test is a physical law, not an implementation detail:
boosts preserve the Minkowski metric and every mass, four-momentum is conserved in the decay and the rest-frame
momentum matches the two-body formula.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def decay(name: str, p_lab: float = 1.7):
    _, big_m, m1, m2, _, _ = tensors.DECAYS[name]
    eta = equation.metric(CPU)
    parent = tensors.parent(big_m, p_lab, CPU)
    theta = tensors.decay_angles(37, CPU)
    p_star = equation.rest_frame_momentum(big_m, m1, m2)
    r1, r2 = tensors.rest_frame_daughters(theta, p_star, big_m, m1, m2)
    l1, l2 = solve.to_lab(parent, r1, r2)
    return big_m, m1, m2, eta, parent, r1, r2, l1, l2


def test_boost_preserves_metric_and_masses():
    for name in tensors.DECAYS:
        big_m, m1, m2, eta, parent, _, _, l1, l2 = decay(name)
        lam = equation.boost(parent[1:] / parent[0])
        assert torch.allclose(torch.einsum("am,ab,bn->mn", lam, eta, lam), eta, atol=1e-12)  # Lorentz: Λᵀ η Λ = η
        assert torch.allclose(
            equation.minkowski(l1, l1, eta), torch.full((37,), m1**2, dtype=torch.float64), atol=1e-12
        )
        assert torch.allclose(
            equation.minkowski(l2, l2, eta), torch.full((37,), m2**2, dtype=torch.float64), atol=1e-12
        )
        assert math.isclose(equation.invariant_mass(parent, eta).item(), big_m, rel_tol=1e-12)


def test_invariant_mass_and_momentum_conservation():
    for name in tensors.DECAYS:
        big_m, _, _, eta, parent, _, _, l1, l2 = decay(name)
        assert torch.allclose(solve.reconstruct(l1, l2, eta), torch.full((37,), big_m, dtype=torch.float64))
        assert torch.allclose(l1 + l2, parent.expand_as(l1), atol=1e-12)  # four-momentum conservation


def test_rest_frame_momentum_formula():
    big_m, m1, m2, _, _, r1, r2, _, _ = decay("lambda")
    p_star = equation.rest_frame_momentum(big_m, m1, m2)
    assert math.isclose(p_star, 0.101, abs_tol=1e-3)  # Λ → p π⁻: p* ≈ 101 MeV
    assert torch.allclose(r1[:, 0] + r2[:, 0], torch.full((37,), big_m, dtype=torch.float64))  # E1 + E2 = M
    assert torch.allclose(r1[:, 1:].norm(dim=1), torch.full((37,), p_star, dtype=torch.float64))
    assert math.isclose(equation.rest_frame_momentum(0.134977, 0.0, 0.0), 0.134977 / 2)  # π⁰ → γγ: p* = M/2
