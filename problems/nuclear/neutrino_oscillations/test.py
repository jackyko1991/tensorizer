"""Physics checks, written first (TDD): what any correct neutrino-oscillation code must satisfy.

Run with `pytest problems/nuclear/neutrino_oscillations/test.py`. Each test is a physical law, not an implementation
detail: the PMNS matrix is unitary so probabilities sum to one, the two-flavour limit is exact and CPT relates
P(α→β; δ) to P(β→α; −δ).
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_pmns_is_unitary_and_probabilities_sum_to_one():
    u = equation.pmns(*(math.radians(x) for x in (33.4, 8.6, 49.0, 195.0)))
    eye = torch.eye(3, dtype=torch.complex128)
    assert torch.allclose(u @ u.conj().T, eye, atol=1e-14)  # PMNS is unitary
    m2 = tensors.squared_masses(7.4e-5, 2.5e-3, CPU)
    l_over_e = torch.linspace(0.0, 40000.0, 500, dtype=torch.float64)
    p = equation.probability(u, equation.phases(m2, l_over_e))  # (K, β, α)
    assert torch.allclose(p.sum(dim=1), torch.ones(500, 3, dtype=torch.float64), atol=1e-12)  # Σ_β
    assert torch.allclose(p.sum(dim=2), torch.ones(500, 3, dtype=torch.float64), atol=1e-12)  # Σ_α
    assert torch.allclose(p[0], torch.eye(3, dtype=torch.float64), atol=1e-12)  # no change at L = 0


def test_two_flavour_limit():
    theta23, dm31 = math.radians(40.0), 2.5e-3
    u = equation.pmns(0.0, 0.0, theta23, 0.0)  # only ν_μ–ν_τ mixing
    m2 = tensors.squared_masses(0.0, dm31, CPU)
    dist = torch.linspace(1.0, 5000.0, 400, dtype=torch.float64)
    p = solve.oscillate(u, m2, dist, 1.0, alpha=1)
    # two-flavour limit: sin²2θ sin²(Δm²L/4E)
    assert torch.allclose(p[:, 2], equation.two_flavour(theta23, dm31, dist), atol=1e-12)
    assert torch.allclose(p[:, 0], torch.zeros_like(dist), atol=1e-12)


def test_cpt_relation():
    m2 = tensors.squared_masses(7.4e-5, 2.5e-3, CPU)
    l_over_e = torch.linspace(0.0, 3000.0, 100, dtype=torch.float64)
    angles = [math.radians(x) for x in (33.4, 8.6, 49.0)]
    p_plus = equation.probability(equation.pmns(*angles, 1.2), equation.phases(m2, l_over_e))
    p_minus = equation.probability(equation.pmns(*angles, -1.2), equation.phases(m2, l_over_e))
    assert torch.allclose(p_plus, p_minus.transpose(1, 2), atol=1e-12)  # P(α→β; δ) = P(β→α; −δ)
