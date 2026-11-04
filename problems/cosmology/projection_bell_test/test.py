"""Checks, written first (TDD): Bell correlations of the projection race, step 5 of the projection hypothesis.

Run with `pytest problems/cosmology/projection_bell_test/test.py`. The hypothesis is speculative; these tests check
the mathematics. The singlet gives E(a, b) = -a.b; a race run jointly on the shared state Psi with one hidden complex
dimension per outcome (m = 1) gives exactly that, reaches Tsirelson's 2 sqrt 2 and never signals. Races run by each
party on locally accessible hidden coordinates are local hidden-variable models and obey CHSH S <= 2. A joint race
with m > 1 breaks Tsirelson's bound and, for a partly entangled state, lets one party signal to the other.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")
TSIRELSON = 2 * math.sqrt(2)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_singlet_correlation_is_minus_cosine():
    a, b = tensors.random_directions(500, gen(1), CPU), tensors.random_directions(500, gen(2), CPU)
    P = equation.joint_born(equation.singlet(CPU), a, b)  # (B, 2, 2)
    assert torch.allclose(P.sum((-1, -2)), torch.ones(500, dtype=torch.float64))
    assert torch.allclose(equation.correlation(P), -(a * b).sum(-1), atol=1e-12)


def test_joint_race_with_one_hidden_dimension_is_quantum_mechanics():
    theta = torch.linspace(0, math.pi, 13, dtype=torch.float64)
    a, b = tensors.planar_pairs(theta, CPU)
    w = equation.joint_born(equation.singlet(CPU), a, b).flatten(-2)  # (B, 4)
    P = equation.race_probabilities(w, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(equation.correlation(P.view(-1, 2, 2)), -torch.cos(theta), atol=1e-9)
    E_mc = equation.joint_race_correlation(w, tensors.hidden_block_energies(40_000, 4, 1, 32, gen(3), CPU))
    assert (E_mc + torch.cos(theta)).abs().max() < 5 / math.sqrt(40_000) * 2


def test_chsh_reaches_tsirelson_for_the_joint_model_and_never_exceeds_it():
    S_opt = equation.chsh(lambda a, b: equation.correlation(equation.joint_born(equation.singlet(CPU), a, b)),
                          *tensors.optimal_settings(CPU))  # fmt: skip
    assert torch.allclose(S_opt, torch.tensor([TSIRELSON], dtype=torch.float64), atol=1e-12)
    settings = tensors.random_settings(3000, gen(4), CPU)
    S = equation.chsh(lambda a, b: equation.correlation(equation.joint_born(equation.singlet(CPU), a, b)), *settings)
    assert S.max() <= TSIRELSON + 1e-9


@pytest.mark.parametrize("m", [1.0, 2.0, math.inf])
def test_local_race_obeys_chsh(m):
    n = tensors.shared_directions(4000, 64, gen(5), CPU)  # hidden direction both parties can read
    settings = tensors.random_settings(500, gen(6), CPU)
    S = equation.chsh(lambda a, b: equation.local_correlation(a, b, n, m), *settings)
    assert S.max() <= 2 + 1e-9  # Bell / CHSH: any local hidden-variable model
    S_opt = equation.chsh(lambda a, b: equation.local_correlation(a, b, n, m), *tensors.optimal_settings(CPU))
    assert S_opt.item() <= 2 + 1e-9


def test_local_race_closed_forms():
    n = tensors.shared_directions(200_000, 16, gen(7), CPU)
    theta = torch.linspace(0, math.pi, 9, dtype=torch.float64)
    a, b = tensors.planar_pairs(theta, CPU)
    E1 = equation.local_correlation(a, b, n, 1.0)  # race with Born weights per party: -cos(theta) / 3
    assert torch.allclose(E1, -torch.cos(theta) / 3, atol=0.01)
    Einf = equation.local_correlation(a, b, n, math.inf)  # deterministic sign(a.n): Bell's 1964 linear model
    assert torch.allclose(Einf, -(1 - 2 * theta / math.pi), atol=0.01)


def test_shared_direction_is_uniform_for_any_dimension():
    for D in (8, 512):
        n = tensors.shared_directions(30_000, D, gen(8), CPU)
        assert torch.allclose(n.norm(dim=-1), torch.ones(30_000, dtype=torch.float64))
        cov = n.T @ n / n.shape[0]
        assert torch.allclose(cov, torch.eye(3, dtype=torch.float64) / 3, atol=0.01)


def test_no_signalling_for_the_joint_born_race():
    a = tensors.random_directions(1, gen(9), CPU).expand(200, 3)
    b = tensors.random_directions(200, gen(10), CPU)
    for psi in (equation.singlet(CPU), equation.partly_entangled(0.3, CPU)):
        P = equation.race_probabilities(equation.joint_born(psi, a, b).flatten(-2), torch.tensor(1.0)).view(-1, 2, 2)
        pA = P.sum(-1)[:, 0]  # Alice's marginal P(+ | a, b) for 200 settings of Bob
        assert (pA - pA[0]).abs().max() < 1e-9


def test_joint_race_beyond_born_breaks_tsirelson_and_signals():
    m = torch.tensor([1.0, 1.5, 2.0, 4.0], dtype=torch.float64)
    S = equation.joint_race_chsh(equation.singlet(CPU), m, *tensors.optimal_settings(CPU))  # (M,)
    assert abs(S[0].item() - TSIRELSON) < 1e-9
    assert (S[1:] > TSIRELSON + 1e-3).all() and (S.diff() > 0).all() and (S < 4).all()
    sig = equation.max_signalling(equation.partly_entangled(0.3, CPU), m, gen(11), CPU)  # (M,)
    assert sig[0] < 1e-9 and (sig[1:] > 1e-3).all()


def test_bell_data_rule_out_local_models():
    rows = tensors.bell_tests()
    with_S = [r for r in rows if r["S"]]
    assert len(with_S) >= 3
    for r in with_S:
        S, sig = float(r["S"]), float(r["sigma"])
        assert S > 2 and S < TSIRELSON + 2 * sig  # above every local model, not above quantum mechanics
    assert any(float(r["p_value"]) < 1e-30 for r in rows if r["p_value"])


@pytest.mark.slow
def test_run_accepts_parameters(tmp_path):
    from . import main

    params = {"D": 32, "m_max": 3, "settings": 300, "trials": 4000}
    out = main.run(params, tmp_path)
    assert out.exists() and out.stat().st_size > 0
