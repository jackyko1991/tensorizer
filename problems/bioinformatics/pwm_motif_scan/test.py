"""Physics checks, written first (TDD): what any correct motif-scanning code must satisfy.

Run with `pytest problems/bioinformatics/pwm_motif_scan/test.py`. Each test is an exact invariant: PWM columns are
probability distributions, information content lies in [0, 2] bits, the minus strand is the reverse complement, and
p-values are uniform under the null.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _pwm(n_train=40, strength=0.9, seed=0):
    gen = torch.Generator().manual_seed(seed)
    M = tensors.true_motif(strength, CPU)
    sites = tensors.sample_sites(M, n_train, gen)
    return M, solve.build_pwm(sites, 0.5, tensors.background(CPU)), gen


def test_columns_sum_to_one_and_ic_in_range():
    M, pwm, _ = _pwm()
    assert torch.allclose(M.sum(dim=1), torch.ones(M.shape[0]))
    assert torch.allclose(pwm["P"].sum(dim=1), torch.ones(M.shape[0]))
    assert (pwm["P"] > 0).all()
    assert pwm["C"].sum(dim=1).eq(40).all()
    IC = pwm["IC"]
    assert (IC >= 0).all() and (IC <= 2).all()
    # with q uniform, IC = 2 - entropy
    H = -(pwm["P"] * torch.log2(pwm["P"])).sum(dim=1)
    assert torch.allclose(IC, 2 - H, atol=1e-6)


def test_information_content_limits():
    q = tensors.background(CPU)
    assert torch.allclose(equation.information_content(q[None].clone(), q), torch.zeros(1), atol=1e-7)
    near_pure = equation.frequencies(torch.tensor([[1e6, 0.0, 0.0, 0.0]]), 1e-3)
    assert equation.information_content(near_pure, q).item() == pytest.approx(2.0, abs=1e-3)


def test_reverse_complement_matrix_scores_the_other_strand():
    _, pwm, gen = _pwm()
    seq = torch.randint(0, 4, (200,), generator=gen)
    scores = solve.scan_both_strands(seq, pwm["W"])  # (2, n)
    rc = tensors.reverse_complement(seq)
    scores_rc = solve.scan_both_strands(rc, pwm["W"])
    # the minus-strand score at n equals the plus-strand score of the reverse complement at the mirrored position
    assert torch.allclose(scores[1], scores_rc[0].flip(0), atol=1e-5)


def test_scan_finds_planted_sites_on_both_strands():
    M, pwm, gen = _pwm(n_train=30, seed=0)
    seq, starts, strands = tensors.sequence_with_sites(M, 5000, 8, gen)
    scores = solve.scan_both_strands(seq, pwm["W"])
    null = solve.scan_both_strands(tensors.shuffles(seq, 20, gen), pwm["W"])
    p = equation.p_values(scores, null)
    hits = solve.call_hits(p, 1e-4)
    found = solve.match(hits, starts, strands)
    assert found.float().mean() >= 0.75
    assert found[strands > 0].any() and found[strands < 0].any()
    # few false positives: about p · 2(L - m + 1) ≈ 1 expected
    others = [h for h in hits if not any(abs(h[0] - int(s)) < M.shape[0] for s in starts)]
    assert len(others) <= 4


def test_p_values_are_uniform_ish_on_null():
    _, pwm, gen = _pwm()
    seq = torch.randint(0, 4, (4000,), generator=gen)
    null = solve.scan_both_strands(tensors.shuffles(seq, 10, gen), pwm["W"])
    fresh = solve.scan_both_strands(torch.randint(0, 4, (4000,), generator=gen), pwm["W"])
    p = equation.p_values(fresh, null)
    assert (p <= 0.1).float().mean().item() == pytest.approx(0.1, abs=0.03)
