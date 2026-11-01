"""Physics checks, written first (TDD): what any correct repetition-code code must satisfy.

Run with `pytest problems/quantum/error_correction_threshold/test.py`. Each test is a physical law, not an
implementation detail: majority vote follows the exact binomial law, the space–time decoder is exact minimum weight,
and curves of different distance cross at a threshold below which bigger codes suppress errors.
"""

import itertools
import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_binomial_formula_small_cases():
    p = torch.tensor([0.01, 0.1, 0.3], dtype=torch.float64)
    assert torch.allclose(equation.binomial_logical_error(p, 3), 3 * p**2 - 2 * p**3)  # d = 3: P_L = 3p² − 2p³
    assert torch.allclose(equation.binomial_logical_error(torch.tensor(0.5), 7), torch.tensor(0.5, dtype=torch.float64))


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("p", [0.1, 0.25])
def test_majority_vote_matches_exact_binomial(d, p):
    gen = torch.Generator().manual_seed(d)
    shots = 200_000
    p_l, _ = solve.run_point(d, p, 0.0, 1, shots, gen, CPU)
    exact = equation.binomial_logical_error(torch.tensor(p), d).item()
    sigma = math.sqrt(exact * (1 - exact) / shots)
    assert abs(p_l - exact) < 4 * sigma + 1e-12  # Monte Carlo agrees with the binomial law within 4σ


def test_space_time_decoder_reduces_to_majority_vote_without_misreads():
    """One round, checks that never lie (huge misread weight): min-weight decoding = majority vote."""
    d, n = 7, 5000
    gen = torch.Generator().manual_seed(1)
    e, _ = tensors.sample_noise(n, 1, d, 0.3, 0.0, gen, CPU)
    x = equation.accumulate(e)
    sigma = equation.syndrome(x, equation.parity_check(d))
    fails = solve.decode_fails(x, sigma, tensors.mismatch_table(d).float(), 1.0, 100.0)
    assert torch.equal(fails.bool(), equation.majority_vote_fails(x[:, -1]))


def test_space_time_decoder_is_exact_minimum_weight():
    """Compare the dynamic programme with brute force over every data-error history (d = 3, R = 3)."""
    d, rounds, w_d, w_m = 3, 3, 1.0, 1.3
    h = equation.parity_check(d)
    gen = torch.Generator().manual_seed(2)
    e, m = tensors.sample_noise(300, rounds, d, 0.2, 0.2, gen, CPU)
    x = equation.accumulate(e)
    sigma = equation.syndrome(x, h, m)
    fails = solve.decode_fails(x, sigma, tensors.mismatch_table(d).float(), w_d, w_m)

    histories = torch.tensor(list(itertools.product([0, 1], repeat=rounds * d))).view(-1, rounds, d)  # (2^9, R, d)
    xh = equation.accumulate(histories.bool())  # candidate accumulated errors
    sh = equation.syndrome(xh, h)  # their noiseless syndromes
    for s in range(x.shape[0]):
        cost = w_d * histories.sum((1, 2)) + w_m * (sh != sigma[s]).sum((1, 2))  # (2^9,)
        final = xh[:, -1]
        best_true = cost[(final == x[s, -1]).all(1)].min()
        best_flip = cost[(final == 1 - x[s, -1]).all(1)].min()
        expected = 1.0 if best_flip < best_true else (0.5 if best_flip == best_true else 0.0)
        assert fails[s].item() == pytest.approx(expected)


def test_curves_cross_near_threshold_and_lambda_above_one_below_it():
    ds = [3, 5, 7]
    ps = torch.tensor([0.04, 0.07, 0.09, 0.11, 0.13, 0.16], dtype=torch.float64)
    res = solve.sweep(ds, ps, 1.0, "phenomenological", 6000, seed=3, device=CPU)
    pl = res["p_logical"]
    for a in range(len(ds) - 1):
        pc = solve.crossing(ps, pl[a], pl[a + 1])
        assert 0.08 < pc < 0.14, f"d = {ds[a]}/{ds[a + 1]} crossing at {pc}"
    lam = solve.suppression(res["eps"], pl, res["shots"])
    assert torch.all(lam[:, 0] > 1.5)  # p = 0.04, well below threshold: strong suppression
    assert torch.all(pl[:, -1][1:] > pl[:, -1][:-1])  # p = 0.16, above threshold: bigger codes are worse
