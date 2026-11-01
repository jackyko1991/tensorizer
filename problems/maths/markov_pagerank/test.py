"""Physics checks, written first (TDD): what any correct PageRank code must satisfy.

Run with `pytest problems/maths/markov_pagerank/test.py`. Each test is a mathematical invariant, not an implementation
detail: probability is conserved (column-stochastic matrix, ranks sum to 1), the stationary vector is the eigenvector
for eigenvalue 1, and power iteration contracts at rate |λ₂| ≤ d.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def chain():
    links, _, _ = tensors.web_graph(25, 3, 1, CPU)
    links = links.double()
    s = equation.transition(links)
    ds = torch.tensor([0.5, 0.85, 0.95], dtype=s.dtype)
    g = equation.google(s, ds)
    history, residual = solve.power_iteration(g, 200)
    return links, s, ds, g, history, residual


def test_transition_is_column_stochastic_with_dangling_pages(chain):
    links, s, _, g, _, _ = chain
    assert (links.sum(0) == 0).sum() == 3
    assert torch.allclose(s.sum(0), torch.ones(25, dtype=s.dtype))  # column-stochastic: probability is conserved
    assert torch.allclose(g.sum(1), torch.ones(3, 25, dtype=g.dtype))


def test_ranks_sum_to_one_and_are_non_negative(chain):
    _, _, ds, _, history, _ = chain
    pi = history[-1]  # (B, N)
    assert torch.allclose(pi.sum(-1), torch.ones(3, dtype=pi.dtype), atol=1e-12)
    assert (pi >= (1 - ds[:, None]) / 25 - 1e-12).all()  # teleporting alone gives every page (1 − d)/N


def test_power_iteration_matches_eigenvector(chain):
    _, _, _, g, history, _ = chain
    for b in range(3):
        pi_eig, mags = solve.eig_stationary(g[b])
        assert mags[0].item() == pytest.approx(1.0, abs=1e-12)
        assert torch.allclose(history[-1, b], pi_eig, atol=1e-10)


def test_convergence_factor_is_lambda2_and_at_most_d(chain):
    _, _, ds, g, _, residual = chain
    factor = solve.convergence_factor(residual)
    for b, d in enumerate(ds.tolist()):
        _, mags = solve.eig_stationary(g[b])
        assert mags[1].item() <= d + 1e-9
        assert factor[b].item() <= d + 0.02
        assert factor[b].item() == pytest.approx(mags[1].item(), abs=0.05)


def test_monte_carlo_surfers_match_power_iteration(chain):
    _, s, ds, _, history, _ = chain
    occupancy = solve.monte_carlo(s, 0.85, 5000, 60, seed=0)
    assert torch.allclose(occupancy.sum(-1), torch.ones(61))
    estimate = occupancy[30:].mean(0).double()
    pi = history[-1, 1]
    assert torch.corrcoef(torch.stack([estimate, pi]))[0, 1] > 0.99
    assert solve.spearman(estimate, pi) > 0.9
    assert (estimate - pi).abs().sum() < 0.05
