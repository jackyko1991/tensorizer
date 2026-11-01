"""Physics checks, written first (TDD): what any correct population-genetics code must satisfy.

Run with `pytest problems/bioinformatics/wright_fisher_drift/test.py`. Each test is a law of population genetics:
Hardy–Weinberg frequencies sum to one, a neutral allele fixes with probability 1/2N, selection follows Kimura's
formula, and heterozygosity decays as (1 − 1/2N)^t.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_hardy_weinberg_sums_to_one_and_sampling_matches():
    p = torch.linspace(0, 1, 11, dtype=torch.float64)
    G = equation.hardy_weinberg(p)
    assert torch.allclose(G.sum(dim=-1), torch.ones(11, dtype=torch.float64))
    gen = torch.Generator().manual_seed(0)
    obs = solve.sample_genotypes(torch.full((200,), 0.3, dtype=torch.float64), 500, gen).mean(dim=0)
    assert torch.allclose(obs, equation.hardy_weinberg(torch.tensor(0.3, dtype=torch.float64)), atol=0.01)


def test_neutral_fixation_fraction_is_one_over_2n():
    N, R = 25, 40_000
    gen = torch.Generator().manual_seed(1)
    s = torch.zeros(1, dtype=torch.float64)
    u = solve.fixation(tensors.new_mutants(1, R, N, CPU), N, s, gen, max_gen=40 * N)
    se = (1 / (2 * N) * (1 - 1 / (2 * N)) / R) ** 0.5
    assert abs(u.item() - 1 / (2 * N)) < 4 * se


def test_fixation_with_selection_matches_kimura():
    N, R = 25, 20_000
    gen = torch.Generator().manual_seed(2)
    s = torch.tensor([4.0, 8.0], dtype=torch.float64) / (4 * N)
    u = solve.fixation(tensors.new_mutants(2, R, N, CPU), N, s, gen, max_gen=40 * N)
    u_k = equation.kimura_fixation(1 / (2 * N), N, s)
    assert torch.allclose(u, u_k, rtol=0.15)
    assert equation.kimura_fixation(0.3, N, torch.zeros(1)).item() == pytest.approx(0.3)


def test_heterozygosity_decays_as_one_minus_one_over_2n_to_the_t():
    N, R, G = 40, 20_000, 120
    gen = torch.Generator().manual_seed(3)
    p = solve.simulate(tensors.initial_frequencies(R, 0.5, CPU), N, 0.0, G, gen)
    assert p.shape == (R, G + 1)
    H = equation.heterozygosity(p).mean(dim=0)
    t = torch.arange(G + 1, dtype=torch.float64)
    H_th = equation.expected_heterozygosity(0.5, N, t)
    assert torch.allclose(H, H_th, rtol=0.05)
    # the mean allele frequency does not drift without selection
    assert p[:, -1].mean().item() == pytest.approx(0.5, abs=0.01)
