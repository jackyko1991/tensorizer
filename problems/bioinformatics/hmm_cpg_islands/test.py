"""Physics checks, written first (TDD): what any correct hidden-Markov-model code must satisfy.

Run with `pytest problems/bioinformatics/hmm_cpg_islands/test.py`. Each test is an exact invariant: the forward
likelihood is the sum over all paths, posteriors sum to one, Viterbi is the brute-force argmax, and Baum–Welch never
decreases the likelihood.
"""

import itertools
import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def tiny_model():
    pi = torch.tensor([0.6, 0.4], dtype=torch.float64)
    A = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
    E = torch.tensor([[0.4, 0.1, 0.2, 0.3], [0.1, 0.4, 0.3, 0.2]], dtype=torch.float64)
    return pi, A, E


def brute_force(x: list[int], pi, A, E):
    """Joint probability of every state path: dict path → P(x, z)."""
    out = {}
    for z in itertools.product(range(2), repeat=len(x)):
        p = pi[z[0]] * E[z[0], x[0]]
        for t in range(1, len(x)):
            p = p * A[z[t - 1], z[t]] * E[z[t], x[t]]
        out[z] = float(p)
    return out


def test_forward_likelihood_equals_sum_over_paths_and_posteriors_sum_to_one():
    pi, A, E = tiny_model()
    seqs = [[0, 2, 1, 1, 3], [1, 1, 2, 2, 0]]
    x = tensors.one_hot(torch.tensor(seqs))
    gamma, xi, log_px = solve.e_step(x, pi, A, E)
    assert torch.allclose(gamma.sum(-1), torch.ones(2, 5, dtype=torch.float64))
    for b, s in enumerate(seqs):
        joint = brute_force(s, pi, A, E)
        total = sum(joint.values())
        assert math.exp(log_px[b]) == pytest.approx(total, rel=1e-10)
        for t in range(5):  # posterior marginal P(z_t = 1 | x)
            marginal = sum(p for z, p in joint.items() if z[t] == 1) / total
            assert gamma[b, t, 1].item() == pytest.approx(marginal, rel=1e-9)


def test_viterbi_equals_brute_force_argmax():
    pi, A, E = tiny_model()
    seqs = [[0, 2, 1, 1, 3, 1], [1, 1, 2, 2, 0, 3], [3, 0, 0, 1, 2, 1]]
    x = tensors.one_hot(torch.tensor(seqs))
    path = solve.viterbi(equation.log_emissions(x, E), pi, A)
    for b, s in enumerate(seqs):
        joint = brute_force(s, pi, A, E)
        assert tuple(path[b].tolist()) == max(joint, key=joint.get)


def test_viterbi_recovers_planted_islands():
    truth = tensors.true_model(0.7, CPU)
    codes, z = tensors.planted_sequences(6, 3000, 3, truth[2], seed=5)
    x = tensors.one_hot(codes)
    path = solve.viterbi(equation.log_emissions(x, truth[2]), truth[0], truth[1])
    accuracy = (path == z).double().mean().item()
    assert accuracy > 0.93
    recall = (path[z == 1] == 1).double().mean().item()
    assert recall > 0.85


def test_baum_welch_never_decreases_the_likelihood_and_learns_emissions():
    truth = tensors.true_model(0.7, CPU)
    codes, _ = tensors.planted_sequences(4, 2000, 2, truth[2], seed=2)
    models, log_lik, _ = solve.baum_welch(tensors.one_hot(codes), *tensors.initial_guess(CPU), 10)
    assert all(b >= a - 1e-6 for a, b in zip(log_lik, log_lik[1:]))
    E = models[-1][2]
    assert torch.allclose(E.sum(1), torch.ones(2, dtype=torch.float64))
    assert (E[1, 1:3].sum() - truth[2][1, 1:3].sum()).abs() < 0.06  # island GC content
    assert (E[0, 1:3].sum() - truth[2][0, 1:3].sum()).abs() < 0.03  # background GC content
