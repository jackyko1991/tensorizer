"""Physics checks, written first (TDD): what any correct k-mer and GC-skew code must satisfy.

Run with `pytest problems/bioinformatics/kmer_gc_skew/test.py`. Each test is an exact invariant: k-mer counts sum to
L − k + 1 and match a plain Counter, and the cumulative skew minimum marks the replication origin with the sign set
by the leading strand.
"""

from collections import Counter

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_kmer_counts_match_python_counter(k):
    seq, _, _ = tensors.synthetic_genome(3000, 0.3, 0.5, 0.05, seed=1, device=CPU)
    X = tensors.one_hot(seq)
    counts = solve.count_kmers(X, tensors.place_values(k, CPU))
    assert counts.shape == (4**k,)
    assert counts.sum().item() == len(seq) - k + 1

    text = "".join(tensors.BASES[a] for a in seq.tolist())
    expected = Counter(text[n : n + k] for n in range(len(text) - k + 1))
    names = tensors.kmer_names(k)
    assert {names[m]: c for m, c in enumerate(counts.tolist()) if c} == dict(expected)


def test_kmer_index_reads_words_in_base_four():
    seq = torch.tensor([0, 1, 2, 3, 3])  # A C G T T
    m = equation.kmer_index(equation.unfold(tensors.one_hot(seq), 3), tensors.place_values(3, CPU))
    names = tensors.kmer_names(3)
    assert [names[i] for i in m.tolist()] == ["ACG", "CGT", "GTT"]


def test_gc_content_and_skew_of_known_window():
    X = tensors.one_hot(torch.tensor([2, 2, 2, 1, 0, 3, 0, 3]))  # GGGC ATAT
    is_gc, g_minus_c = tensors.gc_strand_weights(CPU)
    n = equation.window_base_counts(X, 4, 4)
    assert torch.allclose(equation.gc_content(n, is_gc), torch.tensor([1.0, 0.0]))
    assert torch.allclose(equation.gc_skew(n, is_gc, g_minus_c), torch.tensor([0.5, 0.0]))


@pytest.mark.parametrize("origin", [0.2, 0.55])
def test_cumulative_skew_minimum_at_planted_origin(origin):
    L = 200_000
    seq, ori, ter = tensors.synthetic_genome(L, origin, 0.5, 0.05, seed=3, device=CPU)
    _, found_ori, found_ter = solve.find_origin(tensors.one_hot(seq), tensors.gc_strand_weights(CPU)[1])
    assert solve.circular_distance(found_ori, ori, L) < 0.01 * L
    assert solve.circular_distance(found_ter, ter, L) < 0.01 * L


def test_window_skew_sign_follows_strand():
    L = 100_000
    seq, ori, _ = tensors.synthetic_genome(L, 0.0, 0.5, 0.1, seed=4, device=CPU)
    is_gc, g_minus_c = tensors.gc_strand_weights(CPU)
    win = solve.windows(tensors.one_hot(seq), 2000, is_gc, g_minus_c)
    leading = win["start"] < L // 2 - 2000
    lagging = (win["start"] > L // 2) & (win["start"] < L - 2000)
    assert win["skew"][leading].mean() > 0.05
    assert win["skew"][lagging].mean() < -0.05
