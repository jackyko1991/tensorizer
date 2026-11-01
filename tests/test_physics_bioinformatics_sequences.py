"""Checks for sequence alignment, the CpG-island HMM and phylogenetic trees against brute force and known results."""

import itertools
import math
import random

import pytest
import torch

from problems.bioinformatics.hmm_cpg_islands import equation as hmm_eq
from problems.bioinformatics.hmm_cpg_islands import solve as hmm_solve
from problems.bioinformatics.hmm_cpg_islands import tensors as hmm_t
from problems.bioinformatics.phylogenetic_tree import equation as phy_eq
from problems.bioinformatics.phylogenetic_tree import solve as phy_solve
from problems.bioinformatics.phylogenetic_tree import tensors as phy_t
from problems.bioinformatics.sequence_alignment import equation as aln_eq
from problems.bioinformatics.sequence_alignment import solve as aln_solve
from problems.bioinformatics.sequence_alignment import tensors as aln_t

CPU = torch.device("cpu")

# --- sequence alignment ---


def reference_score(a: str, c: str, match, mismatch, go, ge, local: bool) -> float:
    """Gotoh's algorithm cell by cell in plain Python (no tensors)."""
    n, m, NEG = len(a), len(c), -math.inf
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    E = [[NEG] * (m + 1) for _ in range(n + 1)]
    F = [[NEG] * (m + 1) for _ in range(n + 1)]
    if not local:
        for j in range(1, m + 1):
            H[0][j] = E[0][j] = -(go + (j - 1) * ge)
        for i in range(1, n + 1):
            H[i][0] = F[i][0] = -(go + (i - 1) * ge)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i][j] = max(H[i][j - 1] - go, E[i][j - 1] - ge)
            F[i][j] = max(H[i - 1][j] - go, F[i - 1][j] - ge)
            s = match if a[i - 1] == c[j - 1] else mismatch
            H[i][j] = max(H[i - 1][j - 1] + s, E[i][j], F[i][j], 0.0 if local else NEG)
            best = max(best, H[i][j])
    return best if local else H[n][m]


def align(pairs: list[tuple[str, str]], match, mismatch, go, ge, local: bool):
    """Batched wavefront alignment of a list of string pairs: scores (B,), ends (B, 2) and the tables."""
    codes_a, len_a = aln_t.pad([aln_t.encode(a) for a, _ in pairs], CPU)
    codes_c, len_c = aln_t.pad([aln_t.encode(c) for _, c in pairs], CPU)
    W = aln_t.substitution_matrix(match, mismatch, CPU)
    S = torch.zeros(len(pairs), codes_a.shape[1] + 1, codes_c.shape[1] + 1)
    S[:, 1:, 1:] = aln_eq.substitution_scores(aln_t.one_hot(codes_a), W, aln_t.one_hot(codes_c))
    H, E, F = aln_solve.fill(S, go, ge, local)
    scores, ends = aln_solve.best_cells(H, len_a, len_c, local)
    return scores, ends, (H, E, F, S)


def score_alignment(top: str, bottom: str, match, mismatch, go, ge) -> float:
    """Score of a given gapped alignment, column by column."""
    total, gap_top, gap_bottom = 0.0, False, False
    for x, y in zip(top, bottom):
        if x == "-":
            total -= ge if gap_top else go
            gap_top, gap_bottom = True, False
        elif y == "-":
            total -= ge if gap_bottom else go
            gap_top, gap_bottom = False, True
        else:
            total += match if x == y else mismatch
            gap_top = gap_bottom = False
    return total


def random_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        ("".join(rng.choice("ACGT") for _ in range(rng.randint(1, 25))),
         "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 25))))
        for _ in range(n)
    ]  # fmt: skip


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("scheme", [(1, -1, 1, 1), (2, -1, 3, 1), (3, -2, 5, 0.5)])
def test_wavefront_matches_plain_python_dp(local, scheme):
    pairs = random_pairs(40, seed=hash(scheme) % 1000 + local)
    scores, _, _ = align(pairs, *scheme, local)
    expected = [reference_score(a, c, *scheme, local) for a, c in pairs]
    assert scores.tolist() == pytest.approx(expected)


def test_needleman_wunsch_textbook_example():
    # Wikipedia's Needleman–Wunsch example: match +1, mismatch −1, gap −1 → optimal score 0
    scores, ends, (H, E, F, S) = align([("GCATGCG", "GATTACA")], 1, -1, 1, 1, local=False)
    assert scores.item() == 0
    _, top, bottom = aln_solve.traceback(H[0], E[0], F[0], S[0], "GCATGCG", "GATTACA", (7, 7), 1, 1, False)
    assert top.replace("-", "") == "GCATGCG" and bottom.replace("-", "") == "GATTACA"
    assert score_alignment(top, bottom, 1, -1, 1, 1) == 0


def test_smith_waterman_textbook_example():
    # Wikipedia's Smith–Waterman example: match +3, mismatch −3, linear gap 2 → score 13, GTT-AC / GTTGAC
    a, c = "TGTTACGG", "GGTTGACTA"
    scores, ends, (H, E, F, S) = align([(a, c)], 3, -3, 2, 2, local=True)
    assert scores.item() == 13
    _, top, bottom = aln_solve.traceback(H[0], E[0], F[0], S[0], a, c, tuple(ends[0].tolist()), 2, 2, True)
    assert (top, bottom) == ("GTT-AC", "GTTGAC")


def test_traceback_reproduces_the_dp_score():
    scheme = (2, -1, 3, 1)
    pairs = random_pairs(20, seed=7)
    for local in (False, True):
        scores, ends, (H, E, F, S) = align(pairs, *scheme, local)
        for b, (a, c) in enumerate(pairs):
            _, top, bottom = aln_solve.traceback(
                H[b], E[b], F[b], S[b], a, c, tuple(ends[b].tolist()), scheme[2], scheme[3], local
            )
            assert len(top) == len(bottom)
            assert score_alignment(top, bottom, *scheme) == pytest.approx(scores[b].item())
            if not local:
                assert top.replace("-", "") == a and bottom.replace("-", "") == c


def test_smith_waterman_is_non_negative_and_finds_the_planted_motif():
    codes_a, len_a, codes_c, len_c, spans = aln_t.planted_pairs(24, 60, 0.1, 3, CPU)
    W = aln_t.substitution_matrix(2, -2, CPU)
    S = torch.zeros(24, codes_a.shape[1] + 1, codes_c.shape[1] + 1)
    S[:, 1:, 1:] = aln_eq.substitution_scores(aln_t.one_hot(codes_a), W, aln_t.one_hot(codes_c))
    H, E, F = aln_solve.fill(S, 3, 1, local=True)
    assert (H >= 0).all()
    scores, ends = aln_solve.best_cells(H, len_a, len_c, local=True)
    hits = 0
    for b in range(24):
        a, c = aln_t.decode(codes_a[b]), aln_t.decode(codes_c[b])
        path, _, _ = aln_solve.traceback(H[b], E[b], F[b], S[b], a, c, tuple(ends[b].tolist()), 3, 1, True)
        (i0, j0), (i1, j1) = path[0], path[-1]
        sa, ea, sc, ec = spans[b].tolist()
        overlap_a = max(0, min(i1, ea) - max(i0, sa)) / (ea - sa)
        overlap_c = max(0, min(j1, ec) - max(j0, sc)) / (ec - sc)
        hits += overlap_a > 0.8 and overlap_c > 0.8
    assert hits >= 22


# --- CpG-island HMM ---


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
    x = hmm_t.one_hot(torch.tensor(seqs))
    gamma, xi, log_px = hmm_solve.e_step(x, pi, A, E)
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
    x = hmm_t.one_hot(torch.tensor(seqs))
    path = hmm_solve.viterbi(hmm_eq.log_emissions(x, E), pi, A)
    for b, s in enumerate(seqs):
        joint = brute_force(s, pi, A, E)
        assert tuple(path[b].tolist()) == max(joint, key=joint.get)


def test_viterbi_recovers_planted_islands():
    truth = hmm_t.true_model(0.7, CPU)
    codes, z = hmm_t.planted_sequences(6, 3000, 3, truth[2], seed=5)
    x = hmm_t.one_hot(codes)
    path = hmm_solve.viterbi(hmm_eq.log_emissions(x, truth[2]), truth[0], truth[1])
    accuracy = (path == z).double().mean().item()
    assert accuracy > 0.93
    recall = (path[z == 1] == 1).double().mean().item()
    assert recall > 0.85


def test_baum_welch_never_decreases_the_likelihood_and_learns_emissions():
    truth = hmm_t.true_model(0.7, CPU)
    codes, _ = hmm_t.planted_sequences(4, 2000, 2, truth[2], seed=2)
    models, log_lik, _ = hmm_solve.baum_welch(hmm_t.one_hot(codes), *hmm_t.initial_guess(CPU), 10)
    assert all(b >= a - 1e-6 for a, b in zip(log_lik, log_lik[1:]))
    E = models[-1][2]
    assert torch.allclose(E.sum(1), torch.ones(2, dtype=torch.float64))
    assert (E[1, 1:3].sum() - truth[2][1, 1:3].sum()).abs() < 0.06  # island GC content
    assert (E[0, 1:3].sum() - truth[2][0, 1:3].sum()).abs() < 0.03  # background GC content


# --- phylogenetic trees ---


def test_jukes_cantor_matrix_is_a_markov_semigroup():
    P = phy_eq.jc_transition
    assert torch.allclose(P(0.4, CPU).sum(1), torch.ones(4))
    assert torch.allclose(P(0.3, CPU) @ P(0.5, CPU), P(0.8, CPU), atol=1e-6)
    assert torch.allclose(P(0.0, CPU), torch.eye(4))


def test_p_distance_einsum_counts_mismatches():
    X = phy_t.simulate(phy_t.random_tree(5, 0.5, True, 1), 300, 1, 2, CPU)[0]
    letters = X.argmax(-1)
    direct = (letters[:, None, :] != letters[None, :, :]).float().mean(-1)
    assert torch.allclose(phy_eq.p_distance(X), direct, atol=1e-6)


def test_jukes_cantor_distances_are_additive_on_the_true_tree_in_expectation():
    tree = phy_t.random_tree(6, 0.35, False, seed=4)
    M, b = phy_t.path_matrix(tree, CPU)
    d_true = phy_eq.patristic(M, b)
    X = phy_t.simulate(tree, 400, 200, seed=6, device=CPU)  # (R, n, L, 4): 200 replicate alignments
    p = phy_eq.p_distance(X)  # (R, n, n)
    d_mean = phy_eq.jukes_cantor(p).mean(0)
    off = ~torch.eye(6, dtype=torch.bool)
    rel = ((d_mean - d_true).abs() / d_true)[off]
    assert rel.max() < 0.05
    far = d_true == d_true.max()
    assert (p.mean(0)[far] < 0.9 * d_true[far]).all()  # the raw p-distance underestimates long paths


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_neighbour_joining_recovers_topology_from_additive_distances(seed):
    tree = phy_t.random_tree(12, 0.6, False, seed)  # no clock: not ultrametric
    D = phy_eq.patristic(*phy_t.path_matrix(tree, CPU)).double()
    nj = phy_solve.neighbour_joining(D)
    assert phy_solve.robinson_foulds(nj, tree) == 0
    assert torch.allclose(phy_eq.patristic(*phy_t.path_matrix(nj, CPU)).double(), D, atol=1e-5)
    rooted = phy_solve.midpoint_root(nj)
    assert phy_solve.splits(rooted) == phy_solve.splits(nj)


def test_upgma_recovers_clock_trees_and_rf_counts_differences():
    tree = phy_t.random_tree(10, 0.5, True, seed=3)
    D = phy_eq.patristic(*phy_t.path_matrix(tree, CPU))
    up = phy_solve.upgma(D)
    assert phy_solve.robinson_foulds(up, tree) == 0
    other = phy_t.random_tree(10, 0.5, True, seed=8)
    rf = phy_solve.robinson_foulds(other, tree)
    assert rf % 2 == 0 and 0 < rf <= 2 * (10 - 3)
