from collections import Counter

import pytest
import torch

from problems.bioinformatics.kmer_gc_skew import equation as km_eq
from problems.bioinformatics.kmer_gc_skew import solve as km_solve
from problems.bioinformatics.kmer_gc_skew import tensors as km_t
from problems.bioinformatics.pwm_motif_scan import equation as pwm_eq
from problems.bioinformatics.pwm_motif_scan import solve as pwm_solve
from problems.bioinformatics.pwm_motif_scan import tensors as pwm_t
from problems.bioinformatics.wright_fisher_drift import equation as wf_eq
from problems.bioinformatics.wright_fisher_drift import solve as wf_solve
from problems.bioinformatics.wright_fisher_drift import tensors as wf_t

CPU = torch.device("cpu")

# --- k-mers and GC skew ---


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_kmer_counts_match_python_counter(k):
    seq, _, _ = km_t.synthetic_genome(3000, 0.3, 0.5, 0.05, seed=1, device=CPU)
    X = km_t.one_hot(seq)
    counts = km_solve.count_kmers(X, km_t.place_values(k, CPU))
    assert counts.shape == (4**k,)
    assert counts.sum().item() == len(seq) - k + 1

    text = "".join(km_t.BASES[a] for a in seq.tolist())
    expected = Counter(text[n : n + k] for n in range(len(text) - k + 1))
    names = km_t.kmer_names(k)
    assert {names[m]: c for m, c in enumerate(counts.tolist()) if c} == dict(expected)


def test_kmer_index_reads_words_in_base_four():
    seq = torch.tensor([0, 1, 2, 3, 3])  # A C G T T
    m = km_eq.kmer_index(km_eq.unfold(km_t.one_hot(seq), 3), km_t.place_values(3, CPU))
    names = km_t.kmer_names(3)
    assert [names[i] for i in m.tolist()] == ["ACG", "CGT", "GTT"]


def test_gc_content_and_skew_of_known_window():
    X = km_t.one_hot(torch.tensor([2, 2, 2, 1, 0, 3, 0, 3]))  # GGGC ATAT
    is_gc, g_minus_c = km_t.gc_strand_weights(CPU)
    n = km_eq.window_base_counts(X, 4, 4)
    assert torch.allclose(km_eq.gc_content(n, is_gc), torch.tensor([1.0, 0.0]))
    assert torch.allclose(km_eq.gc_skew(n, is_gc, g_minus_c), torch.tensor([0.5, 0.0]))


@pytest.mark.parametrize("origin", [0.2, 0.55])
def test_cumulative_skew_minimum_at_planted_origin(origin):
    L = 200_000
    seq, ori, ter = km_t.synthetic_genome(L, origin, 0.5, 0.05, seed=3, device=CPU)
    _, found_ori, found_ter = km_solve.find_origin(km_t.one_hot(seq), km_t.gc_strand_weights(CPU)[1])
    assert km_solve.circular_distance(found_ori, ori, L) < 0.01 * L
    assert km_solve.circular_distance(found_ter, ter, L) < 0.01 * L


def test_window_skew_sign_follows_strand():
    L = 100_000
    seq, ori, _ = km_t.synthetic_genome(L, 0.0, 0.5, 0.1, seed=4, device=CPU)
    is_gc, g_minus_c = km_t.gc_strand_weights(CPU)
    win = km_solve.windows(km_t.one_hot(seq), 2000, is_gc, g_minus_c)
    leading = win["start"] < L // 2 - 2000
    lagging = (win["start"] > L // 2) & (win["start"] < L - 2000)
    assert win["skew"][leading].mean() > 0.05
    assert win["skew"][lagging].mean() < -0.05


# --- position weight matrix ---


def _pwm(n_train=40, strength=0.9, seed=0):
    gen = torch.Generator().manual_seed(seed)
    M = pwm_t.true_motif(strength, CPU)
    sites = pwm_t.sample_sites(M, n_train, gen)
    return M, pwm_solve.build_pwm(sites, 0.5, pwm_t.background(CPU)), gen


def test_pwm_columns_sum_to_one_and_ic_in_range():
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
    q = pwm_t.background(CPU)
    assert torch.allclose(pwm_eq.information_content(q[None].clone(), q), torch.zeros(1), atol=1e-7)
    near_pure = pwm_eq.frequencies(torch.tensor([[1e6, 0.0, 0.0, 0.0]]), 1e-3)
    assert pwm_eq.information_content(near_pure, q).item() == pytest.approx(2.0, abs=1e-3)


def test_reverse_complement_matrix_scores_the_other_strand():
    _, pwm, gen = _pwm()
    seq = torch.randint(0, 4, (200,), generator=gen)
    scores = pwm_solve.scan_both_strands(seq, pwm["W"])  # (2, n)
    rc = pwm_t.reverse_complement(seq)
    scores_rc = pwm_solve.scan_both_strands(rc, pwm["W"])
    # the minus-strand score at n equals the plus-strand score of the reverse complement at the mirrored position
    assert torch.allclose(scores[1], scores_rc[0].flip(0), atol=1e-5)


def test_scan_finds_planted_sites_on_both_strands():
    M, pwm, gen = _pwm(n_train=30, seed=0)
    seq, starts, strands = pwm_t.sequence_with_sites(M, 5000, 8, gen)
    scores = pwm_solve.scan_both_strands(seq, pwm["W"])
    null = pwm_solve.scan_both_strands(pwm_t.shuffles(seq, 20, gen), pwm["W"])
    p = pwm_eq.p_values(scores, null)
    hits = pwm_solve.call_hits(p, 1e-4)
    found = pwm_solve.match(hits, starts, strands)
    assert found.float().mean() >= 0.75
    assert found[strands > 0].any() and found[strands < 0].any()
    # few false positives: about p · 2(L - m + 1) ≈ 1 expected
    others = [h for h in hits if not any(abs(h[0] - int(s)) < M.shape[0] for s in starts)]
    assert len(others) <= 4


def test_p_values_are_uniform_ish_on_null():
    _, pwm, gen = _pwm()
    seq = torch.randint(0, 4, (4000,), generator=gen)
    null = pwm_solve.scan_both_strands(pwm_t.shuffles(seq, 10, gen), pwm["W"])
    fresh = pwm_solve.scan_both_strands(torch.randint(0, 4, (4000,), generator=gen), pwm["W"])
    p = pwm_eq.p_values(fresh, null)
    assert (p <= 0.1).float().mean().item() == pytest.approx(0.1, abs=0.03)


# --- Hardy–Weinberg and Wright–Fisher ---


def test_hardy_weinberg_sums_to_one_and_sampling_matches():
    p = torch.linspace(0, 1, 11, dtype=torch.float64)
    G = wf_eq.hardy_weinberg(p)
    assert torch.allclose(G.sum(dim=-1), torch.ones(11, dtype=torch.float64))
    gen = torch.Generator().manual_seed(0)
    obs = wf_solve.sample_genotypes(torch.full((200,), 0.3, dtype=torch.float64), 500, gen).mean(dim=0)
    assert torch.allclose(obs, wf_eq.hardy_weinberg(torch.tensor(0.3, dtype=torch.float64)), atol=0.01)


def test_neutral_fixation_fraction_is_one_over_2n():
    N, R = 25, 40_000
    gen = torch.Generator().manual_seed(1)
    s = torch.zeros(1, dtype=torch.float64)
    u = wf_solve.fixation(wf_t.new_mutants(1, R, N, CPU), N, s, gen, max_gen=40 * N)
    se = (1 / (2 * N) * (1 - 1 / (2 * N)) / R) ** 0.5
    assert abs(u.item() - 1 / (2 * N)) < 4 * se


def test_fixation_with_selection_matches_kimura():
    N, R = 25, 20_000
    gen = torch.Generator().manual_seed(2)
    s = torch.tensor([4.0, 8.0], dtype=torch.float64) / (4 * N)
    u = wf_solve.fixation(wf_t.new_mutants(2, R, N, CPU), N, s, gen, max_gen=40 * N)
    u_k = wf_eq.kimura_fixation(1 / (2 * N), N, s)
    assert torch.allclose(u, u_k, rtol=0.15)
    assert wf_eq.kimura_fixation(0.3, N, torch.zeros(1)).item() == pytest.approx(0.3)


def test_heterozygosity_decays_as_one_minus_one_over_2n_to_the_t():
    N, R, G = 40, 20_000, 120
    gen = torch.Generator().manual_seed(3)
    p = wf_solve.simulate(wf_t.initial_frequencies(R, 0.5, CPU), N, 0.0, G, gen)
    assert p.shape == (R, G + 1)
    H = wf_eq.heterozygosity(p).mean(dim=0)
    t = torch.arange(G + 1, dtype=torch.float64)
    H_th = wf_eq.expected_heterozygosity(0.5, N, t)
    assert torch.allclose(H, H_th, rtol=0.05)
    # the mean allele frequency does not drift without selection
    assert p[:, -1].mean().item() == pytest.approx(0.5, abs=0.01)
