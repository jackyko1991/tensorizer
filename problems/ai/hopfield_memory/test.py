"""Mathematical checks, written first (TDD): what any correct Hopfield network code must satisfy.

Run with `pytest problems/ai/hopfield_memory/test.py`. Each test is a theorem or a known result, not an
implementation detail: asynchronous updates never raise the classic energy; below capacity the stored patterns
are fixed points; the one-step bit error rate is Φ(−1/√α); recall collapses past α ≈ 0.138 while the modern
network retrieves 2N patterns in one step for large β; its energy never rises; and β → 0 gives the mean pattern.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_classic_energy_never_increases_under_single_neuron_updates():
    x = tensors.random_patterns(30, 100, 0, CPU)  # 0.3 N patterns: far past capacity, many spurious minima
    w = equation.hebbian(x, torch.ones(1, 30))
    s = tensors.random_patterns(16, 100, 1, CPU)[None]  # (1, 16, 100) random starts
    energies = [equation.energy(s, w)]
    gen = torch.Generator().manual_seed(2)
    for i in torch.randint(0, 100, (600,), generator=gen):
        s = equation.async_sweep(s, w, i[None])  # one neuron at a time
        energies.append(equation.energy(s, w))
    e = torch.stack(energies)  # (601, 1, 16)
    assert (e.diff(dim=0) <= 1e-5).all()  # ΔE = −Δs_i h_i ≤ 0 for symmetric W with zero diagonal


def test_stored_patterns_are_fixed_points_below_capacity():
    n = 500
    x = tensors.random_patterns(25, n, 3, CPU)  # α = 0.05, well below 0.138
    w = equation.hebbian(x, torch.ones(1, 25))[0]
    assert torch.equal(torch.sign(x @ w), x)  # every bit of every stored pattern is stable


def test_one_step_bit_errors_follow_the_gaussian_crosstalk_law():
    alphas = torch.tensor([0.1, 0.2, 0.3])
    _, _, err = solve.capacity_curve(alphas, 400, 0.1, 8, 0, 1.0, 0)
    theory = torch.tensor([0.5 * math.erfc(1 / math.sqrt(2 * a)) for a in alphas.tolist()])  # Φ(−1/√α)
    assert torch.allclose(err, theory, rtol=0.35)  # crosstalk of P − 1 random patterns ≈ Gaussian, variance α


def test_classic_recall_collapses_past_capacity_but_modern_does_not():
    alphas = torch.tensor([0.05, 0.1, 0.25, 0.4])
    classic, modern, _ = solve.capacity_curve(alphas, 300, 0.1, 8, 4, 1.0, 5)
    assert classic[0] > 0.98 and classic[1] > 0.95  # below α_c ≈ 0.138: near-perfect retrieval
    assert classic[2] < 0.8 and classic[3] < 0.7  # above it the retrieval states disappear
    assert (modern > 0.999).all()  # the modern network does not care


def test_modern_update_retrieves_2n_patterns_in_one_step():
    n = 200
    x = tensors.random_patterns(2 * n, n, 6, CPU)  # twice as many patterns as neurons
    q = tensors.flip_bits(x[:40], 0.2, 7)  # 20 % of the bits wrong
    out = equation.modern_update(q, x, beta=1.0)
    assert (equation.overlap(out, x[:40]) > 0.999).all()


def test_modern_energy_never_increases():
    x = tensors.faces(12, 24, 8, CPU).reshape(12, -1)
    xi = tensors.random_patterns(10, x.shape[1], 9, CPU) * 0.5
    for beta in (0.005, 0.05, 1.0):
        e, state = [equation.modern_energy(xi, x, beta)], xi
        for _ in range(5):
            state = equation.modern_update(state, x, beta)
            e.append(equation.modern_energy(state, x, beta))
        assert (torch.stack(e).diff(dim=0) <= 1e-3).all()  # the update is a concave-convex procedure (CCCP)


def test_small_beta_gives_the_average_pattern():
    x = tensors.faces(8, 24, 10, CPU).reshape(8, -1)
    q = tensors.flip_bits(x[:3], 0.1, 11)
    out = equation.modern_update(q, x, beta=1e-6)
    assert torch.allclose(out, x.mean(0).expand_as(out), atol=1e-3)  # softmax → uniform: a metastable average


def test_masked_faces_are_retrieved_by_the_modern_network_only():
    faces = tensors.faces(20, 32, 12, CPU)
    q = tensors.corrupt(faces, 0.4, 0.05, 13).reshape(20, -1)  # bottom 40 % blanked, 5 % noise
    x = faces.reshape(20, -1)
    used = torch.ones(1, 20)
    modern = solve.modern_recall(x, used, q, 0.1)[0]
    assert (equation.overlap(modern, x) > 0.99).all()
    classic = solve.classic_recall(x, used, q, 4, 14)[0]
    assert equation.overlap(classic, x).mean() < 0.9  # 20 correlated faces ≪ 0.138 N, yet the classic net mixes them
