"""Checks written first (TDD): what any correct track-finding ILP code must satisfy.

Run with `pytest problems/nuclear/ilp_track_finding/test.py`. Each test is a mathematical law or a property of the
detector geometry: the first-order LP solver reaches the exact LP optimum, the integer solution after rounding obeys
every constraint, the LP relaxation bounds the integer optimum from above, the Hungarian algorithm is exactly
optimal, and on an easy toy event the segment builder and the trackers miss nothing.
"""

import itertools

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_pdlp_reaches_the_highs_optimum_of_a_tiny_lp():
    # max cᵀz  s.t.  A z ≤ b, 0 ≤ z ≤ 1. Optimum from HiGHS (offline: scipy 1.18.1 linprog, method="highs"):
    # z* = (1, 0.5, 0, 1, 0.5), cᵀz* = 3.65.
    a = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.5, 1.0, 2.0, 1.5, 4.0], dtype=torch.float64)
    c = torch.tensor([1.0, 1.6, 0.4, 1.2, 1.3], dtype=torch.float64)
    out = solve.pdlp(a.to_sparse(), b, c, max_iter=20000, tol=1e-9)
    assert out["objective"] == pytest.approx(3.65, abs=1e-4)
    assert torch.allclose(out["z"], torch.tensor([1.0, 0.5, 0.0, 1.0, 0.5], dtype=torch.float64), atol=1e-4)
    assert (a @ out["z"] - b).max() < 1e-6  # primal feasible


def _toy_problem(n_particles=12, noise=0.0, seed=3):
    event = tensors.toy_event(n_particles, noise_fraction=noise, seed=seed, device=CPU)
    cuts = tensors.Cuts()
    seg, trip, score = equation.candidates(event["xyz"], event["layer"], cuts)
    return event, cuts, seg, trip, score


def test_segment_builder_finds_every_true_segment_of_a_noise_free_toy_event():
    event = tensors.toy_event(40, seed=1, device=CPU)
    seg = equation.segments(event["xyz"], event["layer"], tensors.Cuts())
    pid, layer = event["particle"], event["layer"]
    # true segments: consecutive layers of the same particle
    same = (pid[:, None] == pid[None, :]) & (layer[None, :] == layer[:, None] + 1)
    true_pairs = {tuple(p) for p in same.nonzero().tolist()}
    found = {tuple(p) for p in seg.tolist()}
    assert len(true_pairs) == 40 * (tensors.N_LAYERS - 1)
    assert true_pairs <= found  # the geometric cuts never lose a true segment above the pT threshold


def test_hit_exclusivity_holds_after_rounding_and_repair():
    event, cuts, seg, trip, score = _toy_problem(n_particles=60, noise=0.3, seed=5)
    a, b, c = equation.ilp(seg, trip, score, event["xyz"].shape[0], cuts.lam)
    lp = solve.pdlp(a, b, c, max_iter=300, tol=1e-12)  # stopped early on purpose: a fractional, imperfect LP point
    x, y = solve.round_and_repair(lp["z"], seg, trip, score, event["xyz"].shape[0])
    z = torch.cat([x, y]).double()
    assert ((a @ z) <= b + 1e-12).all()  # every constraint: one segment in and one out per hit, continuity
    label = equation.tracks_from_segments(seg[y.bool()], event["xyz"].shape[0])
    used = label[label >= 0]
    assert used.numel() == (label >= 0).sum()  # each hit carries one track label at most
    in_deg = torch.bincount(seg[y.bool(), 1], minlength=event["xyz"].shape[0])
    out_deg = torch.bincount(seg[y.bool(), 0], minlength=event["xyz"].shape[0])
    assert in_deg.max() <= 1 and out_deg.max() <= 1


@pytest.mark.slow
def test_lp_relaxation_bounds_the_integer_objective():
    event, cuts, seg, trip, score = _toy_problem(n_particles=80, noise=0.5, seed=7)
    a, b, c = equation.ilp(seg, trip, score, event["xyz"].shape[0], cuts.lam)
    lp = solve.pdlp(a, b, c, max_iter=20000, tol=1e-7)
    x, y = solve.round_and_repair(lp["z"], seg, trip, score, event["xyz"].shape[0])
    integer = equation.objective(x, y, score, cuts.lam)
    assert lp["dual_bound"] >= integer - 1e-9  # weak duality: any dual point bounds every feasible integer point
    assert lp["objective"] >= integer - 1e-4  # the LP optimum ≥ the integer optimum ≥ this integer solution


def test_efficiency_is_one_on_a_noise_free_well_separated_toy_event():
    event, cuts, seg, trip, score = _toy_problem(n_particles=12, seed=2)
    n = event["xyz"].shape[0]
    a, b, c = equation.ilp(seg, trip, score, n, cuts.lam)
    lp = solve.pdlp(a, b, c, max_iter=5000, tol=1e-6)
    x, y = solve.round_and_repair(lp["z"], seg, trip, score, n)
    label = equation.tracks_from_segments(seg[y.bool()], n)
    eff, fake = equation.efficiency_fake(label, event["particle"], event["layer"], event["pt"], cuts.pt_min)
    assert eff == 1.0 and fake == 0.0
    y_h = solve.hungarian_tracks(event["xyz"], event["layer"], seg, cuts)
    eff_h, fake_h = equation.efficiency_fake(
        equation.tracks_from_segments(seg[y_h], n), event["particle"], event["layer"], event["pt"], cuts.pt_min
    )
    assert eff_h == 1.0 and fake_h == 0.0


@pytest.mark.parametrize("n", [1, 3, 5, 7])
def test_hungarian_equals_brute_force_over_all_permutations(n):
    gen = torch.Generator().manual_seed(n)
    cost = torch.rand(n, n, generator=gen, dtype=torch.float64)
    col = solve.hungarian(cost)
    perms = torch.tensor(list(itertools.permutations(range(n))))  # (n!, n)
    best = cost[torch.arange(n), perms].sum(-1).min()  # brute force: every assignment
    assert sorted(col.tolist()) == list(range(n))  # a permutation
    assert cost[torch.arange(n), col].sum() == pytest.approx(best.item(), abs=1e-12)


def test_hungarian_handles_rectangular_costs():
    gen = torch.Generator().manual_seed(11)
    cost = torch.rand(4, 6, generator=gen, dtype=torch.float64)
    col = solve.hungarian(cost)
    best = min(sum(cost[i, p[i]].item() for i in range(4)) for p in itertools.permutations(range(6), 4))
    assert len(set(col.tolist())) == 4
    assert cost[torch.arange(4), col].sum().item() == pytest.approx(best, abs=1e-12)


def test_trackml_score_is_one_for_truth_and_near_zero_for_random_labels():
    event = tensors.load_event(device=CPU)
    part = tensors.wedge(event, 30.0)
    truth = part["particle"].clone()
    truth[truth == 0] = -1  # noise hits unassigned (they have weight 0 anyway)
    assert equation.trackml_score(truth, part["particle"], part["weight"]) == pytest.approx(1.0, abs=1e-12)
    gen = torch.Generator().manual_seed(0)
    random = torch.randint(0, 300, (truth.numel(),), generator=gen)
    assert equation.trackml_score(random, part["particle"], part["weight"]) < 0.02


def test_scaling_fit_recovers_a_known_exponent():
    n = torch.tensor([10.0, 30.0, 100.0, 300.0, 1000.0], dtype=torch.float64)
    t = 2e-6 * n**2.7 * (1 + 0.01 * torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0], dtype=torch.float64))
    k, c = solve.fit_scaling(n, t)
    assert k == pytest.approx(2.7, abs=0.02)
    assert c == pytest.approx(2e-6, rel=0.1)


def test_bundled_event_is_real_trackml_data():
    event = tensors.load_event(device=CPU)
    r = event["xyz"][:, :2].norm(dim=-1)
    for k, r0 in enumerate(tensors.LAYER_R):  # the four pixel-barrel layers of the TrackML detector
        assert (r[event["layer"] == k] - r0).abs().max() < 6.0  # mm (modules are flat and tilted)
    assert event["weight"].sum() > 0 and (event["weight"][event["particle"] == 0] == 0).all()  # noise has no weight
