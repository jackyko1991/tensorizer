"""Main loop: real TrackML hits → candidate segments and triplets → ILP → tensor PDLP + repair, Hungarian baseline."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def near_ties(x: torch.Tensor, trip: torch.Tensor, seg: torch.Tensor, score: torch.Tensor, tol: float = 0.05) -> int:
    """Chosen triplets that have an unchosen rival sharing 2 of their 3 hits with a score within `tol`: places where
    another solver may legitimately pick a different triplet at (almost) the same objective."""
    hits = torch.stack([seg[trip[:, 0], 0], seg[trip[:, 0], 1], seg[trip[:, 1], 1]], 1)  # (T, 3)
    chosen = x.bool().nonzero().squeeze(1)
    if chosen.numel() == 0:
        return 0
    shared = (hits[chosen][:, None, :, None] == hits[None, :, None, :]).any(-1).sum(-1)  # (C, T) hits in common
    rival = (shared == 2) & ~x.bool()[None, :] & ((score[chosen][:, None] - score[None, :]).abs() < tol)
    return int(rival.any(1).sum())


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    cuts = tensors.Cuts(pt_min=params["pt_min"])
    event = tensors.load_event(device=dev)
    part = tensors.wedge(event, params["dphi"])
    n = part["xyz"].shape[0]
    n_part = int(torch.unique(part["particle"][part["particle"] > 0]).numel())
    print(
        f"TrackML event 1000, pixel barrel, |η| < 1, φ wedge {params['dphi']:g}°: {n} hits from {n_part} particles "
        f"+ {int((part['particle'] == 0).sum())} noise hits (device {dev.type})"
    )

    # 1. candidates and the ILP
    t0 = time.perf_counter()
    seg_all = equation.segments(part["xyz"], part["layer"], cuts)
    seg, trip, score = equation.candidates(part["xyz"], part["layer"], cuts)
    a, b, c = equation.ilp(seg, trip, score, n, cuts.lam)
    t_build = time.perf_counter() - t0
    print(
        f"candidates (p_T ≥ {cuts.pt_min:g} GeV, |z₀| ≤ {cuts.z0_max:g} mm): {seg_all.shape[0]} segments → "
        f"{trip.shape[0]} triplets with s_t > 0 on {seg.shape[0]} segments; ILP {a.shape[1]} variables × "
        f"{a.shape[0]} constraints ({a._nnz()} non-zeros), built in {t_build:.2f} s"
    )

    # 2. tensor PDLP on the LP relaxation, then rounding + greedy repair
    lp = solve.pdlp(a, b, c, max_iter=params["max_iter"], tol=10.0 ** -params["digits"], check=params["check"])
    t1 = time.perf_counter()
    x, y = solve.round_and_repair(lp["z"], seg, trip, score, n)
    t_repair = time.perf_counter() - t1
    obj = equation.objective(x, y, score, cuts.lam)
    feasible = bool(((a @ torch.cat([x, y]).double()) <= b + 1e-9).all())
    h = lp["history"]
    print(
        f"PDLP: {lp['iterations']} iterations in {lp['seconds']:.2f} s, primal {h['primal'][-1]:.1e}, "
        f"dual {h['dual'][-1]:.1e}, gap {h['gap'][-1]:.1e}; LP objective {lp['objective']:.3f} "
        f"(dual bound {lp['dual_bound']:.3f})"
    )
    frac = ((lp["z"] > 1e-3) & (lp["z"] < 1 - 1e-3)).double().mean().item()
    print(
        f"rounding + repair ({t_repair * 1e3:.0f} ms): {int(x.sum())} triplets, {int(y.sum())} segments, integer "
        f"objective {obj:.3f}, all constraints hold: {feasible}; {frac:.1%} of LP values fractional"
    )
    print(f"  certified: within {max(lp['dual_bound'] - obj, 0) / lp['dual_bound']:.2%} of the optimum (dual bound)")

    # 3. physics quality against the truth
    label = equation.tracks_from_segments(seg[y.bool()], n)
    eff, fake = equation.efficiency_fake(label, part["particle"], part["layer"], part["pt"], cuts.pt_min)
    score_ml = equation.trackml_score(label, part["particle"], part["weight"])
    reco = equation.reconstructable(part["particle"], part["layer"], part["pt"], cuts.pt_min)
    oracle = torch.where(reco[part["particle"]], part["particle"], -1)  # perfect tracks of reconstructable particles
    score_max = equation.trackml_score(oracle, part["particle"], part["weight"])
    print(
        f"tensor PDLP tracks: efficiency {eff:.1%} of {int(reco.sum())} reconstructable particles, fake rate "
        f"{fake:.1%}, TrackML score {score_ml:.3f} (perfect tracking of the reconstructable particles: {score_max:.3f})"
    )

    # 4. Hungarian baseline: exact layer-to-layer assignments, chained
    t2 = time.perf_counter()
    y_h = solve.hungarian_tracks(part["xyz"], part["layer"], seg, cuts)
    t_h = time.perf_counter() - t2
    x_h = solve.triplets_of(y_h, trip)
    obj_h = equation.objective(x_h, y_h.long(), score, cuts.lam)
    label_h = equation.tracks_from_segments(seg[y_h], n)
    eff_h, fake_h = equation.efficiency_fake(label_h, part["particle"], part["layer"], part["pt"], cuts.pt_min)
    score_h = equation.trackml_score(label_h, part["particle"], part["weight"])
    diff = int((x.bool() ^ x_h.bool()).sum())
    print(
        f"Hungarian chains ({t_h:.2f} s): objective {obj_h:.3f}, efficiency {eff_h:.1%}, fake rate {fake_h:.1%}, "
        f"TrackML score {score_h:.3f}; {diff} triplets differ from PDLP"
    )
    print(f"near-ties: {near_ties(x, trip, seg, score)} chosen triplets have a rival sharing 2 hits with |Δs| < 0.05")

    # 5. the offline benchmark (HiGHS, Gurobi, cuOpt attempt, PDLP on GPU, scipy assignment)
    bench = tensors.load_table(tensors.BENCHMARK)
    same = [r for r in bench if r["dphi_deg"] == params["dphi"]] if abs(cuts.pt_min - 0.5) < 1e-9 else []
    highs = [r for r in same if r["solver"] == "HiGHS MILP"]
    opt = highs[0]["objective"] if highs else None
    if opt is not None:
        print(
            f"exact MILP optimum (HiGHS, offline): {opt:.3f} → PDLP gap {max(opt - obj, 0) / opt:.2%}, "
            f"Hungarian gap {max(opt - obj_h, 0) / opt:.2%}"
        )
        for r in same:
            if r["solver"] != "HiGHS MILP" and r["n_diff_vs_highs"] != "":
                print(f"  {r['solver']}: {int(r['n_diff_vs_highs'])} selected triplets differ from HiGHS")
    else:
        print("no offline HiGHS optimum for these settings (benchmark: p_T,min = 0.5 GeV, Δφ ∈ 5…360°)")
    fits = plot.scaling_fits(bench)
    for name, (k, _) in fits.items():
        print(f"scaling: {name}: wall time ∝ N^{k:.2f}")
    live = dict(
        obj=obj,
        eff=eff,
        fake=fake,
        score=score_ml,
        seconds=lp["seconds"] + t_repair,
        n_hits=n,
        opt=opt,
        bound=lp["dual_bound"],
        obj_h=obj_h,
        eff_h=eff_h,
        fake_h=fake_h,
        score_h=score_h,
        seconds_h=t_h,
    )

    return plot.animate(
        part,
        seg_all,
        seg,
        trip,
        score,
        lp,
        label_h,
        live,
        bench,
        cuts,
        params["dphi"],
        out_dir / "ilp_track_finding.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
