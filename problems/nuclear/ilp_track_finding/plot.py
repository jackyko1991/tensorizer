"""Candidate segments fade in, then the chosen tracks appear as PDLP converges; solver comparison panels below.

Top: the detector seen along the beam (x–y) and from the side (r–z), and the PDLP residuals against iteration.
Bottom: wall time against problem size for every solver (log–log, fitted exponents), the gap / efficiency / fake
rate of each solver at this size, and the TrackML score against time per event next to the cited state of the art.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from PIL import Image

from problems._common.progress import report

from . import equation, solve, tensors

GREY, INK = "#9ca3af", "#111827"
SOLVERS = {  # one colour per solver, used in every panel
    "HiGHS MILP": "#111827",
    "HiGHS LP": "#6b7280",
    "Gurobi MILP": "#dc2626",
    "tensor PDLP (CPU)": "#2563eb",
    "tensor PDLP (CUDA)": "#06b6d4",
    "Hungarian (ours; PyTorch)": "#f59e0b",
    "scipy linear_sum_assignment": "#a16207",
}
SHORT = {
    "HiGHS MILP": "HiGHS MILP",
    "HiGHS LP": "HiGHS LP",
    "Gurobi MILP": "Gurobi",
    "tensor PDLP (CPU)": "PDLP CPU",
    "tensor PDLP (CUDA)": "PDLP GPU",
    "Hungarian (ours; PyTorch)": "Hungarian",
    "scipy linear_sum_assignment": "scipy LSA",
}
SOTA_COLOUR = "#16a34a"
TRACK_COLOURS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#4f46e5",
    "#b45309",
    "#0d9488",
    "#be123c",
]
N_FADE, N_PDLP, N_HOLD = 14, 44, 16
RC = {  # readable at dpi 80: titles 11 pt, axis labels and ticks 10 pt, legends and notes 9 pt
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
}


def scaling_fits(bench: list[dict]) -> dict:
    """Fitted t ≈ C Nᵏ per solver, N = hits in the wedge: on N ≥ 1000 when there are ≥ 3 such sizes, else all."""
    fits = {}
    for name in SOLVERS:
        rows = [r for r in bench if r["solver"] == name and isinstance(r["wall_s"], float)]
        large = [r for r in rows if r["n_hits"] >= 1000]  # fixed overheads dominate the small instances
        rows = large if len(large) >= 3 else rows
        if len(rows) >= 3:
            n = torch.tensor([r["n_hits"] for r in rows], dtype=torch.float64)
            t = torch.tensor([r["wall_s"] for r in rows], dtype=torch.float64)
            fits[name] = solve.fit_scaling(n, t)
    return fits


def _save_shared_palette(fig, update, frames, path: Path, fps: int = 10) -> Path:
    """Render every frame, quantise all of them with ONE palette (colours never drift), write the GIF."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)


def _track_lines(xyz, seg, chosen, label):
    """x–y and r–z line segments of the chosen segments, with one colour per track."""
    s = seg[chosen]
    p, q = xyz[s[:, 0]], xyz[s[:, 1]]
    xy = np.stack([p[:, :2], q[:, :2]], 1)
    rz = np.stack(
        [np.stack([p[:, 2], np.hypot(p[:, 0], p[:, 1])], -1), np.stack([q[:, 2], np.hypot(q[:, 0], q[:, 1])], -1)], 1
    )
    colours = [TRACK_COLOURS[int(k) % len(TRACK_COLOURS)] for k in label[s[:, 0]]]
    return xy, rz, colours


def _animate(part, seg_all, seg, trip, score, lp, label_h, live, bench, cuts, dphi, path: Path) -> Path:
    xyz = part["xyz"].cpu().numpy()
    seg_c, trip_c, score_c = seg.cpu(), trip.cpu(), score.cpu()
    n = xyz.shape[0]
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    hist = {k: v.cpu().numpy() for k, v in lp["history"].items()}
    snaps = lp["snaps"]
    pick = sorted(set(np.linspace(0, len(snaps) - 1, min(N_PDLP, len(snaps))).round().astype(int).tolist()))
    states = []
    for i in pick:
        x, y = solve.round_and_repair(snaps[i][1].cpu(), seg_c, trip_c, score_c, n)
        lab = equation.tracks_from_segments(seg_c[y.bool()], n)
        states.append((snaps[i][0], *_track_lines(xyz, seg_c.numpy(), y.bool().numpy(), lab.numpy()), int(x.sum())))

    fig = plt.figure(figsize=(14, 8.6))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.1, 1], hspace=0.42, wspace=0.34, left=0.06, right=0.98, top=0.95, bottom=0.1
    )
    ax_xy, ax_rz, ax_res = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    ax_sc = fig.add_subplot(grid[1, 0])
    sub = grid[1, 1].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.08)
    ax_gap, ax_eff = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ax_ml = fig.add_subplot(grid[1, 2])

    # detector views (fixed limits from all hits)
    pad = 8.0
    ax_xy.set_xlim(xyz[:, 0].min() - pad, xyz[:, 0].max() + pad)
    ax_xy.set_ylim(xyz[:, 1].min() - pad, xyz[:, 1].max() + pad)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("$x$ (mm)")
    ax_xy.set_ylabel("$y$ (mm)")
    ax_rz.set_xlim(xyz[:, 2].min() - 20, xyz[:, 2].max() + 20)
    ax_rz.set_ylim(0, r.max() + 15)
    ax_rz.set_xlabel("$z$ (mm)")
    ax_rz.set_ylabel("$r$ (mm)")
    noise = part["particle"].cpu().numpy() == 0
    for ax, (u, v) in ((ax_xy, (xyz[:, 0], xyz[:, 1])), (ax_rz, (xyz[:, 2], r))):
        ax.scatter(u[~noise], v[~noise], s=4, color=INK, zorder=3, lw=0)
        ax.scatter(u[noise], v[noise], s=5, color=GREY, marker="x", zorder=3, lw=0.6)
    sa = seg_all.cpu().numpy()
    if len(sa) > 20000:  # draw a fixed random sample of the candidates on very wide wedges
        sa = sa[np.random.default_rng(0).choice(len(sa), 20000, replace=False)]
    cand_xy = LineCollection(np.stack([xyz[sa[:, 0], :2], xyz[sa[:, 1], :2]], 1), colors=GREY, lw=0.3, alpha=0.0)
    cand_rz = LineCollection(
        np.stack([np.stack([xyz[sa[:, 0], 2], r[sa[:, 0]]], -1), np.stack([xyz[sa[:, 1], 2], r[sa[:, 1]]], -1)], 1),
        colors=GREY,
        lw=0.3,
        alpha=0.0,
    )
    ax_xy.add_collection(cand_xy)
    ax_rz.add_collection(cand_rz)
    trk_xy = LineCollection([], lw=1.6, zorder=4)
    trk_rz = LineCollection([], lw=1.6, zorder=4)
    ax_xy.add_collection(trk_xy)
    ax_rz.add_collection(trk_rz)
    title_xy = ax_xy.set_title("")
    ax_rz.set_title(f"side view: {len(seg_all)} segments, {trip.shape[0]} triplets")

    # residuals: axis ranges from the whole history, curves drawn progressively
    it = hist["iter"]
    floor = 1e-12
    ax_res.set_xlim(0, it.max() * 1.02)
    vals = np.concatenate([hist["primal"], hist["dual"], hist["gap"]])
    ax_res.set_yscale("log")
    ax_res.set_ylim(floor * 0.3, vals.max() * 3)
    ax_res.set_xlabel("PDLP iteration (dimensionless)")
    ax_res.set_ylabel("relative residual (dimensionless)")
    (l_p,) = ax_res.plot([], [], "o-", ms=3, color="#2563eb", lw=1.8, label="primal")
    (l_d,) = ax_res.plot([], [], "o-", ms=3, color="#dc2626", lw=1.8, label="dual")
    (l_g,) = ax_res.plot([], [], "o-", ms=3, color="#16a34a", lw=1.8, label="duality gap")
    ax_res.legend(loc="upper right")
    res_title = ax_res.set_title("")

    # scalability
    fits = scaling_fits(bench)
    sizes = [rw["n_hits"] for rw in bench]
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    for name, colour in SOLVERS.items():
        rows = [rw for rw in bench if rw["solver"] == name and isinstance(rw["wall_s"], float)]
        if not rows:
            continue
        nn = np.array([rw["n_hits"] for rw in rows])
        tt = np.array([rw["wall_s"] for rw in rows])
        lab = SHORT[name] + (f" $N^{{{fits[name][0]:.2f}}}$" if name in fits else "")
        ax_sc.plot(nn, tt, "o", ms=4, color=colour, label=lab)
        if name in fits:
            k, cc = fits[name]
            g = np.geomspace(min(nn), max(nn), 20)
            ax_sc.plot(g, cc * g**k, color=colour, lw=1.2, ls="--" if "LP" in name and "PDLP" not in name else "-")
    ax_sc.plot([live["n_hits"]], [live["seconds"]], "*", ms=13, color="#2563eb", mec=INK, mew=0.6, label="this run")
    ax_sc.plot(
        [live["n_hits"]], [live["seconds_h"]], "*", ms=12, color=SOLVERS["Hungarian (ours; PyTorch)"], mec=INK, mew=0.6
    )
    ts = [rw["wall_s"] for rw in bench if isinstance(rw["wall_s"], float)] + [live["seconds"], live["seconds_h"]]
    ax_sc.set_xlim(min(sizes) * 0.7, max(sizes) * 1.4)
    ax_sc.set_ylim(min(ts) * 0.4, max(ts) * 3e3)
    ax_sc.set_xlabel("hits in the wedge $N$ (dimensionless)")
    ax_sc.set_ylabel("wall time (s)")
    ax_sc.legend(
        loc="upper left",
        ncol=2,
        handlelength=0.8,
        columnspacing=0.5,
        handletextpad=0.3,
        borderpad=0.3,
        labelspacing=0.25,
    )
    ax_sc.set_title("wall time vs size (offline)")

    # gap / efficiency / fake per solver, at this wedge (or the nearest benchmark wedge)
    grid_dphi = sorted({rw["dphi_deg"] for rw in bench})
    near = min(grid_dphi, key=lambda d: abs(d - dphi))
    rows = {rw["solver"]: rw for rw in bench if rw["dphi_deg"] == near and rw["solver"] != "HiGHS LP"}
    names = [s for s in SOLVERS if s in rows]
    gaps = [max(100 * rows[s]["gap"], 0.0) for s in names]
    effs = [100 * rows[s]["efficiency"] for s in names]
    fakes = [100 * rows[s]["fake_rate"] for s in names]
    yb = np.arange(len(names))
    ax_gap.barh(yb, gaps, color=[SOLVERS[s] for s in names])
    ax_gap.set_yticks(yb, [SHORT[s] for s in names])
    ax_gap.invert_yaxis()
    ax_gap.set_xlim(0, max(max(gaps), 0.1) * 2.6)
    ax_gap.set_xlabel("gap to optimum (%)")
    for yy, g in zip(yb, gaps):
        ax_gap.text(g, yy, " 0 (exact)" if g < 1e-6 else f" {g:.2f}%", va="center", fontsize=9)
    w = 0.4
    ax_eff.barh(yb - w / 2, effs, height=w, color=[SOLVERS[s] for s in names])
    ax_eff.barh(yb + w / 2, fakes, height=w, color=[SOLVERS[s] for s in names], alpha=0.45, hatch="///")
    ax_eff.set_yticks(yb, [""] * len(names))
    ax_eff.invert_yaxis()
    ax_eff.set_xlim(0, 100)
    ax_eff.set_xlabel("efficiency / fake (%)")
    ax_eff.text(
        0.98,
        1.02,
        "solid: efficiency\nhatched: fake rate",
        transform=ax_eff.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax_gap.set_title(f"solvers at Δφ = {near:g}°", loc="left")

    # TrackML score vs time per event
    sota = tensors.load_table(tensors.SOTA)
    ax_ml.set_xscale("log")
    ax_ml.set_xlim(1e-3, 1e3)
    ax_ml.set_ylim(0.3, 1.05)
    ax_ml.set_xlabel("time per event (s)")
    ax_ml.set_ylabel("TrackML score (dimensionless)")
    labelled = {"Mikado": (-30, -16), "FASTrack": (2, 6), "Exa.TrkX": (6, -5)}
    for rw in sota:
        name = rw["method"].split(" (")[0]
        if isinstance(rw["time_s"], float):
            ax_ml.plot(rw["time_s"], rw["score"], "D", ms=6, color=SOTA_COLOUR)
            if name in labelled:
                ax_ml.annotate(
                    name,
                    (rw["time_s"], rw["score"]),
                    fontsize=9,
                    xytext=labelled[name],
                    textcoords="offset points",
                    color="#14532d",
                )
        else:
            ax_ml.axhline(rw["score"], color=SOTA_COLOUR, lw=0.8, ls=":")
            if name == "Top Quarks":
                ax_ml.text(1.3e-3, rw["score"] + 0.008, "Top Quarks (not timed)", fontsize=9, color="#14532d")
    full = [rw for rw in bench if rw["dphi_deg"] == max(grid_dphi) and isinstance(rw["trackml_score"], float)]
    for rw in full:
        ax_ml.plot(rw["wall_s"], rw["trackml_score"], "o", ms=6, color=SOLVERS[rw["solver"]], mec=INK, mew=0.4)
    ax_ml.plot([live["seconds"]], [live["score"]], "*", ms=13, color="#2563eb", mec=INK, mew=0.6)
    ax_ml.plot(
        [live["seconds_h"]], [live["score_h"]], "*", ms=12, color=SOLVERS["Hungarian (ours; PyTorch)"], mec=INK, mew=0.6
    )
    keys = [
        plt.Line2D([], [], ls="", marker="D", ms=6, color=SOTA_COLOUR, label="cited SOTA (full events)"),
        plt.Line2D([], [], ls="", marker="o", ms=6, color=GREY, mec=INK, label="ours, 360° subset"),
        plt.Line2D([], [], ls="", marker="*", ms=11, color=GREY, mec=INK, label="this run"),
    ]
    ax_ml.legend(handles=keys, loc="center left", bbox_to_anchor=(0.0, 0.66), frameon=False)
    ax_ml.set_title("TrackML score vs time")

    # frames: 0..N_FADE-1 fade in, then PDLP states, then hold
    total = N_FADE + len(states) + N_HOLD
    stamp = fig.text(0.5, 0.012, "", ha="center", fontsize=10, color="#374151")

    def draw_residuals(iteration):
        upto = it <= iteration
        upto[: min(2, len(upto))] = True  # the first checks are visible from frame 0
        for line, key in ((l_p, "primal"), (l_d, "dual"), (l_g, "gap")):
            line.set_data(it[upto], np.maximum(hist[key][upto], floor))

    def update(f):
        if f < N_FADE:
            alpha = 0.2 * (f + 1) / N_FADE
            cand_xy.set_alpha(alpha)
            cand_rz.set_alpha(alpha)
            title_xy.set_text(f"hits and candidates, Δφ = {dphi:g}°")
            draw_residuals(it[0])
            res_title.set_text(f"PDLP residuals, iteration {int(it[min(1, len(it) - 1)])}")
            return
        k = min(f - N_FADE, len(states) - 1)
        iteration, xy, rz, colours, n_trip = states[k]
        cand_xy.set_alpha(0.06)
        cand_rz.set_alpha(0.06)
        trk_xy.set_segments(xy)
        trk_xy.set_color(colours)
        trk_rz.set_segments(rz)
        trk_rz.set_color(colours)
        draw_residuals(iteration)
        title_xy.set_text(f"tracks at iteration {iteration} ({n_trip} triplets)")
        res_title.set_text(f"PDLP residuals, iteration {iteration}")
        if f >= N_FADE + len(states):
            gap = (
                f"gap to HiGHS {100 * max(live['opt'] - live['obj'], 0) / live['opt']:.2f}%"
                if live["opt"]
                else (f"≤ {100 * max(live['bound'] - live['obj'], 0) / live['bound']:.2f}% from optimum (dual bound)")
            )
            stamp.set_text(
                f"PDLP + repair: {gap}, efficiency {100 * live['eff']:.1f}%, fake {100 * live['fake']:.1f}%, "
                f"score {live['score']:.3f}   |   Hungarian: efficiency {100 * live['eff_h']:.1f}%, "
                f"fake {100 * live['fake_h']:.1f}%, score {live['score_h']:.3f}"
            )

    frames = list(range(total))
    return _save_shared_palette(fig, update, frames, path)


def animate(*args, **kwargs) -> Path:
    """Render the GIF with readable font sizes (see RC)."""
    with plt.rc_context(RC):
        return _animate(*args, **kwargs)
