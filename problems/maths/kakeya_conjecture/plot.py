"""Perron trees refining, the three-tree Kakeya set with a needle in every direction, and δ-tube scaling in ℝⁿ."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import solve, tensors

N_FRAMES = 40


def animate(trees: list, kakeya: list, tubes: dict, segments3: tuple, path: Path) -> Path:
    h = tensors.HALF_WIDTH
    k_max = trees[-1]["k"]
    polys = [solve.tree_polygons(t).cpu().numpy() for t in trees]
    kpolys = [solve.kakeya_polygons(t).cpu().numpy() for t in trees]
    areas = np.array([t["area"] / h for t in trees])
    alphas = np.array([t["alpha"] for t in trees])
    ks = np.arange(k_max + 1)
    centres = [solve.tree_centre(t).cpu().numpy() for t in trees]

    fig = plt.figure(figsize=(12, 7.2))
    ax_t = fig.add_subplot(2, 3, 1)
    ax_k = fig.add_subplot(2, 3, 2)
    ax_a = fig.add_subplot(2, 3, 3)
    ax_3 = fig.add_subplot(2, 3, 4, projection="3d")
    ax_d = fig.add_subplot(2, 3, 5)
    ax_v = fig.add_subplot(2, 3, 6)

    tree_coll = PolyCollection(polys[0], facecolor="#60a5fa", edgecolor="#1e40af", lw=0.3)
    ax_t.add_collection(tree_coll)
    ax_t.plot([-h, 0, h, -h], [0, 1, 0, 0], color="#9ca3af", lw=1, ls="--")
    (needle_t,) = ax_t.plot([], [], color=ACCENT[1], lw=2.2)
    x_lo = min(p[..., 0].min() for p in polys) - 0.05
    x_hi = max(p[..., 0].max() for p in polys) + 0.05
    ax_t.set_xlim(x_lo, x_hi)
    ax_t.set_ylim(-0.05, 1.08)
    ax_t.set_aspect("equal")
    title_t = ax_t.set_title("", fontsize=9)

    kak_coll = PolyCollection(kpolys[0], facecolor="#60a5fa", edgecolor="#1e40af", lw=0.3)
    ax_k.add_collection(kak_coll)
    (needle_k,) = ax_k.plot([], [], color=ACCENT[1], lw=2.2)
    lim = max(np.abs(p).max() for p in kpolys) + 0.05
    ax_k.set_xlim(-lim, lim)
    ax_k.set_ylim(-lim, lim)
    ax_k.set_aspect("equal")
    title_k = ax_k.set_title("", fontsize=9)

    ax_a.plot(ks[1:], areas[1:], "o-", color=ACCENT[0], ms=4, label="Perron tree area / triangle")
    bound = np.minimum(alphas ** (2 * ks) + 2 * (1 - alphas), 1)
    ax_a.plot(ks[1:], bound[1:], "s--", color="#9ca3af", ms=3, label=r"bound $\alpha^{2k} + 2(1-\alpha)$")
    c = (areas[1:] * ks[1:]).min()
    ax_a.plot(ks[1:], c / ks[1:], ":", color=ACCENT[1], label=r"$c/k = c/\log_2 N$ (Córdoba shape)")
    (dot_a,) = ax_a.plot([], [], "o", color=ACCENT[1], ms=9, mfc="none", mew=2)
    ax_a.set_xlabel("refinement $k$ ($N = 2^k$ triangles)")
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(fontsize=7)
    ax_a.set_title("area → 0, but only logarithmically", fontsize=9)

    start, end = (s.cpu().numpy() for s in segments3)
    ax_3.add_collection3d(Line3DCollection(np.stack([start, end], 1), colors=ACCENT[0], lw=0.7))
    for setter in (ax_3.set_xlim, ax_3.set_ylim, ax_3.set_zlim):
        setter(-1, 1)
    ax_3.set_xticklabels([])
    ax_3.set_yticklabels([])
    ax_3.set_zticklabels([])
    ax_3.set_title(rf"$\mathbb{{R}}^3$: one unit tube per direction ({len(start)} directions)", fontsize=9)

    styles = {2: ACCENT[0], 3: ACCENT[2], 4: ACCENT[3]}
    for (n, arr), (deltas, counts, slope) in tubes.items():
        inv = 1 / np.array(deltas)
        ls = "-" if arr == "bush" else "--"
        ax_d.loglog(inv, counts, ls, marker="o", ms=3, color=styles[n], label=f"n={n} {arr}: slope {slope:.2f}")
        vol = np.array(counts) * np.array(deltas) ** n
        ax_v.semilogx(inv, vol, ls, marker="o", ms=3, color=styles[n], label=f"n={n} {arr}")
    ax_d.set_xlabel(r"$1/\delta$")
    ax_d.set_ylabel(r"$N(\delta)$ cubes of side $\delta$")
    ax_d.legend(fontsize=6.5, loc="upper left")
    ax_d.set_title(r"Minkowski dimension: slope of $\log N$ vs $\log 1/\delta$", fontsize=9)

    kd = np.array([1 / d for d, _ in kakeya])
    kv = np.array([cells * d**2 for d, cells in kakeya])
    ax_v.semilogx(kd, kv, "-", marker="D", ms=4, color=ACCENT[1], lw=2, label="n=2 three Perron trees")
    ax_v.set_yscale("log")
    ax_v.set_xlabel(r"$1/\delta$")
    ax_v.set_ylabel(r"volume $\approx N(\delta)\,\delta^n$")
    ax_v.set_ylim(0.3, 300)
    ax_v.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax_v.set_title("tube unions level off; the Perron set keeps shrinking", fontsize=9)

    hold = N_FRAMES // (k_max + 1)

    def needle(theta: float, centre: np.ndarray) -> tuple:
        d = 0.5 * np.array([math.cos(theta), math.sin(theta)])
        return [centre[0] - d[0], centre[0] + d[0]], [centre[1] - d[1], centre[1] + d[1]]

    def update(f: int):
        k = min(k_max, f // hold)
        tree_coll.set_verts(polys[k])
        kak_coll.set_verts(kpolys[k])
        edge = "#1e40af" if k <= 5 else "#60a5fa"  # thousands of dark edges would hide the slivers
        tree_coll.set_edgecolor(edge)
        kak_coll.set_edgecolor(edge)
        n_tri = 2**k
        # needle in the tree: direction (b, −1) inside the cone, along a segment of triangle i
        b = -h + 2 * h * ((f * 7) % N_FRAMES + 0.5) / N_FRAMES
        i = min(n_tri - 1, int((b + h) / (2 * h / n_tri)))
        s = trees[k]["shift"][i].item()
        apex, foot = np.array([s, 1.0]), np.array([b + s, 0.0])
        unit = (foot - apex) / np.linalg.norm(foot - apex)
        needle_t.set_data([apex[0], apex[0] + unit[0]], [apex[1], apex[1] + unit[1]])
        title_t.set_text(f"Perron tree: $2^{{{k}}}$ triangles, area {areas[k]:.1%} of the triangle")
        # needle in the Kakeya set: sweep all 180°, placed on the copy and triangle that hold that direction
        theta = math.pi * f / N_FRAMES
        phi = (theta + math.pi / 2) % math.pi  # copy c holds line angles c·60° − 90° + atan(b), |atan b| ≤ 30°
        copy = round(phi / (math.pi / 3)) % 3
        local = (phi - copy * math.pi / 3 + math.pi / 2) % math.pi - math.pi / 2
        b = math.tan(local)
        i = min(n_tri - 1, max(0, int((b + h) / (2 * h / n_tri))))
        s = trees[k]["shift"][i].item()
        rot = tensors.rotation(copy * math.pi / 3).numpy()
        a0 = rot @ (np.array([s, 1.0]) - centres[k])
        f0 = rot @ (np.array([b + s, 0.0]) - centres[k])
        unit = (f0 - a0) / np.linalg.norm(f0 - a0)
        needle_k.set_data([a0[0], a0[0] + unit[0]], [a0[1], a0[1] + unit[1]])
        title_k.set_text(
            f"three trees: a unit needle at {math.degrees(theta):3.0f}°, area ≤ {3 * trees[k]['area']:.3f}"
        )
        dot_a.set_data([k], [areas[k]] if k > 0 else [1.0])
        ax_3.view_init(elev=20, azim=30 + 300 * f / N_FRAMES)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=6)
