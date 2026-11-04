"""Lumpy spheres rounding out (coloured by Gauss curvature), their curvature spread, and the dumbbell 3-sphere's
profile ψ(s) through neckpinch, surgery and rounding."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from problems._common.plotting import ACCENT, save_gif

from .equation import embed_profile, profile


def _pick(n_have: int, n_want: int) -> list[int]:
    return sorted({round(k * (n_have - 1) / max(1, n_want - 1)) for k in range(n_want)})


def _neck(psi: np.ndarray) -> float:
    """Smallest interior local minimum of the profile (nan if there is none)."""
    mid = psi[1:-1]
    is_min = (mid < psi[:-2]) & (mid < psi[2:])
    return float(mid[is_min].min()) if is_min.any() else float("nan")


def animate(x, surf, spread, gb, grid, pre, post, n_frames: int, path: Path) -> Path:
    # 3-sphere frames: pre-surgery snapshots, then post-surgery ones
    n_post = 0 if post is None else n_frames * 2 // 5
    frames3 = [("pre", i) for i in _pick(len(pre["t_save"]), n_frames - n_post)]
    if post is not None:
        frames3 += [("post", i) for i in _pick(len(post["t_save"]), n_post)]
    n_total = len(frames3)
    surf_idx = [round((f / max(1, n_total - 1)) ** 2 * (len(surf["t"]) - 1)) for f in range(n_total)]  # slow start

    shapes = []
    for s in range(len(surf["t"])):
        rho, z, _ = embed_profile(surf["u"][s], x)
        shapes.append((rho.cpu().numpy(), z.cpu().numpy()))
    k_all = surf["K"].cpu().numpy()
    t_s, spr = surf["t"].cpu().numpy(), spread.cpu().numpy()

    def prof(u):
        s, psi = profile(u, grid)
        s, psi = s[0].cpu().numpy(), psi[0].cpu().numpy()
        length = s[-1] + (s[-1] - s[-2]) / 2
        return np.r_[0, s, length], np.r_[0, psi, 0]

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1.4, 1]}
    )
    names = ["peanut", "pear", "lumpy"]
    offsets = [-2.6, 0.0, 2.6]
    norm = plt.Normalize(0, 2)
    lcs, fills = [], []
    for b, off in enumerate(offsets):
        lc = LineCollection([], cmap="coolwarm", norm=norm, lw=2.5)
        ax_a.add_collection(lc)
        lcs.append(lc)
        fills.append(None)
        ax_a.text(off, -1.75, names[b], ha="center", fontsize=8, color="#374151")
    ax_a.set_xlim(-4, 4)
    ax_a.set_ylim(-1.9, 1.6)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    fig.colorbar(lcs[0], ax=ax_a, fraction=0.03, pad=0.01).set_label("Gauss curvature $K$", fontsize=8)
    a_title = ax_a.set_title("", fontsize=9)

    for b in range(len(names)):
        ax_b.semilogy(t_s, spr[:, b], color=ACCENT[b], lw=1.5, label=names[b])
    cur_b = ax_b.axvline(0, color="#9ca3af", lw=1)
    ax_b.set_xlabel("$t$")
    ax_b.set_ylabel(r"$(K_\max - K_\min)/\bar K$", fontsize=8)
    ax_b.legend(fontsize=7)
    dev = np.abs(gb.cpu().numpy() / (4 * np.pi) - 1).max()
    ax_b.set_title(rf"$\int K\,dA = 4\pi\,(1 \pm {max(dev, 1e-16):.0e})$", fontsize=9)

    lengths = [prof(pre["u"][i])[0][-1] for i in range(len(pre["t_save"]))]
    if post is not None:
        lengths += [2 * prof(post["u"][i])[0][-1] for i in range(len(post["t_save"]))]
    ax_c.set_xlim(-0.05, max(lengths) * 1.03)
    ax_c.set_ylim(-1.0, 1.0)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("arclength $s$")
    ax_c.set_ylabel(r"$\pm\psi(s)$")
    c_title = ax_c.set_title("", fontsize=9)
    c_art = []

    t_pre, neck = pre["t"].cpu().numpy(), pre["neck"].cpu().numpy()
    ok = np.isfinite(neck)
    ax_d.plot(t_pre[ok], neck[ok] ** 2, color=ACCENT[0], lw=1.5, label=r"$\psi_\mathrm{neck}^2$")
    ax_d.set_ylim(0, None)
    ax_d.set_xlabel("$t$")
    ax_d.set_ylabel(r"neck radius$^2$", fontsize=8, color=ACCENT[0])
    if post is not None:
        ax_r = ax_d.twinx()
        t_post, ratio = post["t_save"].cpu().numpy(), post["ratio"].cpu().numpy()
        ax_r.semilogy(t_post, ratio - 1, color=ACCENT[1], lw=1.5)
        ax_r.set_ylabel(r"$K_\max/K_\min - 1$ (capped piece)", fontsize=8, color=ACCENT[1])
        ax_r.grid(False)
        ax_d.axvline(t_pre[-1], color="#9ca3af", ls="--", lw=1)
        ax_d.text(t_pre[-1], ax_d.get_ylim()[1] * 0.95, " surgery", fontsize=7, color="#6b7280", va="top")
    cur_d = ax_d.axvline(0, color="#9ca3af", lw=1)
    ax_d.set_title("neckpinch, then roundness", fontsize=9)

    def update(f: int):
        s = surf_idx[f]
        for b, off in enumerate(offsets):
            rho, z = shapes[s][0][b], shapes[s][1][b]
            xs, ys = np.r_[rho[::-1], -rho] + off, np.r_[z[::-1], z]
            kk = np.r_[k_all[s, b][::-1], k_all[s, b]]
            pts = np.c_[xs, ys].reshape(-1, 1, 2)
            lcs[b].set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
            lcs[b].set_array(0.5 * (kk[:-1] + kk[1:]))
            if fills[b] is not None:
                fills[b].remove()
            fills[b] = ax_a.fill(xs, ys, color="#93c5fd", alpha=0.35, lw=0)[0]
        a_title.set_text(f"2-spheres, normalised flow, t = {t_s[s]:.2f}")
        cur_b.set_xdata([t_s[s]])

        while c_art:
            c_art.pop().remove()
        phase, i = frames3[f]
        if phase == "pre":
            sc, psi = prof(pre["u"][i])
            t_now = pre["t_save"][i].item()
            c_art.append(ax_c.fill_between(sc, -psi, psi, color="#93c5fd", alpha=0.6, lw=0))
            c_art.extend(ax_c.plot(sc, psi, color="#2563eb", lw=1.5) + ax_c.plot(sc, -psi, color="#2563eb", lw=1.5))
            c_title.set_text(rf"dumbbell $S^3$, $t = {t_now:.4f}$, neck $\psi = {_neck(psi):.3f}$")
        else:
            sc, psi = prof(post["u"][i])
            t_now = post["t_save"][i].item()
            for xs, alpha in ((sc, 0.6), (2 * sc[-1] - sc[::-1], 0.25)):  # the piece and its mirror image
                ps = psi if alpha > 0.5 else psi[::-1]
                c_art.append(ax_c.fill_between(xs, -ps, ps, color="#93c5fd", alpha=alpha, lw=0))
                c_art.extend(ax_c.plot(xs, ps, color="#2563eb", lw=1.5, alpha=alpha + 0.3))
                c_art.extend(ax_c.plot(xs, -ps, color="#2563eb", lw=1.5, alpha=alpha + 0.3))
            c_title.set_text(rf"after surgery: two capped pieces, $t = {t_now:.3f}$")
        cur_d.set_xdata([t_now])

    fig.tight_layout()
    return save_gif(fig, update, list(range(n_total)), path, fps=8)
