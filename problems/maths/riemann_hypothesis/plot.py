"""Six panels that grow with the height T: Z(t) near T, S(t), spacings, pair correlation, ψ(x) and the strip."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

DVP = 5.558691  # zero-free region σ > 1 − 1/(5.558691 ln t), proven (Mossinghoff–Trudgian 2015)


def _pair_density(x: np.ndarray, edges: np.ndarray, neighbours: int = 10) -> np.ndarray:
    d = np.concatenate([x[j:] - x[:-j] for j in range(1, neighbours + 1)])
    counts, _ = np.histogram(d, bins=edges)
    return counts / (max(len(x) - neighbours, 1) * (edges[1] - edges[0]))


def animate(t, z, gammas, gram, bad, s, t_s, spacing, x, edges, xs, exact, psi_k, heights, path: Path) -> Path:
    t, z, g = t.cpu().numpy(), z.cpu().numpy(), gammas.cpu().numpy()
    gram, bad = gram.cpu().numpy(), bad.cpu().numpy()
    s, t_s, spacing, x = s.cpu().numpy(), t_s.cpu().numpy(), spacing.cpu().numpy(), x.cpu().numpy()
    edges, xs, exact, psi_k = edges.cpu().numpy(), xs.cpu().numpy(), exact.cpu().numpy(), psi_k.cpu().numpy()
    mid = 0.5 * (edges[1:] + edges[:-1])
    good = np.ones(len(gram), bool)
    good[bad] = False

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    (ax_z, ax_s, ax_h), (ax_r, ax_p, ax_c) = axes

    (z_line,) = ax_z.plot([], [], color="#374151", lw=1)
    ax_z.axhline(0, color="#9ca3af", lw=0.8)
    zeros_dot = ax_z.scatter([], [], s=18, color="#2563eb", zorder=3, label="zeros")
    gram_ok = ax_z.scatter([], [], s=22, marker="|", color="#16a34a", label="Gram point, law holds")
    gram_bad = ax_z.scatter([], [], s=40, marker="x", color="#dc2626", label="Gram's law fails")
    ax_z.set_ylim(-6, 6)
    ax_z.set_xlabel("$t$")
    ax_z.legend(fontsize=7, loc="lower left", ncol=1)
    title_z = ax_z.set_title("", fontsize=9)

    (s_line,) = ax_s.plot([], [], color=ACCENT[0], lw=0.6)
    for level in (-1, 1):
        ax_s.axhline(level, color="#9ca3af", ls="--", lw=0.8)
    ax_s.set_xscale("log")
    ax_s.set_xlim(10, t[-1])
    ax_s.set_ylim(-2.2, 2.2)
    ax_s.set_xlabel("$t$")
    ax_s.set_title(r"$S(t) = N(t) - \theta(t)/\pi - 1$ (Riemann–von Mangoldt)", fontsize=9)

    grid = np.linspace(0, 3, 300)
    bars_h = ax_h.stairs(np.zeros_like(mid), edges, fill=True, color="#93c5fd", label="zeta zeros")
    ax_h.plot(grid, 32 / np.pi**2 * grid**2 * np.exp(-4 * grid**2 / np.pi), color=ACCENT[1], lw=1.5, label="GUE")
    ax_h.plot(grid, np.exp(-grid), color="#6b7280", lw=1.2, ls="--", label="Poisson")
    ax_h.set_xlim(0, 3)
    ax_h.set_ylim(0, 1.15)
    ax_h.set_xlabel("unfolded spacing $s$")
    ax_h.legend(fontsize=7, loc="upper right")
    title_h = ax_h.set_title("", fontsize=9)

    bars_r = ax_r.stairs(np.zeros_like(mid), edges, fill=True, color="#93c5fd", label="zeta zeros")
    ax_r.plot(grid, 1 - np.sinc(grid) ** 2, color=ACCENT[1], lw=1.5, label=r"$1 - (\sin \pi u / \pi u)^2$")
    ax_r.axhline(1, color="#6b7280", lw=1.2, ls="--", label="no correlation")
    ax_r.set_xlim(0, 3)
    ax_r.set_ylim(0, 1.4)
    ax_r.set_xlabel("difference $u$ of unfolded zeros")
    ax_r.legend(fontsize=7, loc="lower right")
    ax_r.set_title("pair correlation (Montgomery's conjecture)", fontsize=9)

    ax_p.step(xs, exact - xs, where="post", color="#374151", lw=1.2, label=r"$\psi(x) - x$ (sieve)")
    (p_line,) = ax_p.plot(xs, psi_k[0] - xs, color=ACCENT[0], lw=1.1, label="explicit formula")
    ax_p.set_ylim((exact - xs).min() - 1.5, (exact - xs).max() + 1.5)
    ax_p.set_xlabel("$x$")
    ax_p.legend(fontsize=7, loc="upper left")
    title_p = ax_p.set_title("", fontsize=9)

    t_top = t[-1]
    tt = np.geomspace(10, t_top, 400)
    ax_c.fill_betweenx(tt, 1 - 1 / (DVP * np.log(tt)), 1.05, color="#bbf7d0", label="zero-free (proven)")
    ax_c.fill_betweenx(tt, -0.05, 1 / (DVP * np.log(tt)), color="#bbf7d0")
    ax_c.axvspan(7 / 8, 1, color="#fed7aa", alpha=0.6, lw=0, label=r"claimed 2026, unverified: $\sigma > 7/8$")
    ax_c.axvspan(0, 1 / 8, color="#fed7aa", alpha=0.6, lw=0)
    ax_c.axvline(0.5, color="#9ca3af", ls="--", lw=1)
    zeros_c = ax_c.scatter([], [], s=3, color="#2563eb", label="zeros found on the line")
    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_yscale("log")
    ax_c.set_ylim(10, t_top)
    ax_c.set_xlabel(r"$\sigma = \mathrm{Re}\, s$")
    ax_c.set_ylabel(r"$t = \mathrm{Im}\, s$")
    ax_c.legend(fontsize=6.5, loc="upper center", framealpha=0.95)
    ax_c.set_title("critical strip: RH says every zero has σ = 1/2", fontsize=9)

    def update(f: int):
        top = heights[f]
        n = int(np.searchsorted(g, top))
        window = (t > top - 8) & (t <= top)
        z_line.set_data(t[window], z[window])
        ax_z.set_xlim(top - 8, top)
        in_w = (g > top - 8) & (g <= top)
        zeros_dot.set_offsets(np.c_[g[in_w], np.zeros(in_w.sum())])
        gw = (gram > top - 8) & (gram <= top)
        gram_ok.set_offsets(np.c_[gram[gw & good], np.zeros((gw & good).sum())])
        gram_bad.set_offsets(np.c_[gram[gw & ~good], np.zeros((gw & ~good).sum())])
        title_z.set_text(f"Z(t) up to T = {top:.0f}: {n} zeros")
        keep = t_s <= top
        s_line.set_data(t_s[keep], s[keep])
        if n > 12:
            bars_h.set_data(np.histogram(spacing[: n - 1], bins=edges, density=True)[0])
            bars_r.set_data(_pair_density(x[:n], edges))
        title_h.set_text(f"nearest-neighbour spacings ({max(n - 1, 0)} gaps)")
        p_line.set_ydata(psi_k[n] - xs)
        title_p.set_text(f"ψ(x) − x rebuilt from the {n} zeros below T")
        zeros_c.set_offsets(np.c_[np.full(n, 0.5), g[:n]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(heights), len(heights)), path, fps=6)
