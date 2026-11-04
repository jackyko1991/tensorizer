"""Animate the demon's engines, the work/information ledger, the erasure cost and the time-reversed run."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, save_gif

LANES = 10
LN2 = math.log(2.0)


def animate(sz: dict, er: dict, s: dict, path: Path) -> Path:
    eps, info, W_ext, sel = s["eps"].numpy(), s["I"].numpy(), s["W_ext"].numpy(), s["sel"]
    n = sz["n_per"]
    lanes = np.arange(sel * n, sel * n + LANES)  # the first engines of the selected error rate
    keys_a = sorted(sz["frames"])
    xa = np.stack([sz["frames"][k][0].numpy() for k in keys_a])  # (Fa, B)
    pa = np.stack([sz["frames"][k][1].numpy() for k in keys_a])
    wa = -np.stack([sz["frames"][k][2].numpy() for k in keys_a]).reshape(len(keys_a), len(eps), n).mean(2)  # (Fa, E)
    ta = np.array(keys_a) * s["dt"]
    wrong = sz["wrong"].numpy()

    keys_b = sorted(er["frames"])
    m_er = er["n_per"]
    taus = s["taus"].numpy()
    lanes_b = np.arange((len(taus) - 1) * m_er, (len(taus) - 1) * m_er + LANES)  # slowest erasure group
    xb = np.stack([er["frames"][k][0].numpy()[lanes_b] for k in keys_b])
    pb = np.stack([er["frames"][k][1].numpy()[lanes_b] for k in keys_b])
    tb = np.array(keys_b) * s["dt"]
    Fa, Fb = len(keys_a), len(keys_b)
    total = Fa + 2 * Fb

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax_l, ax_w = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_e, ax_c = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    # --- engines: one box per lane ---
    for j in range(LANES):
        ax.add_patch(plt.Rectangle((0, j - 0.38), 2, 0.76, fill=False, ec="#374151", lw=1.2))
    ax.axvline(1.0, color="#9ca3af", ls="--", lw=0.8)
    bars = [ax.plot([], [], "-", color="#111827", lw=3)[0] for _ in range(LANES)]
    dots = ax.scatter(np.zeros(LANES), np.arange(LANES), s=40, zorder=3)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.7, LANES - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("position $x/a$")
    title = ax.set_title("", fontsize=9)

    # --- ledger for the selected ε ---
    ax_l.axhline(info[sel], color=ACCENT[2], ls="--", lw=1.2, label=f"$kT\\,I$ = {info[sel]:.3f} (info gained)")
    ax_l.axhline(LN2, color="#9ca3af", ls=":", lw=1.2, label="$kT\\ln 2$ (Szilard)")
    led = ax_l.plot([], [], color=ACCENT[0], lw=2, label="$\\langle W_{ext}\\rangle$")[0]
    ax_l.set_xlim(0, ta[-1])
    ax_l.set_ylim(min(-0.05, wa[:, sel].min() - 0.05), 0.8)
    ax_l.set_xlabel("time $t$ ($a^2/D$)")
    ax_l.set_ylabel("work out / kT")
    ax_l.legend(fontsize=7, loc="lower right")
    ax_l.set_title(f"ledger, demon error ε = {eps[sel]:.2f}", fontsize=9)

    # --- W_ext vs I for every ε ---
    grid = np.linspace(0, LN2, 50)
    ax_w.plot(grid, grid, color=ACCENT[2], ls="--", lw=1.2, label="bound $W_{ext} = kT\\,I$")
    ax_w.axhline(LN2, color="#9ca3af", ls=":", lw=1)
    pts = ax_w.scatter(info, wa[0], c=ACCENT[0], s=28, zorder=3, label="engines")
    for e in range(len(eps)):
        ax_w.annotate(
            f"ε={eps[e]:.2f}", (info[e], W_ext[e].mean()), fontsize=6, xytext=(3, -9), textcoords="offset points"
        )
    ax_w.set_xlim(-0.03, 0.75)
    ax_w.set_ylim(-0.1, 0.78)
    ax_w.set_xlabel("mutual information $I$ (nats)")
    ax_w.set_ylabel("$\\langle W_{ext}\\rangle$ / kT")
    ax_w.legend(fontsize=7, loc="upper left")
    su = s["su"].numpy()
    ax_w.set_title(
        f"Sagawa–Ueda $\\langle e^{{-\\beta(W-\\Delta F)-I}}\\rangle$ = {su.min():.2f}…{su.max():.2f}", fontsize=9
    )

    # --- erasure cost ---
    tt = np.geomspace(taus.min() * 0.8, taus.max() * 1.3, 100)
    ax_e.plot(tt, LN2 + s["B_fit"] / tt, color=ACCENT[3], lw=1, label=f"fit $kT\\ln2 + B/\\tau$, B={s['B_fit']:.2f}")
    ax_e.axhline(LN2, color="#9ca3af", ls=":", lw=1.2, label="Landauer $kT\\ln 2$")
    m_er, se_er = s["mean_er"].numpy(), s["se_er"].numpy()
    ax_e.errorbar(taus, m_er, yerr=2 * se_er, fmt="o", color=ACCENT[1], ms=5, label="erasure ⟨W⟩ ± 2 se")
    ax_e.set_xscale("log")
    ax_e.set_xticks(taus, [f"{t:g}" for t in taus])
    ax_e.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_e.set_xlabel("erasure time $\\tau$ ($a^2/D$)")
    ax_e.set_ylabel("work in / kT")
    ax_e.set_ylim(0, max(m_er) + 0.4)
    ax_e.legend(fontsize=7, loc="upper right")
    ax_e.set_title("Landauer: erasing 1 bit (Bérut 2012: → kT ln 2)", fontsize=9)

    # --- Crooks ---
    cr = s["crooks"]
    wf, wr = cr["W_forward"].numpy(), -cr["W_reverse"].numpy()
    bins = np.linspace(min(wf.min(), wr.min()), 4.0, 45)
    ax_c.hist(wf, bins, density=True, alpha=0.6, color=ACCENT[0], label="$P_F(W)$ compress")
    ax_c.hist(wr, bins, density=True, alpha=0.6, color=ACCENT[4], label="$P_R(-W)$ expand")
    ax_c.axvline(cr["dF"], color="#111827", lw=1.2, label=f"$\\Delta F$ = {cr['dF']:.3f}")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("work $W$ / kT")
    ax_c.legend(fontsize=7)
    ax_c.set_title(f"Crooks: curves cross at ΔF, slope {s['slope']:.2f}", fontsize=9)
    for a_ in (ax_w, ax_e, ax_c):
        a_.set_visible(False)

    def update(f: int):
        if f < Fa:  # demon feedback
            x, p = xa[f, lanes], pa[f, lanes]
            col = np.where(wrong[lanes], "#dc2626", ACCENT[0])
            title.set_text(f"Szilard engines, demon error ε = {eps[sel]:.2f}\nblue: read correctly, red: misread"
                           f"   t = {ta[f]:.2f}")  # fmt: skip
            led.set_data(ta[: f + 1], wa[: f + 1, sel])
            pts.set_offsets(np.c_[info, wa[f]])
            ax_w.set_visible(f > Fa // 3)
        else:  # erasure, forward then backward in time
            g = f - Fa
            k = g if g < Fb else 2 * Fb - 1 - g
            x, p = xb[k], pb[k]
            col = np.full(LANES, "#6b7280")
            arrow = "forward" if g < Fb else "played BACKWARDS"
            title.set_text(f"erasing the bit, τ = {taus[-1]:.1f}: piston 2a → a ({arrow})\n"
                           f"t = {tb[k]:.2f}; reversed: σ = ln P_F/P_R = β(W−ΔF)")  # fmt: skip
            ax_w.set_visible(True)
            ax_e.set_visible(True)
            ax_c.set_visible(g >= Fb)
        dots.set_offsets(np.c_[x, np.arange(LANES)])
        dots.set_color(col)
        for j in range(LANES):
            bars[j].set_data([p[j], p[j]], [j - 0.38, j + 0.38])

    fig.tight_layout(h_pad=1.6)
    return save_gif(fig, update, list(range(total)), path, fps=12)
