"""Animate the epidemic spreading on the contact network beside its S, I, R curves; below, the threshold and
the vaccination comparison."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from problems._common.plotting import ACCENT, frame_indices, save_gif

COLOURS = np.array(["#93c5fd", "#dc2626", "#9ca3af"])  # S, I, R


def animate(A, pos, sim, mf, lam, lam_nb, ratio, ratios, sweep_sim, sweep_mf, fractions, vacc, vacc_beta, dt, path):
    states = sim["states"].cpu().numpy()  # (T, N)
    counts = sim["counts"].cpu().numpy()  # (T, B, 3)
    n = states.shape[1]
    alive = np.nonzero(counts[:, 0, 1] > 0)[0]
    t_end = min(len(states) - 1, int(alive[-1]) + 15 if len(alive) else 10)
    t = np.arange(len(states)) * dt
    xy = pos.cpu().numpy()
    i, j = np.nonzero(np.triu(A.cpu().numpy(), 1))
    mf = mf.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    (ax_net, ax_cur), (ax_thr, ax_vac) = axes

    segs = np.stack([xy[i], xy[j]], axis=1)  # (E, 2, 2)
    ax_net.add_collection(LineCollection(segs, colors="#e5e7eb", linewidths=0.5, zorder=0))
    deg = A.sum(1).cpu().numpy()
    dots = ax_net.scatter(xy[:, 0], xy[:, 1], s=12 + 1.2 * deg, c=COLOURS[states[0]], edgecolors="#374151",
                          linewidths=0.4, zorder=1)  # fmt: skip
    ax_net.set_xlim(-1.1, 1.1)
    ax_net.set_ylim(-1.1, 1.1)
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    for k, lab in enumerate(["susceptible", "infected", "recovered"]):
        ax_net.scatter([], [], c=COLOURS[k], edgecolors="#374151", linewidths=0.4, label=lab)
    ax_net.legend(fontsize=7, loc="lower left", ncols=3, handletextpad=0.2, columnspacing=0.8)
    title = ax_net.set_title("")

    mean = counts.mean(1) / n  # (T, 3)
    lines = []
    for k, lab in enumerate("SIR"):
        c = ["#2563eb", "#dc2626", "#6b7280"][k]
        ax_cur.plot(t[: t_end + 1], mf[: t_end + 1, k] / n, "--", color=c, lw=1)
        ax_cur.plot(t[: t_end + 1], mean[: t_end + 1, k], color=c, lw=1, alpha=0.4)
        lines.append(ax_cur.plot([], [], color=c, lw=2, label=f"{lab} (drawn run)")[0])
    ax_cur.plot([], [], "--", color="#374151", lw=1, label="mean-field ODE")
    ax_cur.plot([], [], color="#374151", lw=1, alpha=0.4, label=f"mean of {counts.shape[1]} runs")
    ax_cur.set_xlim(0, t[t_end])
    ax_cur.set_ylim(0, 1.02)
    ax_cur.set_xlabel(r"time (units of $1/\gamma$)")
    ax_cur.set_ylabel("fraction of people")
    ax_cur.legend(fontsize=6.5, loc="center right")
    ax_cur.set_title(rf"$\beta/\gamma$ = {ratio:.3f} = {ratio * lam:.1f} / $\lambda_{{max}}$", fontsize=10)
    cursor = ax_cur.axvline(0, color="#f59e0b", lw=1)

    x = ratios.cpu().numpy() * lam
    ax_thr.plot(x, sweep_sim.cpu().numpy(), "o-", color=ACCENT[0], ms=3.5, label="stochastic SIR (mean)")
    ax_thr.plot(x, sweep_mf.cpu().numpy(), "--", color=ACCENT[3], lw=1.5, label="mean-field ODE")
    ax_thr.axvline(1, color=ACCENT[1], lw=1.2, label=r"$\beta/\gamma = 1/\lambda_{max}$")
    ax_thr.axvline(lam / (lam_nb - 1), color=ACCENT[2], lw=1.2, ls=":",
                   label=r"message passing $1/(\lambda_{NB} - 1)$")  # fmt: skip
    ax_thr.set_xlabel(r"$\beta/\gamma \cdot \lambda_{max}(A)$")
    ax_thr.set_ylabel("fraction ever infected")
    ax_thr.set_title("epidemic threshold (workplace network)", fontsize=10)
    ax_thr.legend(fontsize=6.5, loc="upper left")

    f = fractions.cpu().numpy()
    for (name, res), c in zip(vacc.items(), (ACCENT[1], ACCENT[0])):
        ax_vac.plot(f, res["targeted"][0].cpu().numpy(), "o-", color=c, ms=3.5, label=f"{name}: hubs first")
        ax_vac.plot(f, res["random"][0].cpu().numpy(), "s--", color=c, ms=3, mfc="none", label=f"{name}: random")
        herd = res["targeted"][1].cpu().numpy() < 1  # mean-field herd immunity: β/γ λ_max(after) < 1
        if herd.any():
            ax_vac.axvline(f[np.argmax(herd)], color=c, lw=0.8, ls=":")
    ax_vac.set_xlabel("fraction vaccinated")
    ax_vac.set_ylabel("fraction infected")
    ax_vac.set_title(rf"vaccination, $\beta/\gamma$ = {vacc_beta:.2f} (dotted: mean-field herd immunity)", fontsize=9)
    ax_vac.legend(fontsize=6.5, loc="upper right")
    ax_vac.set_ylim(bottom=0)

    def update(k: int):
        dots.set_facecolors(COLOURS[states[k]])
        c = counts[k, 0]
        title.set_text(f"t = {t[k]:.1f}:  S {c[0]}  I {c[1]}  R {c[2]}  (N = {n})")
        for m, line in enumerate(lines):
            line.set_data(t[: k + 1], counts[: k + 1, 0, m] / n)
        cursor.set_xdata([t[k], t[k]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(t_end + 1, 90), path, fps=10)
