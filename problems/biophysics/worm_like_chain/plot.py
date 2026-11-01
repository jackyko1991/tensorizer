"""Animate the force–extension curve filling in as the Monte Carlo runs, next to chains pulled by four forces."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import KT, fjc_extension, marko_siggia_extension

COLORS = ["#2563eb", "#16a34a", "#ea580c", "#dc2626"]


def animate(
    out: dict,
    F: torch.Tensor,
    P: float,
    a: float,
    show: list[int],
    fit_range: tuple[float, float],
    P_corr: float,
    path: Path,
) -> Path:
    forces = F.cpu().numpy()
    z_chain = out["z"].mean(-1).cpu().numpy()  # (S+1, B) extension averaged over chains, after every sweep
    rec = out["record_sweeps"].numpy()
    # running average over the second half of the sweeps done so far (the first half counts as equilibration)
    cum = np.concatenate([np.zeros((1, z_chain.shape[1])), np.cumsum(z_chain, axis=0)])
    start = rec // 2
    z_run = (cum[rec + 1] - cum[start]) / (rec + 1 - start)[:, None]  # (K, B)
    chains = (a * torch.cumsum(out["snapshots"], dim=2)).cpu().numpy()  # (K, 4, N, 3) bead positions in nm
    in_fit = (forces >= fit_range[0]) & (forces <= fit_range[1])

    F_line = torch.logspace(-2.2, 1.4, 300, dtype=torch.float64)
    z_ms = marko_siggia_extension(F_line, P).numpy()
    z_fjc = fjc_extension(F_line, 2 * P).numpy()

    fig, (ax_f, ax_c) = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- force against extension ---
    ax_f.set_yscale("log")
    ax_f.minorticks_off()
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(F_line[0].item(), F_line[-1].item())
    ax_f.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax_f.axhspan(*fit_range, color="#fef3c7", zorder=0, label="fit range")
    ax_f.plot(z_ms, F_line.numpy(), color="#2563eb", lw=1.8, label=f"worm-like chain, P = {P:g} nm")
    ax_f.plot(z_fjc, F_line.numpy(), "--", color="#6b7280", lw=1.3, label=f"freely jointed, b = 2P = {2 * P:g} nm")
    (dots,) = ax_f.plot([], [], "o", color="#ea580c", ms=5, mec="#9a3412", label="Monte Carlo, discrete WLC")
    ax_f.set_xlabel("relative extension  z / L")
    ax_f.set_ylabel("force F (pN)")
    ax_f.legend(loc="upper left", fontsize=7.5)
    readout = ax_f.text(0.03, 0.62, "", transform=ax_f.transAxes, fontsize=8.5, va="top")
    title = ax_f.set_title("force–extension after 0 sweeps", fontsize=10)  # real text so tight_layout leaves room

    # --- chains pulled along z, side view ---
    gap = 0.45 * chains.shape[2] * a  # horizontal spacing of the four chains
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-0.6 * gap, 3.6 * gap)
    ax_c.set_ylim(-0.25 * gap, chains.shape[2] * a * 1.15)
    ax_c.set_xticks([])
    ax_c.set_ylabel("z (nm), along the force")
    ax_c.grid(False)
    lines, arrows = [], []
    arrow_len = 0.12 * chains.shape[2] * a
    for c, b in enumerate(show):
        arrows.append(  # the pulling force, drawn at the free end of the chain
            ax_c.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": "#dc2626", "lw": 1.5})
        )
        ax_c.text(c * gap, -0.18 * gap, f"{forces[b]:.2g} pN", ha="center", fontsize=8, color=COLORS[c])
        lines.append(ax_c.plot([], [], lw=1.2, color=COLORS[c])[0])
    ax_c.axhline(0, color="#374151", lw=2)
    ax_c.set_title("one chain at each of four forces", fontsize=10)

    def update(k: int):
        dots.set_data(z_run[k, 1:], forces[1:])
        P_b = KT * (z_run[k, 1:] + 0.25 / (1 - z_run[k, 1:]) ** 2 - 0.25) / forces[1:]
        P_in = P_b[in_fit[1:]]
        fit = f"{np.exp(np.log(P_in).mean()):.0f} nm" if (P_in > 0).all() else "…"
        readout.set_text(f"fit: P = {fit}\ntangent correlation: P = {P_corr:.0f} nm")
        title.set_text(f"force–extension after {rec[k]} sweeps")
        for c in range(len(show)):
            r = chains[k, c] - chains[k, c, :1]  # anchor the first bead at the surface
            lines[c].set_data(r[:, 0] + c * gap, r[:, 2])
            end = (r[-1, 0] + c * gap, r[-1, 2])
            arrows[c].xy = (end[0], end[1] + arrow_len)
            arrows[c].set_position(end)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(rec), 60), path, fps=8)
