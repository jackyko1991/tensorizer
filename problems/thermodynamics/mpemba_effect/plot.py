"""Top left: p_i(t) in the staircase potential for three starts. Top right: distance to equilibrium.
Bottom: the slow-mode amplitude a_2 against the initial temperature, with the strong Mpemba point."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from problems._common.plotting import frame_indices, save_gif

COLORS = {"warm": "#ea580c", "T*": "#dc2626", "hot": "#2563eb"}


def _plain_log_ticks(axis, lo: float, hi: float, every: int = 1) -> None:
    """Decade ticks with plain-text labels such as 1e3 (mathtext labels are slow to redraw in every GIF frame)."""
    decades = 10.0 ** np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1, every)
    axis.set_major_locator(FixedLocator(decades))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{np.log10(v):.0f}"))


def animate(
    u: torch.Tensor, pi: torch.Tensor, times: torch.Tensor, p: torch.Tensor, kl: torch.Tensor, T0: torch.Tensor,
    show: dict, T_grid: torch.Tensor, a2_grid: torch.Tensor, T_bath: float, T_star: float, path: Path,
) -> Path:  # fmt: skip
    us, pis, t, ps, kls = u.cpu().numpy(), pi.cpu().numpy(), times.cpu().numpy(), p.cpu().numpy(), kl.cpu().numpy()
    kls = np.maximum(kls, 1e-300)  # round-off can leave D ≈ -1e-12 once p = π
    T0s, Tg, a2g = T0.cpu().numpy(), T_grid.cpu().numpy(), a2_grid.cpu().numpy()
    x = np.arange(len(us))
    t_plot = np.maximum(t, t[1])  # log axis: draw t = 0 at the first positive time

    fig = plt.figure(figsize=(7.4, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    ax_p, ax_k, ax_a = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

    # potential (right axis) and the probability on each site
    ax_u = ax_p.twinx()
    ax_u.fill_between(x, us, us.min() - 0.5, step="mid", color="#e5e7eb", lw=0)
    ax_u.set_ylim(us.min() - 0.5, us.max() + 1.5)
    ax_u.set_ylabel("U (grey)", color="#6b7280", fontsize=8)
    ax_p.set_zorder(ax_u.get_zorder() + 1)  # probability curves above the grey potential
    ax_p.patch.set_visible(False)
    ax_u.grid(False)
    ax_p.plot(x, pis, color="#111827", lw=1, ls="--", label="π (Boltzmann, T_b)")
    lines = {k: ax_p.plot(x, ps[0, b], color=COLORS[k], lw=1.6, label=f"T₀ = {T0s[b]:.2f}")[0] for k, b in show.items()}
    ax_p.set_ylim(0, 1.15 * max(ps[:, list(show.values())].max(), pis.max()))
    ax_p.set_xlabel("site i")
    ax_p.set_ylabel("p_i(t)")
    ax_p.legend(fontsize=7, loc="upper right")
    title = ax_p.set_title(" ", fontsize=9)

    # D(p‖π) for every start, highlighted ones on top
    cmap = plt.get_cmap("Greys")
    for b in range(len(T0s)):
        if b not in show.values():
            ax_k.loglog(t_plot, kls[:, b], color=cmap(0.3 + 0.5 * b / len(T0s)), lw=0.8)
    for k, b in show.items():
        ax_k.loglog(t_plot, kls[:, b], color=COLORS[k], lw=1.8)
    y_lo = max(kls[-1].min(), 1e-12)
    ax_k.set_xlim(t_plot[0], t_plot[-1])
    ax_k.set_ylim(y_lo, kls.max() * 2)
    _plain_log_ticks(ax_k.xaxis, t_plot[0], t_plot[-1])
    _plain_log_ticks(ax_k.yaxis, y_lo, kls.max() * 2, every=3)
    ax_k.set_xlabel("time t")
    ax_k.set_ylabel("D(p ‖ π)")
    ax_k.set_title("distance to equilibrium (grey: other T₀)", fontsize=9)
    cursor = ax_k.axvline(t_plot[0], color="#f59e0b", lw=1.2)

    # a_2(T): a zero crossing above T_b is the strong Mpemba effect
    ax_a.semilogx(Tg, a2g, color="#374151", lw=1.6)
    ax_a.axhline(0, color="#9ca3af", lw=1)
    ax_a.axvline(T_bath, color="#9ca3af", ls="--", lw=1)
    ax_a.text(T_bath, a2g.max(), "T_b ", color="#6b7280", fontsize=8, va="top", ha="right")
    ax_a.axvline(T_star, color=COLORS["T*"], ls="--", lw=1)
    ax_a.text(T_star, a2g.max(), f" T* = {T_star:.2f}  (a₂ = 0)", color=COLORS["T*"], fontsize=8, va="top")
    a2_at = np.interp(np.log(T0s), np.log(Tg), a2g)
    for k, b in show.items():
        ax_a.plot([T0s[b]], [a2_at[b]], "o", color=COLORS[k], ms=6)
    ax_a.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100]))
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.set_xlabel("initial temperature T₀")
    ax_a.set_ylabel("a₂(T₀)")

    def update(step: int):
        for k, b in show.items():
            lines[k].set_ydata(ps[step, b])
        cursor.set_xdata([t_plot[step]])
        title.set_text(f"t = {t[step]:.3g}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 72), path, fps=12)
