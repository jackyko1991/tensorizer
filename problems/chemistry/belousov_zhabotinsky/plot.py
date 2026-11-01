"""Animate the oscillation: time series, phase portrait with nullclines, Hopf bifurcation diagram and spiral wave."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import jacobian, rhs, steady_state, trace, x_nullcline

BLUE, RED, GREY, AMBER, DARK = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b", "#374151"
LOG_TICKS = [1e-3, 1e-2, 1e-1, 1]
LOG_LABELS = ["0.001", "0.01", "0.1", "1"]  # plain labels: no mathtext and no minor ticks to redraw every frame


def _log_axis(ax, which: str = "y"):
    if which == "y":
        ax.set_yscale("log")
        ax.set_yticks(LOG_TICKS, LOG_LABELS)
    else:
        ax.set_xscale("log")
        ax.set_xticks(LOG_TICKS, LOG_LABELS)
    ax.minorticks_off()


def animate(t, x, z, f, lo, hi, hopf, eps: float, q: float, spiral, path: Path) -> Path:
    """t (T,); x, z (T, B + 1) with the selected f last; f (B + 1,); lo, hi (B + 1,); spiral (S, n, n) or None."""
    ts, X, Z = t.numpy(), x.cpu().numpy(), z.cpu().numpy()
    fv = f.cpu().numpy()
    f_sel = float(fv[-1])
    lo, hi = lo.cpu().numpy(), hi.cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.08, hspace=0.38, wspace=0.28)
    ax_t, ax_p, ax_b, ax_s = axs.flat

    # --- time series of x for the chosen f and two stable neighbours ---
    others = [int(np.abs(fv[:-1] - v).argmin()) for v in (0.4, 2.8)]
    for b in others:
        ax_t.plot(ts, X[:, b], color=GREY, lw=1)
        ax_t.text(ts[-1], X[-1, b], f" f={fv[b]:.2f}", fontsize=7, color="#6b7280", va="center", ha="right")
    ax_t.plot(ts, X[:, -1], color=BLUE, lw=1.4, label=f"x ∝ [HBrO₂], f = {f_sel:g}")
    ax_t.plot(ts, Z[:, -1], color=RED, lw=1.2, label="z ∝ [Ce⁴⁺]")
    _log_axis(ax_t)
    ax_t.set_ylim(3e-4, 1.5)
    ax_t.set_xlim(0, ts[-1])
    ax_t.set_xlabel("scaled time t", labelpad=1)
    ax_t.set_title("well-mixed oscillations (stiff: x jumps ×1000)", fontsize=9)
    ax_t.legend(fontsize=7, loc="lower right", frameon=False)
    (cursor,) = ax_t.plot([0, 0], [3e-4, 1.5], color=AMBER, lw=1)

    # --- phase portrait in (log x, z) with the nullclines ---
    xg = np.logspace(np.log10(q) + 0.002, 0, 400)
    ax_p.plot(xg, x_nullcline(torch.tensor(xg), f_sel, q).numpy(), color=DARK, lw=1.2, label="ẋ = 0")
    ax_p.plot(xg, xg, color=DARK, lw=1.2, ls=(0, (4, 3)), label="ż = 0  (z = x)")
    gx, gz = np.meshgrid(np.logspace(-3.3, 0, 15), np.linspace(0.0, 0.4, 13))
    dx, dz = rhs(torch.tensor(gx), torch.tensor(gz), torch.tensor(f_sel), eps, q)
    u = (dx / torch.tensor(gx)).numpy() * np.log10(np.e) / 3.3  # direction in the plotted (log x, z) coordinates
    v = dz.numpy() / 0.4
    norm = np.hypot(u, v) + 1e-12
    ax_p.quiver(gx, gz, u / norm, v / norm, color="#d1d5db", angles="xy", scale=30, width=0.004)
    late = len(ts) // 3
    ax_p.plot(X[late:, -1], Z[late:, -1], color=BLUE, lw=1.2)
    xs = steady_state(torch.tensor([f_sel], dtype=torch.float64), q).item()
    ax_p.plot(xs, xs, "o", mfc="white", mec=DARK, ms=6)
    (dot,) = ax_p.plot([], [], "o", color=AMBER, ms=7, mec=DARK)
    _log_axis(ax_p, "x")
    ax_p.set_xlim(5e-4, 1)
    ax_p.set_ylim(0, 0.4)
    ax_p.set_xlabel("x", labelpad=1)
    ax_p.set_ylabel("z", labelpad=1)
    ax_p.set_title(f"phase portrait, f = {f_sel:g} (open dot: steady state)", fontsize=9)
    ax_p.legend(fontsize=7, loc="upper right", frameon=False)

    # --- Hopf bifurcation diagram over the batch ---
    fine = torch.linspace(0.3, 3.0, 300, dtype=torch.float64)
    xf = steady_state(fine, q)
    stable = (trace(jacobian(xf, xf, fine, eps, q)) < 0).numpy()
    fn, xn = fine.numpy(), xf.numpy()
    ax_b.plot(fn, np.where(stable, xn, np.nan), color=DARK, lw=1.5, label="steady state (stable)")
    ax_b.plot(fn, np.where(stable, np.nan, xn), color=DARK, lw=1.2, ls=(0, (4, 3)), label="steady state (unstable)")
    osc = hi[:-1] / lo[:-1] > 1.5
    ax_b.plot(fv[:-1][osc], hi[:-1][osc], "o", color=BLUE, ms=3.5, label="limit cycle: max, min of x")
    ax_b.plot(fv[:-1][osc], lo[:-1][osc], "o", color=BLUE, ms=3.5)
    for h in hopf.tolist():
        ax_b.axvline(h, color=AMBER, lw=1.2)
        ax_b.text(h, 1.3, f" Hopf\n f={h:.2f}", fontsize=7, color="#b45309", va="top")
    ax_b.axvline(f_sel, color=GREY, lw=0.8, ls=(0, (2, 2)))
    _log_axis(ax_b)
    ax_b.set_ylim(3e-4, 1.5)
    ax_b.set_xlim(0.3, 3.0)
    ax_b.set_xlabel("stoichiometric factor f", labelpad=1)
    ax_b.set_ylabel("x", labelpad=1)
    ax_b.set_title(f"bifurcation diagram, {len(fv) - 1} f values in one batch", fontsize=9)
    ax_b.legend(fontsize=6.5, loc="center right", frameon=False)

    # --- 2D spiral wave ---
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.grid(False)
    image = None
    if spiral is not None:
        S = spiral.cpu().numpy()
        cmap = plt.get_cmap("RdBu_r", 12)  # oxidised catalyst red, reduced blue, like ferroin in a dish
        image = ax_s.imshow(S[0], cmap=cmap, vmin=0, vmax=max(0.05, float(S.max())), interpolation="nearest")
        title = ax_s.set_title("", fontsize=9)
    else:
        ax_s.text(0.5, 0.5, "2D spiral switched off", ha="center", va="center", transform=ax_s.transAxes)

    def update(i: int):
        cursor.set_data([ts[i], ts[i]], [3e-4, 1.5])
        dot.set_data([X[i, -1]], [Z[i, -1]])
        if image is not None:
            s = round(i / (len(ts) - 1) * (len(S) - 1))
            image.set_data(S[s])
            title.set_text(f"spiral wave in a dish, z at t = {ts[i]:.1f}")

    return save_gif(fig, update, frame_indices(len(ts), 50), path, fps=8)
