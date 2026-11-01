"""Animate the burn: ⟨σv⟩(T), trajectories in the (ρR, T) plane over the ignition map, and T(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from problems._common.plotting import frame_indices, save_gif

from .equation import sigma_v


def animate(
    times: torch.Tensor,
    states: torch.Tensor,
    t0: torch.Tensor,
    r0: torch.Tensor,
    ign: torch.Tensor,
    t_curve: torch.Tensor,
    r_curve: torch.Tensor,
    shown: list[int],
    path: Path,
) -> Path:
    tt = times.numpy()
    T = states[:, :, 0].exp().numpy()  # (S, B)
    R = states[:, :, 1].exp().numpy()
    ign_np = ign.numpy()

    fig, (ax_s, ax_p, ax_t) = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [0.8, 1.2, 1.0]})

    # (a) DT reactivity
    Tg = torch.logspace(0, 2, 200, dtype=torch.float64)
    ax_s.loglog(Tg.numpy(), sigma_v(Tg).numpy(), color="#2563eb", lw=2)
    sv10 = sigma_v(torch.tensor(10.0, dtype=torch.float64)).item()
    ax_s.plot([10], [sv10], "o", color="#dc2626")
    ax_s.annotate(f"{sv10:.2e} m³/s\nat 10 keV", (10, sv10), (12, 2e-24), fontsize=7, color="#b91c1c")
    ax_s.set_xlabel("T (keV)")
    ax_s.set_ylabel(r"$\langle\sigma v\rangle_{DT}$ (m³/s)")
    ax_s.set_title("Bosch–Hale reactivity", fontsize=9)

    # (b) ignition map in the (ρR, T) plane
    grid = ign_np.reshape(len(t0), len(r0)).astype(float)
    r_edges = _edges(r0.numpy())
    t_edges = _edges(t0.numpy())
    ax_p.pcolormesh(r_edges, t_edges, grid, cmap=ListedColormap(["#eff6ff", "#fee2e2"]), shading="flat", zorder=0)
    ax_p.plot(r_curve.numpy(), t_curve.numpy(), color="#374151", lw=1.5, ls="--", label="heating = losses at t = 0")
    ax_p.set_xscale("log")
    ax_p.set_yscale("log")
    ax_p.set_xlim(r_edges[0], max(r_edges[-1], R.max() * 1.1))
    ax_p.set_ylim(0.45, max(t_edges[-1], min(T.max(), 200) * 1.2))
    ax_p.set_xlabel("hot-spot areal density ρR (g/cm²)")
    ax_p.set_ylabel("T (keV)")
    ax_p.text(r_edges[-1] * 0.95, t_edges[0] * 1.1, "ignites", color="#b91c1c", ha="right", fontsize=8)
    ax_p.text(r_edges[0] * 1.1, t_edges[0] * 1.1, "quenches", color="#1e40af", fontsize=8)
    ax_p.axhline(0.5, color="#9ca3af", lw=1)
    colors = ["#dc2626" if ign_np[b] else "#2563eb" for b in shown]
    paths = [ax_p.plot([], [], lw=1.4, color=c)[0] for c in colors]
    dots = [ax_p.plot([], [], "o", ms=4, color=c)[0] for c in colors]
    ax_p.legend(loc="upper left", fontsize=7)
    title_p = ax_p.set_title("t = 0 ps", fontsize=9)

    # (c) temperature histories
    curves = [ax_t.plot([], [], lw=1.4, color=c)[0] for c in colors]
    ax_t.set_xlim(0, tt[-1])
    ax_t.set_yscale("log")
    ax_t.set_ylim(0.4, max(60.0, min(T[:, shown].max(), 200) * 1.3))
    ax_t.axhline(0.5, color="#9ca3af", lw=1)
    ax_t.set_xlabel("time (ps)")
    ax_t.set_ylabel("hot-spot T (keV)")
    burn = ax_t.text(0.03, 0.95, "", transform=ax_t.transAxes, va="top", fontsize=8)
    ax_t.set_title("red: ignition and burn, blue: quench", fontsize=9)

    # Plain tick labels on the log axes: mathtext labels would be re-typeset on every frame (slow).
    for ax, axis, ticks in (
        (ax_s, ax_s.xaxis, [1, 3, 10, 30, 100]),
        (ax_s, ax_s.yaxis, [1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21]),
        (ax_p, ax_p.xaxis, [0.05, 0.1, 0.2, 0.5, 1, 2, 4]),
        (ax_p, ax_p.yaxis, [0.5, 1, 2, 5, 10, 20, 50, 100, 200]),
        (ax_t, ax_t.yaxis, [0.5, 1, 2, 5, 10, 20, 50, 100, 200]),
    ):
        lo, hi = ax.get_xlim() if axis is ax.xaxis else ax.get_ylim()
        kept = [v for v in ticks if lo <= v <= hi]
        axis.set_major_locator(FixedLocator(kept))
        axis.set_ticklabels([f"{v:g}" for v in kept])
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    phi = states[:, :, 2].numpy()

    def update(s: int):
        for b, line, dot, curve in zip(shown, paths, dots, curves):
            line.set_data(R[: s + 1, b], T[: s + 1, b])
            dot.set_data([R[s, b]], [T[s, b]])
            curve.set_data(tt[: s + 1], T[: s + 1, b])
        title_p.set_text(f"{ign_np.sum()} of {len(ign_np)} hot spots ignite   t = {tt[s]:.0f} ps")
        burned = phi[s][ign_np].mean() if ign_np.any() else 0.0
        burn.set_text(f"mean burn-up of ignited hot spots: {100 * burned:.0f} %")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(tt), 48), path)


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges for a log-spaced grid of centres."""
    lc = np.log(centres)
    mids = (lc[1:] + lc[:-1]) / 2
    return np.exp(np.concatenate([[2 * lc[0] - mids[0]], mids, [2 * lc[-1] - mids[-1]]]))
