"""Animate the transfers (orbits, transfer ellipses, spacecraft, burns) next to total Δv against r2/r1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from problems._common.plotting import ACCENT, frame_indices, save_gif

GREY = "#9ca3af"


def ellipse(x_plus: float, x_minus: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Orbit with the central body at a focus, crossing the +x axis at x_plus and the -x axis at -x_minus."""
    theta = np.linspace(0, 2 * np.pi, n)
    a, centre, b = (x_plus + x_minus) / 2, (x_plus - x_minus) / 2, np.sqrt(x_plus * x_minus)
    return centre + a * np.cos(theta), b * np.sin(theta)


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    burn_t: torch.Tensor,
    burn_dv: torch.Tensor,
    burn_pos: torch.Tensor,
    kinds: list[str],
    radii: tuple[float, float, float],
    body: str,
    unit: str,
    t_label: str,
    R: torch.Tensor,
    curves: dict[str, torch.Tensor],
    crossovers: tuple[float, float],
    rb_ratio: float,
    path: Path,
) -> Path:
    t = times.cpu().numpy()  # (T,)
    pos = traj[:, :, 0, :].cpu().numpy()  # (T, B, 2)
    bt, bdv, bpos = burn_t.cpu().numpy(), burn_dv.cpu().numpy(), burn_pos.cpu().numpy()  # (B, 3), (B, 3), (B, 3, 2)
    r1, r2, rb = radii
    B = pos.shape[1]
    colours = [ACCENT[0] if k == "hohmann" else ACCENT[2] for k in kinds]

    fig, (ax, ax_dv) = plt.subplots(1, 2, figsize=(8.6, 4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.set_aspect("equal", adjustable="datalim")
    unit = r"$R_\oplus$" if unit == "R⊕" else unit  # mathtext: the ⊕ glyph is missing from the default font
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")

    # start and target circles, and the transfer ellipses (dashed)
    for r, label in [(r1, "$r_1$"), (r2, "$r_2$")]:
        ax.plot(*ellipse(r, r), color="#374151", lw=1.2)
        ax.annotate(label, (r * 0.71, -r * 0.71), xytext=(4, -4), textcoords="offset points", fontsize=10, va="top")
    for k, kind in enumerate(kinds):
        legs = [(r1, r2)] if kind == "hohmann" else [(r1, rb), (r2, rb)]
        for x_plus, x_minus in legs:
            ax.plot(*ellipse(x_plus, x_minus), "--", color=colours[k], lw=0.9, alpha=0.6)
    if body == "earth":
        ax.add_patch(Circle((0, 0), 1.0, color="#93c5fd", ec="#1e40af", lw=1, zorder=4))  # Earth to scale
    else:
        ax.plot([0], [0], "*", ms=14, color="#f59e0b", zorder=4)  # the Sun (much smaller than these orbits)

    trails = [ax.plot([], [], lw=1.6, color=colours[b])[0] for b in range(B)]
    dots = [
        ax.plot([], [], "o", ms=6, color=colours[b], label=f"{kinds[b]}: Σ|Δv| = {np.abs(bdv[b]).sum():.2f} km/s")[0]
        for b in range(B)
    ]
    burns = []  # (time, marker) pairs: a red cross appears where and when a burn fires
    for b in range(B):
        for k in range(bt.shape[1]):
            if bdv[b, k] != 0:
                (mark,) = ax.plot([bpos[b, k, 0]], [bpos[b, k, 1]], "X", ms=8, color="#dc2626", mec="white", zorder=6)
                mark.set_visible(False)
                burns.append((bt[b, k], mark))
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    clock = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("× = impulsive burn (prograde or retrograde)", fontsize=9)

    # total cost against the radius ratio, in units of the starting circular speed
    Rn = R.cpu().numpy()
    ax_dv.semilogx(Rn, curves["hohmann"].cpu().numpy(), color=ACCENT[0], lw=1.8, label="Hohmann")
    ax_dv.semilogx(
        Rn,
        curves["bi-elliptic"].cpu().numpy(),
        color=ACCENT[2],
        lw=1.5,
        label=f"bi-elliptic, $r_b = {rb_ratio:g}\\,r_2$",
    )
    ax_dv.semilogx(
        Rn, curves["bi-parabolic"].cpu().numpy(), ":", color=ACCENT[3], lw=1.5, label=r"bi-elliptic, $r_b\to\infty$"
    )
    for x, text in zip(crossovers, ["11.94", "15.58"]):
        ax_dv.axvline(x, color=GREY, ls="--", lw=1)
        ax_dv.text(
            x * 1.04,
            0.98,
            text,
            fontsize=8,
            color="#4b5563",
            rotation=90,
            va="top",
            transform=ax_dv.get_xaxis_transform(),
        )
    ratio = r2 / r1
    if 1 <= ratio <= 100:
        k = int(np.argmin(np.abs(Rn - ratio)))
        for name in ["hohmann", "bi-elliptic"]:
            ax_dv.plot([ratio], [float(curves[name][k])], "o", ms=6, mfc="white", mec="#111827", zorder=5)
    ax_dv.set_xlim(1, 100)
    ax_dv.xaxis.set_major_locator(
        FixedLocator([1, 2, 5, 10, 20, 50, 100])
    )  # plain labels, no minor ticks: faster frames
    ax_dv.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax_dv.xaxis.set_minor_locator(NullLocator())
    top = max(float(c.max()) for c in curves.values())
    ax_dv.set_ylim(0, min(1.12 * top, 0.9))
    ax_dv.set_xlabel("$r_2 / r_1$")
    ax_dv.set_ylabel(r"total $\Delta v$ / $v_{c1}$")
    ax_dv.legend(fontsize=7, loc="lower right")
    ax_dv.set_title("cost of the transfer (○ = this run)", fontsize=9)

    def update(step: int):
        for b in range(B):
            trails[b].set_data(pos[: step + 1, b, 0], pos[: step + 1, b, 1])
            dots[b].set_data([pos[step, b, 0]], [pos[step, b, 1]])
        for when, mark in burns:
            mark.set_visible(t[step] >= when)
        clock.set_text(f"t = {t[step]:.1f} {t_label}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
