"""Sweep the physical error rate: a sample syndrome history, the threshold plot and the suppression factor Λ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter

from problems._common.plotting import frame_indices, save_gif

from .equation import binomial_logical_error, detection_events


def animate(
    ds: list[int],
    ps: torch.Tensor,
    result: dict,
    lam: torch.Tensor,
    p_th: float,
    model: str,
    path: Path,
) -> Path:
    p = ps.numpy()
    pl, err = result["p_logical"].numpy(), result["stderr"].numpy()  # (D, P)
    lam_np = lam.numpy().copy()  # (D − 1, P)
    if np.isfinite(p_th):
        lam_np[:, p >= p_th] = np.nan  # per-round Λ is only meaningful below threshold
    d_max = ds[-1]

    fig, (ax_s, ax_p, ax_l) = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [0.8, 1.2, 1.0]})

    # (a) one shot of the largest code: detection events in space–time
    first = detection_events(result["samples"][(d_max, 0)][None])[0].numpy()
    image = ax_s.imshow(
        first, cmap=ListedColormap(["#eff6ff", "#dc2626"]), vmin=0, vmax=1, aspect="auto", origin="lower"
    )
    ax_s.set_xlabel("check j")
    ax_s.set_ylabel("round t")
    ax_s.grid(False)
    title_s = ax_s.set_title("d, p: detection events", fontsize=9)

    # (b) logical error probability of a d-round memory experiment
    lines, bars = [], []
    for d in ds:
        (line,) = ax_p.plot([], [], "o-", ms=4, lw=1.6, label=f"d = {d}")
        lines.append(line)
        if model == "code_capacity":
            exact = binomial_logical_error(ps, d).numpy()
            ax_p.plot(p, exact, "--", lw=1, color=line.get_color(), alpha=0.7)
        bars.append(line.get_color())
    if np.isfinite(p_th):
        ax_p.axvline(p_th, color="#f59e0b", lw=1.5, ls="--")
        ax_p.text(p_th, 1.0 / result["shots"], f"$p_{{th}}$ ≈ {p_th:.3f} ", color="#b45309", fontsize=8, ha="right")
    ax_p.set_xscale("log")
    ax_p.set_yscale("log")
    ax_p.set_xlim(p[0] * 0.9, p[-1] * 1.1)
    ax_p.set_ylim(0.5 / result["shots"], 0.7)
    ax_p.set_xlabel("physical error rate p")
    ax_p.set_ylabel("logical error $P_L$" + (" (dashed: exact)" if model == "code_capacity" else " per d rounds"))
    ax_p.legend(loc="lower right", fontsize=7)
    shade = [None]

    # (c) error suppression Λ = ε_d / ε_{d+2}
    lam_lines = [ax_l.plot([], [], "s-", ms=4, lw=1.4, label=f"{d}→{d + 2}")[0] for d in ds[:-1]]
    ax_l.axhline(1.0, color="#9ca3af", lw=1.5)
    ax_l.set_xscale("log")
    ax_l.set_xlim(p[0] * 0.9, p[-1] * 1.1)
    finite = lam_np[np.isfinite(lam_np)]
    ax_l.set_ylim(0, max(2.5, float(finite.max()) * 1.1) if finite.size else 3)
    ax_l.set_xlabel("physical error rate p")
    ax_l.set_ylabel(r"$\Lambda = \varepsilon_d / \varepsilon_{d+2}$ (per round)")
    ax_l.legend(loc="upper right", fontsize=7, title="d")
    if np.isfinite(p_th):
        ax_l.axvspan(p_th, p[-1] * 1.1, color="#f3f4f6", zorder=0)
        ax_l.text(p_th, 0.15, " above threshold", fontsize=7, color="#6b7280")

    for ax in (ax_p, ax_l):
        ticks = [v for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if p[0] * 0.9 <= v <= p[-1] * 1.1]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())

    def update(b: int):
        events = detection_events(result["samples"][(d_max, b)][None])[0].numpy()
        image.set_data(events)
        title_s.set_text(f"d = {d_max}, p = {p[b]:.3f}: {int(events.sum())} detection events")
        for a, line in enumerate(lines):
            keep = pl[a, : b + 1] > 0
            line.set_data(p[: b + 1][keep], pl[a, : b + 1][keep])
        if shade[0] is not None:
            shade[0].remove()
        lo = np.clip(pl[:, : b + 1] - err[:, : b + 1], 1e-9, None)
        shade[0] = ax_p.vlines(
            np.broadcast_to(p[: b + 1], lo.shape).ravel(),
            lo.ravel(),
            (pl[:, : b + 1] + err[:, : b + 1]).ravel(),
            colors=np.repeat(bars, b + 1),
            lw=1,
        )
        for k, line in enumerate(lam_lines):
            line.set_data(p[: b + 1], lam_np[k, : b + 1])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(p), len(p)), path, fps=2)
