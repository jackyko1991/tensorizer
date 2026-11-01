"""Static figure: sequence logo, log-odds matrix, null score distribution and the scan along the sequence."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from problems._common.plotting import save_png

from .tensors import BASES

LETTER_COLORS = {"A": "#16a34a", "C": "#2563eb", "G": "#f59e0b", "T": "#dc2626"}
PLUS, MINUS = "#2563eb", "#ea580c"
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def figure(
    pwm: dict[str, torch.Tensor],
    scores: torch.Tensor,
    null: torch.Tensor,
    s_thr: float,
    p_max: float,
    starts: torch.Tensor,
    strands: torch.Tensor,
    hits: list[tuple[int, int]],
    path: Path,
) -> Path:
    P, W, IC = pwm["P"].cpu().numpy(), pwm["W"].cpu().numpy(), pwm["IC"].cpu().numpy()
    sc, nl = scores.cpu().numpy(), null.flatten().cpu().numpy()
    m = P.shape[0]

    fig = plt.figure(figsize=(9.6, 5.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1.05])
    ax_logo = fig.add_subplot(grid[0, 0])
    ax_w = fig.add_subplot(grid[0, 1])
    ax_h = fig.add_subplot(grid[0, 2])
    ax_s = fig.add_subplot(grid[1, :])

    # --- sequence logo: column height = information content, letters scaled by P_ja · IC_j ---
    for j in range(m):
        y = 0.0
        for a in np.argsort(P[j]):  # smallest at the bottom, the preferred letter on top
            h = P[j, a] * IC[j]
            if h > 0.01:
                _letter(ax_logo, BASES[a], j + 1, y, h)
            y += h
    ax_logo.set_xlim(0.5, m + 0.5)
    ax_logo.set_ylim(0, 2)
    ax_logo.set_xticks(range(1, m + 1))
    ax_logo.tick_params(axis="x", labelsize=7)
    ax_logo.set_ylabel("bits")
    ax_logo.set_xlabel("position $j$", labelpad=1)
    ax_logo.set_title(f"sequence logo, total {IC.sum():.1f} bits", fontsize=10)
    ax_logo.grid(False)

    # --- log-odds matrix W_ja ---
    lim = np.abs(W).max()
    im = ax_w.imshow(W.T, cmap=plt.get_cmap("RdBu_r", 12), vmin=-lim, vmax=lim, aspect="auto")
    ax_w.set_yticks(range(4), list(BASES))
    ax_w.set_xticks(range(0, m, 2), [str(j + 1) for j in range(0, m, 2)])
    ax_w.set_xlabel("position $j$", labelpad=1)
    ax_w.set_title(r"log-odds $W_{ja}$ (bits)", fontsize=10)
    ax_w.grid(False)
    fig.colorbar(im, ax=ax_w, fraction=0.06, pad=0.03)

    # --- null distribution from shuffled sequences vs the real scan ---
    bins = np.linspace(min(nl.min(), sc.min()), max(nl.max(), sc.max()) + 0.5, 50)
    ax_h.hist(nl, bins=bins, color="#9ca3af", density=True, label="shuffled")
    ax_h.hist(sc.ravel(), bins=bins, histtype="step", color="#111827", density=True, label="sequence")
    ax_h.axvline(s_thr, color="#dc2626", ls="--", lw=1.2)
    ax_h.set_yscale("log")
    ax_h.set_xlabel("score (bits)", labelpad=1)
    ax_h.set_title(f"null scores, cut at p = {p_max:.0e}", fontsize=10)
    ax_h.legend(fontsize=7, loc="upper left")

    # --- scan along the sequence: plus strand up, minus strand mirrored down ---
    x = np.arange(sc.shape[1])
    ax_s.plot(x, np.clip(sc[0], 0, None), color=PLUS, lw=0.6, label="+ strand")
    ax_s.plot(x, -np.clip(sc[1], 0, None), color=MINUS, lw=0.6, label="− strand")
    for sign in (1, -1):
        ax_s.axhline(sign * s_thr, color="#dc2626", ls="--", lw=1)
    top = max(sc.max(), s_thr) * 1.25
    for s0, sd in zip(starts.tolist(), strands.tolist()):
        ax_s.plot([s0], [sd * top * 0.93], "v" if sd > 0 else "^", color="#374151", ms=6)
    for h, hs in hits:
        ax_s.plot([h], [hs * sc[0 if hs > 0 else 1, h]], "o", ms=7, mfc="none", mew=1.5, color="#111827")
    ax_s.set_xlim(0, len(x))
    ax_s.set_ylim(-top, top)
    ax_s.set_xlabel("window start $n$ (bp)", labelpad=1)
    ax_s.set_ylabel("score (bits, > 0 shown)")
    ax_s.set_title("scan: ▼▲ planted sites, ○ hits above the threshold (dashed)", fontsize=10)
    ax_s.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.tight_layout()
    return save_png(fig, path)


def _letter(ax: plt.Axes, char: str, x: float, y: float, height: float) -> None:
    """Draw one logo letter filling the box [x - 0.45, x + 0.45] × [y, y + height]."""
    tp = TextPath((0, 0), char, size=1, prop=FONT)
    box = tp.get_extents()
    t = Affine2D().translate(-box.x0, -box.y0).scale(0.9 / box.width, height / box.height).translate(x - 0.45, y)
    ax.add_patch(PathPatch(t.transform_path(tp), color=LETTER_COLORS[char], lw=0))
