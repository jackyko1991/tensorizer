"""Left: every level as the spin–orbit strength grows. Right: the level scheme with the shell gaps marked."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .solve import largest_gaps, level_scheme
from .tensors import L_LETTERS, label


def animate(strengths: torch.Tensor, energies: torch.Tensor, ell: torch.Tensor, j: torch.Tensor, path: Path) -> Path:
    """strengths (S,), energies (S, K, n_radial) with NaN for unbound levels."""
    s = strengths.cpu().numpy()
    e = energies.cpu().numpy()
    l_np, j_np = ell.cpu().numpy().astype(int), j.cpu().numpy()
    e_min = np.nanmin(e) * 1.15  # room for the legend below the deepest level

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(8.5, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}, sharey=True)
    ax_s.set_xlim(0, max(s[-1], 1e-9))
    ax_s.set_ylim(e_min, 0)
    ax_s.set_xlabel(r"spin–orbit strength $V_{so}$ (MeV)")
    ax_s.set_ylabel("single-particle energy E (MeV)")
    traces = {}
    for k in range(e.shape[1]):
        for n in range(e.shape[2]):
            if not np.all(np.isnan(e[:, k, n])):
                style = "-" if j_np[k] > l_np[k] else "--"  # solid j = l + ½, dashed j = l − ½
                traces[(k, n)] = ax_s.plot([], [], style, color=ACCENT[l_np[k] % len(ACCENT)], lw=1.2)[0]
    for ll in range(int(l_np.max()) + 1):
        ax_s.plot([], [], color=ACCENT[ll % len(ACCENT)], lw=2, label=L_LETTERS[ll])
    ax_s.legend(loc="lower left", fontsize=7, ncol=5, title="l (solid: j = l+½)", title_fontsize=7)

    ax_l.set_xlim(0, 1)
    ax_l.set_xticks([])
    ax_l.grid(False)
    drawn: list = []

    def update(step: int):
        for (k, n), line in traces.items():
            line.set_data(s[: step + 1], e[: step + 1, k, n])
        for artist in drawn:
            artist.remove()
        drawn.clear()
        scheme = level_scheme(energies[step], ell, j)
        energy = scheme["energy"].cpu().numpy()
        label_y = _spread(energy, 0.022 * abs(e_min), top=-0.01 * abs(e_min))
        for i, (en, k, n) in enumerate(zip(energy, scheme["channel"].tolist(), scheme["radial"].tolist())):
            colour = ACCENT[l_np[k] % len(ACCENT)]
            drawn.extend(ax_l.plot([0.05, 0.4], [en, en], color=colour, lw=2))
            drawn.extend(ax_l.plot([0.4, 0.47], [en, label_y[i]], color="#d1d5db", lw=0.6))
            drawn.append(ax_l.text(0.48, label_y[i], label(n + 1, l_np[k], j_np[k]), va="center", fontsize=6))
        gaps, filled = largest_gaps(scheme, 7)
        for number in filled.tolist():
            i = int(np.searchsorted(scheme["filled"].cpu().numpy(), number))
            mid = 0.5 * (energy[i] + energy[i + 1])
            drawn.append(
                ax_l.text(0.82, mid, f"{int(number)}", color="#b45309", fontsize=10, weight="bold", va="center")
            )
        ax_l.set_title(f"$V_{{so}}$ = {s[step]:.1f} MeV  (gold: 7 largest gaps)", fontsize=9)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(s), 60), path, fps=10)


def _spread(y: np.ndarray, gap: float, top: float = 0.0) -> np.ndarray:
    """Push sorted label positions apart so neighbouring labels never overlap, keeping them below `top`."""
    out = y.copy()
    for i in range(1, len(out)):  # upwards: each label at least `gap` above the previous one
        out[i] = max(out[i], out[i - 1] + gap)
    out[-1] = min(out[-1], top)
    for i in range(len(out) - 2, -1, -1):  # downwards: squeeze back under the top
        out[i] = min(out[i], out[i + 1] - gap)
    return out
