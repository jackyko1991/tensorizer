"""Win rates converging inside their binomial bands, and the final rates against the number of doors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import binomial_band, exact_rates, valid_fraction


def animate(doors: torch.Tensor, res: dict, host: str, path: Path) -> Path:
    """doors (B,), res['stay'] and res['switch'] running rates (B, G)."""
    g = res["stay"].shape[1]
    curve = np.unique(np.geomspace(1, g, 400).astype(int))  # log-spaced game counts for the curves
    shown = np.unique(np.geomspace(10, g, 90).astype(int))  # game counts shown frame by frame
    stay, switch = res["stay"].cpu().numpy(), res["switch"].cpu().numpy()
    stay_th, switch_th = (t.numpy() for t in exact_rates(doors, host))
    ns = doors.numpy()
    colours = plt.cm.Blues(np.linspace(0.5, 1.0, len(ns)))
    share = valid_fraction(doors, host).numpy()  # the band narrows with the number of games that count
    n_curve = torch.tensor(curve, dtype=torch.float64)

    fig, (ax_c, ax_n) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    picked = sorted({0, len(ns) // 2, len(ns) - 1})  # a few door counts on the left, all of them on the right
    lines = []
    for b in picked:
        band = binomial_band(torch.tensor(switch_th[b]), n_curve * share[b]).numpy()
        ax_c.fill_between(curve, switch_th[b] - band, switch_th[b] + band, color=colours[b], alpha=0.15, lw=0)
        ax_c.axhline(switch_th[b], color=colours[b], ls="--", lw=0.8)
        (line,) = ax_c.semilogx([], [], color=colours[b], lw=1.6, label=f"switch, N = {ns[b]}")
        lines.append((line, switch[b]))
    band = binomial_band(torch.tensor(stay_th[0]), n_curve * share[0]).numpy()
    ax_c.fill_between(curve, stay_th[0] - band, stay_th[0] + band, color="#f59e0b", alpha=0.18, lw=0)
    ax_c.axhline(stay_th[0], color="#f59e0b", ls="--", lw=0.8)
    (line,) = ax_c.semilogx([], [], color="#ea580c", lw=1.6, label=f"stay, N = {ns[0]}")
    lines.append((line, stay[0]))
    ax_c.set_xlim(1, g)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_xlabel("games played n")
    ax_c.set_ylabel("win rate")
    ax_c.set_title("running win rate (dashed: exact, band: ±2σ)", fontsize=10)
    ax_c.legend(fontsize=8, frameon=False, loc="center right")

    nn = np.geomspace(3, max(ns.max(), 4), 200)
    if host == "knowing":
        ax_n.semilogx(nn, (nn - 1) / nn, color="#2563eb", lw=1.2, ls="--", label="switch: (N − 1)/N")
        ax_n.semilogx(nn, 1 / nn, color="#ea580c", lw=1.2, ls="--", label="stay: 1/N")
    else:
        ax_n.semilogx(nn, 0.5 + 0 * nn, color="#6b7280", lw=1.2, ls="--", label="1/2")
    ax_n.set_ylim(-0.02, 1.02)
    ax_n.set_xlim(2.5, nn[-1] * 1.4)
    ax_n.set_xlabel("number of doors N")
    ax_n.legend(fontsize=8, frameon=False, loc="center right")
    title = ax_n.set_title("", fontsize=10)
    markers: list = []

    def update(k: int):
        n = shown[k]
        mask = curve <= n
        for line, data in lines:
            line.set_data(curve[mask], data[curve[mask] - 1])
        for m in markers:
            m.remove()
        markers.clear()
        for rates, th, fmt, col in ((switch, switch_th, "o", "#2563eb"), (stay, stay_th, "s", "#ea580c")):
            err = binomial_band(torch.tensor(th), torch.tensor(n * share).clamp(min=1)).numpy()
            markers.append(ax_n.errorbar(ns, rates[:, n - 1], yerr=err, fmt=fmt, color=col, ms=5, capsize=3))
        title.set_text(f"after {n:,} games per N ({host} host)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(shown), 90), path, fps=10)
