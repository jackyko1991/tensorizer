"""Static figure: amounts against reaction extent, the yield map over both masses, and the batch of runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_png


def figure(
    xi: torch.Tensor,
    n_sweep: torch.Tensor,
    species: list[str],
    res: dict,
    weighed: torch.Tensor,
    grid_1: torch.Tensor,
    grid_2: torch.Tensor,
    product_map: torch.Tensor,
    ratio: torch.Tensor,
    m0: torch.Tensor,
    out_dir: Path,
) -> Path:
    fig, (ax_x, ax_m, ax_b) = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.1, 1.05, 1]})

    # --- (a) bars of every amount at ξ = 0 … ξ_max (your run) ---
    x, n = xi.cpu().numpy(), n_sweep.cpu().numpy() * 1e3  # mmol
    K, S = n.shape
    width = 0.8 / S
    for s in range(S):
        ax_x.bar(np.arange(K) + (s - (S - 1) / 2) * width, n[:, s], width, color=f"C{s}", label=species[s])
    lim = res["limiting"][-1].item()
    ax_x.set_xticks(range(K), [f"{f:.2g}·ξmax\n{v * 1e3:.3g}" for f, v in zip(np.linspace(0, 1, K), x)], fontsize=8)
    ax_x.set_xlabel("reaction extent ξ (mmol)")
    ax_x.set_ylabel("amount n (mmol)")
    ax_x.set_title(f"your run: {species[lim]} runs out first (limiting)", fontsize=10)
    ax_x.legend(fontsize=8)

    # --- (b) theoretical product over both masses, with the stoichiometric line ---
    g1, g2, pm = grid_1.cpu().numpy(), grid_2.cpu().numpy(), product_map.cpu().numpy()
    levels = np.linspace(0, pm.max(), 11)
    filled = ax_m.contourf(g1, g2, pm, levels=levels, cmap="Blues")
    fig.colorbar(filled, ax=ax_m, label=f"theoretical {species[2]} (g)")
    ax_m.plot(g1, ratio.item() * g1, "--", color="#dc2626", lw=1.8, label="stoichiometric ratio")
    ax_m.plot(m0[-1, 0].item(), m0[-1, 1].item(), "o", ms=8, mfc="#f59e0b", mec="#111827", label="your run")
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.8}
    ax_m.text(0.72, 0.28, f"{species[1]} limiting", ha="center", transform=ax_m.transAxes, fontsize=8, bbox=box)
    ax_m.text(0.28, 0.72, f"{species[0]} limiting", ha="center", transform=ax_m.transAxes, fontsize=8, bbox=box)
    ax_m.set_xlim(g1[0], g1[-1])
    ax_m.set_ylim(g2[0], g2[-1])
    ax_m.set_xlabel(f"mass of {species[0]} (g)")
    ax_m.set_ylabel(f"mass of {species[1]} (g)")
    ax_m.set_title("yield map: contours bend at the line", fontsize=10)
    ax_m.legend(fontsize=7.5, loc="upper left")
    ax_m.grid(False)

    # --- (c) the batch: theoretical vs weighed product, percent yield on top ---
    B = len(weighed)
    theory, pct = res["theory"].cpu().numpy(), res["percent"].cpu().numpy()
    names = [f"run {b + 1}" for b in range(B - 1)] + ["your run"]
    ax_b.bar(np.arange(B) - 0.2, theory, 0.4, color="#93c5fd", edgecolor="#1e40af", label="theoretical")
    ax_b.bar(np.arange(B) + 0.2, weighed.cpu().numpy(), 0.4, color="#2563eb", label="weighed")
    for b in range(B):
        limiting = species[res["limiting"][b].item()]
        top = max(theory[b], weighed[b].item())
        ax_b.text(b, top * 1.03, f"{pct[b]:.0f} %\n({limiting} lim.)", ha="center", fontsize=8)
    ax_b.set_xticks(range(B), names)
    ax_b.set_ylim(0, 1.25 * max(theory.max(), weighed.max().item()))
    ax_b.set_ylabel(f"{species[2]} (g)")
    ax_b.set_title("percent yield = weighed / theoretical", fontsize=10)
    ax_b.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    return save_png(fig, out_dir / "limiting_reagent_yield.png")
