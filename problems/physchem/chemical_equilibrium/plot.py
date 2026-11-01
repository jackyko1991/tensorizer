"""Animate a pressure sweep: equilibrium conversion vs P, the mixture's composition, van 't Hoff, added product."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

SPECIES_COLORS = ["#2563eb", "#9ca3af", "#dc2626"]  # first reactant, second reactant, product


def animate(
    r: dict,
    T_list: list[float],
    P: torch.Tensor,
    conversion: torch.Tensor,
    x_eq: torch.Tensor,
    T_line: torch.Tensor,
    lnK_line: torch.Tensor,
    P_added: float,
    added: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    path: Path,
) -> Path:
    """conversion (nT, P) and x_eq (nT, P, S); the last temperature row is the chosen T (drawn in black)."""
    P, conv, x_eq = P.cpu().numpy(), conversion.cpu().numpy(), x_eq.cpu().numpy()
    T_line, lnK_line = T_line.cpu().numpy(), lnK_line.cpu().numpy()
    added = [n.cpu().numpy() for n in added]
    species = r["species"]
    S = len(species)
    sp_colors = [SPECIES_COLORS[0]] + SPECIES_COLORS[1 : S - 1] + [SPECIES_COLORS[-1]]
    n_curves = len(T_list) - 1
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, n_curves))

    fig = plt.figure(figsize=(8.6, 4.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1, 1.1])
    ax_p = fig.add_subplot(grid[:, 0])
    ax_x = fig.add_subplot(grid[0, 1])
    ax_v = fig.add_subplot(grid[1, 1])
    ax_a = fig.add_subplot(grid[2, 1])

    # --- conversion against pressure, one curve per temperature ---
    for t in range(n_curves):
        if abs(T_list[t] - T_list[-1]) < 1e-9:
            continue  # same as the chosen temperature, drawn in black below
        ax_p.plot(P, conv[t], lw=1.3, color=colors[t], label=f"{T_list[t]:.0f} K")
    ax_p.plot(P, conv[-1], lw=2.4, color="#111827", label=f"{T_list[-1]:.0f} K (chosen)")
    ax_p.axvline(P_added, color="#9ca3af", ls="--", lw=1)
    (dot,) = ax_p.plot([], [], "o", ms=8, color="#f59e0b", mec="#b45309", zorder=4)
    ax_p.set_xscale("log")
    ax_p.set_xlim(P[0], P[-1])
    ax_p.set_ylim(0, 1.02)
    ax_p.set_xlabel("total pressure P (bar)")
    ax_p.set_ylabel(r"conversion  $\xi\,/\,\xi_{\max}$")
    ax_p.set_title(r["label"], fontsize=11)
    ax_p.legend(fontsize=7, loc="best")
    p_text = ax_p.text(0.03, 0.04, "", transform=ax_p.transAxes, fontsize=9)

    # --- composition of the mixture at the current pressure (chosen T) ---
    bars = ax_x.bar(range(S), x_eq[-1, 0], color=sp_colors, width=0.6)
    ax_x.set_xticks(range(S), species)
    ax_x.set_ylim(0, 1.2)  # room for the value labels
    ax_x.set_ylabel("mole fraction", fontsize=8)
    ax_x.set_title(f"mixture at {T_list[-1]:.0f} K", fontsize=9, pad=2)
    ax_x.tick_params(labelsize=8)
    labels = [ax_x.text(i, 0, "", ha="center", va="bottom", fontsize=8) for i in range(S)]

    # --- van 't Hoff plot: ln K against 1000/T ---
    ax_v.plot(1000 / T_line, lnK_line, color="#374151", lw=1.4)
    lnK_pts = np.interp(T_list, T_line, lnK_line)
    ax_v.scatter(1000 / np.array(T_list[:-1]), lnK_pts[:-1], c=colors, s=18, zorder=3)
    ax_v.plot([1000 / T_list[-1]], [lnK_pts[-1]], "*", ms=10, color="#111827")
    ax_v.set_xlabel("1000 / T (1/K)", fontsize=8, labelpad=1)
    ax_v.set_ylabel(r"$\ln K$", fontsize=8)
    ax_v.tick_params(labelsize=7)
    slope = -r["dH"] / 8.314462618
    ax_v.set_title(f"van 't Hoff: slope $-\\Delta H°/R$ = {slope:.0f} K", fontsize=9, pad=2)

    # --- added product, grouped by stage: equilibrium → just after adding → new equilibrium ---
    width = 0.8 / S
    for i in range(S):
        stage_moles = [n[i] for n in added]
        ax_a.bar(np.arange(3) + (i - (S - 1) / 2) * width, stage_moles, width, color=sp_colors[i], label=species[i])
    ax_a.set_xticks(range(3), ["equilibrium", f"+ {species[-1]}", "new equil."])
    ax_a.set_ylabel("moles", fontsize=8)
    ax_a.tick_params(labelsize=8)
    ax_a.legend(fontsize=7, ncol=S, loc="upper left")
    ax_a.set_ylim(0, 1.3 * max(n.max() for n in added))
    ax_a.set_title(f"Le Chatelier: add product at {P_added:.3g} bar", fontsize=9, pad=2)

    def update(step: int):
        dot.set_data([P[step]], [conv[-1, step]])
        p_text.set_text(f"P = {P[step]:.3g} bar   conversion = {conv[-1, step]:.3f}")
        for i in range(S):
            bars[i].set_height(x_eq[-1, step, i])
            labels[i].set_position((i, x_eq[-1, step, i] + 0.02))
            labels[i].set_text(f"{x_eq[-1, step, i]:.2f}")

    fig.tight_layout(pad=0.6, h_pad=0.5, w_pad=1.0)
    return save_gif(fig, update, frame_indices(len(P), 61), path)
