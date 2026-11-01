"""Animate binding: an SPR-style sensorgram, the binding isotherms θ(L), and receptors bound on a patch of cell."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

N_RECEPTORS = 16  # receptors drawn on each strip of membrane


def animate(
    times: torch.Tensor,
    theta: torch.Tensor,
    L: torch.Tensor,
    L_fine: torch.Tensor,
    theta_eq: torch.Tensor,
    names: list[str],
    t_assoc: float,
    path: Path,
) -> Path:
    stride = max(1, len(times) // 1500)  # RK4 takes small steps; ~1500 points are plenty for drawing
    t = times[::stride].cpu().numpy()
    th = theta[::stride].cpu().numpy()  # (T, B, C); the last column c = -1 is the sensorgram concentration
    conc = L.cpu().numpy()
    L_f, th_eq = L_fine.cpu().numpy(), theta_eq.cpu().numpy()
    B = th.shape[1]
    colors = [f"C{b}" for b in range(B)]

    fig = plt.figure(figsize=(8.2, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.35, 1])
    ax_s = fig.add_subplot(grid[:, 0])
    ax_i = fig.add_subplot(grid[0, 1])
    ax_r = fig.add_subplot(grid[1, 1])

    # --- sensorgram: θ_b(t) at one ligand concentration ---
    ax_s.set_xlim(0, t[-1])
    ax_s.set_ylim(0, 1.12)
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylabel(r"fraction bound  $\theta = C / R_T$")
    ax_s.set_title(f"SPR sensorgram, L = {_conc_label(conc[-1])}")
    ax_s.axvline(t_assoc, color="#9ca3af", ls="--", lw=1)
    ax_s.text(t_assoc * 0.5, 1.1, "ligand on", ha="center", va="top", fontsize=8, color="#6b7280")
    ax_s.text(t_assoc + (t[-1] - t_assoc) * 0.5, 1.1, "buffer wash", ha="center", va="top", fontsize=8, color="#6b7280")
    traces = [ax_s.plot([], [], lw=1.8, color=colors[b], label=names[b])[0] for b in range(B)]
    ax_s.legend(loc="center left", bbox_to_anchor=(0.1, 0.66), fontsize=7.5)
    clock = ax_s.text(0.02, 0.06, "", transform=ax_s.transAxes, fontsize=9)

    # --- isotherms: equilibrium θ vs log L (dashed), current θ_bc(t) (dots), half point marked ---
    ax_i.set_xscale("log")
    ax_i.set_xlim(L_f[0], L_f[-1])
    ax_i.minorticks_off()  # fewer tick artists to redraw in every frame
    ax_i.set_xticks([1e-2, 1, 1e2, 1e4], ["0.01", "1", "100", "10⁴"])  # plain labels: no mathtext per frame
    ax_i.set_ylim(0, 1.05)
    ax_i.set_xlabel("ligand L (nM)", labelpad=1)
    ax_i.set_ylabel("θ")
    ax_i.set_title("binding isotherms", fontsize=10)
    for b in range(B):
        ax_i.plot(L_f, th_eq[b], "--", lw=1, color=colors[b])
        if th_eq[b, 0] <= 0.5 <= th_eq[b, -1]:  # where θ = ½: at L = K_D (at K_D + R_T/2 with depletion)
            ax_i.plot([np.interp(0.5, th_eq[b], L_f)], [0.5], "|", ms=9, mew=1.5, color=colors[b])
    ax_i.text(0.03, 0.9, "| : half bound, L = $K_D$", transform=ax_i.transAxes, fontsize=7.5)
    dots = [ax_i.plot([], [], "o", ms=3, color=colors[b])[0] for b in range(B)]

    # --- receptors on a patch of membrane (sensorgram concentration) ---
    ax_r.set_xlim(-0.8, N_RECEPTORS + 2.2)
    ax_r.set_ylim(-0.6, B - 0.1)
    ax_r.axis("off")
    ax_r.set_title(f"{N_RECEPTORS} receptors per cell patch", fontsize=10)
    x = np.arange(N_RECEPTORS)
    for b in range(B):
        ax_r.plot([-0.6, N_RECEPTORS - 0.4], [B - 1 - b - 0.25] * 2, color="#374151", lw=1.5)  # membrane
        ax_r.plot(*_receptor_shapes(x, B - 1 - b), color="#9ca3af", lw=1.5)
    ligands = [ax_r.plot([], [], "o", ms=4.5, color=colors[b])[0] for b in range(B)]
    counts = [ax_r.text(N_RECEPTORS, B - 1 - b, "", fontsize=8, va="center", color=colors[b]) for b in range(B)]

    def update(step: int):
        for b in range(B):
            traces[b].set_data(t[: step + 1], th[: step + 1, b, -1])
            dots[b].set_data(conc[:-1], th[step, b, :-1])
            n_bound = int(round(th[step, b, -1] * N_RECEPTORS))
            ligands[b].set_data(x[:n_bound], [B - 1 - b + 0.32] * n_bound)
            counts[b].set_text(f"{n_bound}/{N_RECEPTORS}")
        clock.set_text(f"t = {t[step]:.0f} s")

    fig.tight_layout(rect=(0, 0, 0.98, 1))  # a little room for the last tick label
    return save_gif(fig, update, frame_indices(len(t), 90), path)


def _receptor_shapes(x: np.ndarray, y: float) -> tuple[np.ndarray, np.ndarray]:
    """One polyline of Y-shaped receptors (stem + two arms) at heights around y, separated by NaN gaps."""
    xs, ys = [], []
    for xi in x:
        xs += [xi, xi, np.nan, xi - 0.3, xi, xi + 0.3, np.nan]
        ys += [y - 0.25, y, np.nan, y + 0.22, y, y + 0.22, np.nan]
    return np.array(xs), np.array(ys)


def _conc_label(nM: float) -> str:
    return f"{nM / 1e3:.3g} µM" if nM >= 1e3 else f"{nM:.3g} nM"
