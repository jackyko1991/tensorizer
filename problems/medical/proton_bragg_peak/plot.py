"""Animate: protons slow down in water while their dose builds up, then the SOBP is assembled layer by layer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

FLIGHT, BUILD = 60, 40  # logical steps of the two phases


def animate(
    z: torch.Tensor,  # (Z+1,) depth grid
    times: torch.Tensor,  # (Z+1, B) ns
    E0: torch.Tensor,  # (B,)
    ranges: torch.Tensor,  # (B,)
    z_mid: torch.Tensor,  # (Z,)
    dose: torch.Tensor,  # (B, Z)
    fit: tuple[float, float],
    reference: tuple[torch.Tensor, torch.Tensor],
    layers: torch.Tensor,  # (K, Z) weighted layer doses
    xray: torch.Tensor,  # (Z,)
    target: tuple[float, float],
    path: Path,
) -> Path:
    z, times, E0, ranges = z.cpu().numpy(), times.cpu().numpy(), E0.cpu().numpy(), ranges.cpu().numpy()
    z_mid, dose, layers, xray = z_mid.cpu().numpy(), dose.cpu().numpy(), layers.cpu().numpy(), xray.cpu().numpy()
    e_ref, r_ref = (r.cpu().numpy() for r in reference)
    alpha, p = fit
    z1, z2 = target
    n_b, n_k = len(E0), len(layers)
    colors = [plt.get_cmap("viridis")(0.1 + 0.8 * b / max(1, n_b - 1)) for b in range(n_b)]
    dose_norm = dose / dose.max()
    t_end = times[-1].max() * 1.02

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
    (ax_tank, ax_range), (ax_dose, ax_sobp) = axes
    z_max = z[-1]

    ax_tank.axvspan(0, z_max, color="#dbeafe")
    ax_tank.set_xlim(-1, z_max)
    ax_tank.set_ylim(-0.7, n_b - 0.3)
    ax_tank.set_yticks(range(n_b), [f"{e:.0f}" for e in E0])
    ax_tank.set_ylabel("$E_0$ (MeV)")
    ax_tank.set_xlabel("depth in water (cm)")
    trails = [ax_tank.plot([], [], color=colors[b], lw=2)[0] for b in range(n_b)]
    heads = [ax_tank.plot([], [], "o", color=colors[b], ms=6, mec="#1e40af")[0] for b in range(n_b)]
    clock = ax_tank.set_title("protons in water")

    e_line = np.linspace(min(E0.min(), 50), max(E0.max(), 250), 100)
    ax_range.plot(e_line, alpha * e_line**p, color="#9ca3af", lw=1.5, label=f"$R = {alpha:.4f}\\,E^{{{p:.2f}}}$")
    ax_range.plot(e_ref, r_ref, "x", color="#dc2626", ms=7, label="PSTAR")
    range_dots = ax_range.scatter([], [], color=ACCENT[0], s=22, zorder=3, label="Bethe–Bloch")
    ax_range.set_xlabel("$E_0$ (MeV)")
    ax_range.set_ylabel("range (cm)")
    ax_range.legend(fontsize=8, loc="upper left")

    dose_lines = [ax_dose.plot([], [], color=colors[b], lw=1.4)[0] for b in range(n_b)]
    ax_dose.set_xlim(0, z_max)
    ax_dose.set_ylim(0, 1.08)
    ax_dose.set_xlabel("depth (cm)")
    ax_dose.set_ylabel("dose (rel.)")
    ax_dose.set_title("pristine Bragg peaks")

    ax_sobp.axvspan(z1, z2, color="#fde68a", alpha=0.6, lw=0)
    layer_lines = [ax_sobp.plot([], [], color=ACCENT[0], lw=0.7, alpha=0.6)[0] for _ in range(n_k)]
    (sum_line,) = ax_sobp.plot([], [], color=ACCENT[1], lw=2, label="protons (SOBP)")
    (x_line,) = ax_sobp.plot([], [], color="#374151", lw=1.4, ls="--", label="X-rays")
    ax_sobp.set_xlim(0, max(z2 + 6, 0.6 * z_max))
    ax_sobp.set_ylim(0, 1.9)
    ax_sobp.set_xlabel("depth (cm)")
    ax_sobp.set_ylabel("dose (rel.)")
    ax_sobp.set_title("spread-out Bragg peak")
    ax_sobp.legend(fontsize=8, loc="upper right")

    def update(step: int):
        T = t_end * min(1.0, step / (FLIGHT - 1))
        stopped = []
        for b in range(n_b):
            depth = min(np.interp(T, times[:, b], z), ranges[b])
            trails[b].set_data([0, depth], [b, b])
            heads[b].set_data([depth], [b])
            done = depth >= ranges[b] - 1e-6
            shown = np.ones_like(z_mid, bool) if done else z_mid <= depth  # full curve once stopped
            dose_lines[b].set_data(z_mid[shown], dose_norm[b, shown])
            if done:
                stopped.append(b)
        clock.set_text(f"protons in water   t = {T:.2f} ns")
        range_dots.set_offsets(np.column_stack([E0[stopped], ranges[stopped]]) if stopped else np.empty((0, 2)))

        n_shown = 0 if step < FLIGHT else min(n_k, 1 + (step - FLIGHT) * n_k // (BUILD - 8))
        order = np.arange(n_k)[::-1]  # deepest layer first
        for i, k in enumerate(order):
            layer_lines[k].set_data((z_mid, layers[k]) if i < n_shown else ([], []))
        partial = layers[order[:n_shown]].sum(0) if n_shown else None
        sum_line.set_data((z_mid, partial) if n_shown else ([], []))
        x_line.set_data((z_mid, xray) if n_shown == n_k else ([], []))

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(FLIGHT + BUILD, 80), path, fps=10)
