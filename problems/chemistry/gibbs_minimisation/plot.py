"""Animate a temperature sweep: equilibrium mole fractions x_j(T), the composition at the cursor, and T_ad(φ)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

X_MIN = 1e-6  # lower edge of the log axes
COLORS = plt.get_cmap("tab10").colors
LABELS = {"CO2": "CO₂", "H2O": "H₂O", "H2": "H₂", "O2": "O₂", "N2": "N₂"}


def animate(
    T: torch.Tensor,
    x: torch.Tensor,
    G: torch.Tensor,
    phis: torch.Tensor,
    T_ad: torch.Tensor,
    T_frozen: torch.Tensor,
    phi0: float,
    T_ad0: float,
    names: list[str],
    path: Path,
) -> Path:
    t, xs, gs = T.cpu().numpy(), x.cpu().numpy(), G.cpu().numpy()  # (B,), (B, S), (B,)
    ph, ta, tf = phis.cpu().numpy(), T_ad.cpu().numpy(), T_frozen.cpu().numpy()
    labels = [LABELS.get(s, s) for s in names]
    S = len(names)

    fig = plt.figure(figsize=(8.2, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.2, 1])
    ax_x = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, 1])

    # --- mole fractions against temperature ---
    for j in range(S):
        ax_x.plot(t, np.clip(xs[:, j], X_MIN / 10, 1), color=COLORS[j], lw=1.6)
    for j, y_label in _spread(xs[-1]).items():  # label each curve at the hot end, without overlaps
        ax_x.text(t[-1] + 15, y_label, labels[j], color=COLORS[j], fontsize=7, va="center")
    ax_x.set_yscale("log")
    ax_x.minorticks_off()
    ax_x.set_ylim(X_MIN, 1.5)
    ax_x.set_xlim(t[0], t[-1] + 180)
    ax_x.set_xlabel("temperature T (K)")
    ax_x.set_ylabel("equilibrium mole fraction $x_j$")
    ax_x.set_title(f"CH₄ + air products, φ = {phi0:.2f}", fontsize=10)
    if t[0] < T_ad0 < t[-1]:
        ax_x.axvline(T_ad0, color="#f59e0b", ls="--", lw=1.2)
        ax_x.text(T_ad0 - 20, 2.2 * X_MIN, f"$T_{{ad}}$ = {T_ad0:.0f} K", color="#b45309", fontsize=8, ha="right")
    cursor = ax_x.axvline(t[0], color="#374151", lw=1)

    # --- composition at the cursor temperature ---
    y = np.arange(S)[::-1]
    bars = ax_b.barh(y, np.full(S, X_MIN), left=X_MIN, color=COLORS[:S], height=0.7)
    ax_b.set_xscale("log")
    ax_b.minorticks_off()
    ax_b.set_xlim(X_MIN, 1.5)
    ax_b.set_yticks(y, labels, fontsize=7)
    ax_b.tick_params(axis="x", labelsize=7)
    ax_b.grid(axis="y", visible=False)
    title = ax_b.set_title("", fontsize=9)

    # --- adiabatic flame temperature against equivalence ratio ---
    ax_t.plot(ph, tf, "--", color="#9ca3af", lw=1.4, label="no dissociation")
    ax_t.plot(ph, ta, color="#dc2626", lw=1.8, label="equilibrium")
    ax_t.plot([phi0], [T_ad0], "o", color="#dc2626", ms=5)
    ax_t.set_xlabel("equivalence ratio φ", fontsize=8, labelpad=1)
    ax_t.set_ylabel("$T_{ad}$ (K)", fontsize=8)
    ax_t.tick_params(labelsize=7)
    ax_t.set_title("adiabatic flame temperature", fontsize=9)
    ax_t.legend(fontsize=7, loc="lower center")

    def update(i: int):
        cursor.set_xdata([t[i], t[i]])
        for j, bar in enumerate(bars):
            bar.set_width(max(xs[i, j], X_MIN) - X_MIN)
        title.set_text(f"T = {t[i]:.0f} K   ·   G/RT = {gs[i]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 60), path, fps=10)


def _spread(values: np.ndarray, gap: float = 0.17) -> dict[int, float]:
    """Label heights for the visible curves: start at each curve's end and push neighbours apart (both ways)
    until they are at least `gap` decades apart on the log axis."""
    order = [j for j in np.argsort(values) if values[j] > X_MIN]
    h = np.log10(values[order])
    for _ in range(200):
        for k in range(len(h) - 1):
            overlap = gap - (h[k + 1] - h[k])
            if overlap > 0:
                h[k] -= overlap / 2
                h[k + 1] += overlap / 2
    h -= max(0.0, h.max() - 0.1)  # keep the highest label inside the axes
    return {j: 10**hk for j, hk in zip(order, h)}
