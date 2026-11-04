"""Animate the vibrating string, its level in the mass spectrum, and T-duality of the circle spectrum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, save_gif


def animate(sheet: dict, kind: str, spec: dict, level: int, radii, curves, names, duality_gap, path: Path) -> Path:
    """sheet["X"]: (F, S, 3) one string shape per frame; radii (F,), curves (F, 5) α′M²(R) and (F, 5) at α′/R."""
    X = sheet["X"].cpu().numpy()
    if kind == "closed":
        X = np.concatenate([X, X[:, :1]], axis=1)  # close the loop for drawing
    R = radii.cpu().numpy()
    M2_R, M2_dual = (c.cpu().numpy() for c in curves)
    gap = duality_gap.cpu().numpy()
    M2_lvl, deg = spec["M2"].numpy(), spec["degeneracy"].numpy()

    fig = plt.figure(figsize=(10.5, 3.9))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax_s = fig.add_subplot(1, 3, 2)
    ax_t = fig.add_subplot(1, 3, 3)

    reach = 1.05 * np.abs(X).max()
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-reach, reach)
    ax.set_xlabel("$X^1$", labelpad=-8)
    ax.set_ylabel("$X^2$", labelpad=-8)
    ax.set_zlabel("$X^3$", labelpad=-8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    line = ax.plot([], [], [], "-", lw=2, color=ACCENT[0])[0]
    dot = ax.plot([], [], [], "o", ms=5, color=ACCENT[1])[0]
    shadow = ax.plot([], [], [], "-", lw=1, color="#9ca3af")[0]
    title = ax.set_title("", fontsize=8)
    lvl_name = "N + Ñ" if kind == "closed" else "N"

    # --- spectrum: log10 degeneracy at each mass level ---
    colours = ["#dc2626" if m < 0 else "#93c5fd" for m in M2_lvl]
    colours[level] = ACCENT[0] if M2_lvl[level] >= 0 else "#dc2626"
    width = 0.7 * (M2_lvl[1] - M2_lvl[0])
    bars = ax_s.bar(M2_lvl, np.log10(deg) + 0.4, bottom=-0.4, width=width, color=colours, edgecolor="#1e40af", lw=0.5)
    bars[level].set_edgecolor("#111827")
    bars[level].set_linewidth(2)
    for m, d in zip(M2_lvl, deg):
        ax_s.text(m, np.log10(d) + 0.15, f"{d:.3g}" if d < 1e5 else f"{d:.1e}", ha="center", fontsize=6)
    ax_s.set_xlabel("$\\alpha' M^2$" + ("  ($=4(N-1)$)" if kind == "closed" else "  ($=N-1$)"))
    ax_s.set_ylabel("$\\log_{10}$ degeneracy")
    ax_s.set_ylim(-0.4, np.log10(deg).max() * 1.15 + 0.5)
    ax_s.set_title(f"{kind} string levels from $\\prod(1-q^n)^{{-24}}$", fontsize=9)
    ax_s.text(M2_lvl[0], 0.9, "tachyon", color="#dc2626", ha="center", fontsize=7, rotation=90)
    marker = ax_s.text(0.97, 0.95, "", transform=ax_s.transAxes, ha="right", va="top", fontsize=8)

    # --- T-duality: α′M²(R) on a log axis, mirror image about R = √α′ ---
    grid = np.logspace(np.log10(R.min()), np.log10(R.max()), 300)
    s = np.array([(0, 0, 0, 0), (0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 0)], float)
    full = (s[:, 0] / grid[:, None]) ** 2 + (s[:, 1] * grid[:, None]) ** 2 + 2 * (s[:, 2] + s[:, 3] - 2)
    for j, name in enumerate(names):
        ax_t.plot(grid, full[:, j], color=ACCENT[j % len(ACCENT)], lw=1.3, label=name)
    ax_t.set_xscale("log")
    ax_t.set_ylim(-5, 17)
    ticks = [t for t in (1 / 5, 1 / 3, 1 / 2, 1, 2, 3, 5) if R.min() * 0.99 <= t <= R.max() * 1.01]
    ax_t.set_xticks(ticks, [f"{t:.2g}" for t in ticks])
    ax_t.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_t.set_xlabel("circle radius $R/\\sqrt{\\alpha'}$")
    ax_t.set_ylabel("$\\alpha' M^2$")
    ax_t.axvline(1.0, color="#9ca3af", ls="--", lw=1)
    ax_t.legend(fontsize=6, loc="upper center", ncol=2)
    cur = ax_t.axvline(R[0], color="#111827", lw=1)
    cur_d = ax_t.axvline(1 / R[0], color="#f59e0b", lw=1)
    pts = ax_t.plot([], [], "o", ms=5, color="#111827", mfc="none")[0]
    pts_d = ax_t.plot([], [], "x", ms=5, color="#f59e0b")[0]
    t_title = ax_t.set_title("", fontsize=8)

    def update(f: int):
        line.set_data(X[f, :, 0], X[f, :, 1])
        line.set_3d_properties(X[f, :, 2])
        shadow.set_data(X[f, :, 0], X[f, :, 1])
        shadow.set_3d_properties(np.full(len(X[f]), -reach))
        dot.set_data([X[f, 0, 0]], [X[f, 0, 1]])
        dot.set_3d_properties([X[f, 0, 2]])
        tau = 2 * np.pi * f / max(1, len(X) - 1)
        title.set_text(f"τ = {tau:4.2f}   {lvl_name} from (1/4πα′)∫(Ẋ²+X′²)dσ ={float(sheet['level_field'][f]):.4f}")
        marker.set_text(f"this string: N = {level}" + ("" if kind == "open" else " = Ñ"))
        cur.set_xdata([R[f]])
        cur_d.set_xdata([1 / R[f]])
        pts.set_data(np.full(len(names), R[f]), M2_R[f])
        pts_d.set_data(np.full(len(names), 1 / R[f]), M2_dual[f])
        t_title.set_text(f"T-duality R={R[f]:.2f} ↔ α′/R={1 / R[f]:.2f}\nfull spectra agree: max |ΔM²| = {gap[f]:.1e}")

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(X))), path, fps=15)
