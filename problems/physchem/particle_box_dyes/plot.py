"""Animate the series: each box with its filled levels and HOMO→LUMO jump, λ_max against chain length, colours."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from problems._common.plotting import frame_indices, save_gif

FRAMES_PER_MOLECULE = 10


def _rgb(lam_nm: float) -> tuple[float, float, float]:
    """Display colour for one wavelength (grey outside the visible range)."""
    if not 380 <= lam_nm <= 780:
        return (0.42, 0.45, 0.5)
    from .equation import wavelength_rgb

    r, g, b = wavelength_rgb(torch.tensor([lam_nm])).squeeze(0).tolist()
    return (0.85 * r, 0.85 * g, 0.85 * b)  # a little darker so pale yellows stay visible on white


def animate(names, L, n_bonds, bond, end, n_el, E_num, psi, x, lam, lam_num, colour, measured, path: Path) -> Path:
    L, n_bonds, n_el, E, psi, x = (
        L.cpu().numpy(),
        n_bonds.cpu().numpy(),
        n_el.cpu().numpy(),
        E_num.cpu().numpy(),
        psi.cpu().numpy(),
        x.cpu().numpy(),
    )
    lam, lam_num, colour = lam.cpu().numpy(), lam_num.cpu().numpy(), colour.cpu().numpy()
    B = len(names)
    homo = n_el // 2 - 1  # 0-based index of the HOMO
    e_top = max(E[b, homo[b] + 1] for b in range(B)) * 1.12

    fig = plt.figure(figsize=(8.4, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.6, 1])
    ax_box = fig.add_subplot(grid[:, 0])
    ax_lam = fig.add_subplot(grid[0, 1])
    ax_sw = fig.add_subplot(grid[1, 1])

    # --- λ_max against chain length ---
    k = np.arange(B)
    ax_lam.imshow(
        _rainbow(), extent=(-0.6, B - 0.4, 380, 780), aspect="auto", origin="lower", alpha=0.22, zorder=0
    )  # the visible band
    ax_lam.plot(k, lam, "-", color="#374151", lw=1.4, label="analytic  $h^2(N+1)/8mL^2$")
    ax_lam.plot(k, lam_num, "o", ms=4.5, mfc="none", color="#2563eb", label="numerical eigh")
    if measured is not None:
        km, lm = (t.cpu().numpy() for t in measured)
        ax_lam.plot(km, lm, "s", ms=4, color="#dc2626", label="measured (approx.)")
    (mark,) = ax_lam.plot([], [], "o", ms=9, color="#f59e0b", zorder=5)
    ax_lam.set_xlim(-0.6, B - 0.4)
    ax_lam.set_ylim(min(150, lam.min() * 0.9), max(820, lam.max() * 1.08))
    ax_lam.set_xticks(k, [n.replace("-methine", "") for n in names])
    ax_lam.set_xlabel("CH groups in the chain" if names[0].endswith("methine") else "C=C bonds", labelpad=1)
    ax_lam.set_ylabel(r"$\lambda_{max}$ (nm)")
    ax_lam.set_title("longer box → smaller gap → redder light", fontsize=10)
    ax_lam.legend(fontsize=7, loc="upper left")

    # --- colour swatches: what each dye looks like (white light minus its absorption band) ---
    ax_sw.set_xlim(-0.6, B - 0.4)
    ax_sw.set_ylim(-0.75, 1.0)
    ax_sw.axis("off")
    ax_sw.set_title("colour of the solution", fontsize=10)
    frames_sw = []
    for b in range(B):
        ax_sw.add_patch(Rectangle((b - 0.38, 0.0), 0.76, 0.8, color=colour[b], ec="#9ca3af"))
        ax_sw.text(b, -0.25, f"{lam_num[b]:.0f} nm", ha="center", fontsize=7.5, color=_rgb(lam_num[b]))
        frame = Rectangle((b - 0.45, -0.07), 0.9, 0.94, fill=False, ec="#f59e0b", lw=2.5, visible=False)
        ax_sw.add_patch(frame)
        frames_sw.append(frame)
    ax_sw.text((B - 1) / 2, -0.62, "UV absorbers look colourless", ha="center", fontsize=7, color="#6b7280")

    def draw_box(b: int, f: float):
        ax_box.cla()
        ax_box.set_xlim(-0.08, L.max() + 0.08)
        ax_box.set_ylim(-0.12 * e_top, e_top)
        ax_box.set_xlabel("position along the chain (nm)", labelpad=1)
        ax_box.set_ylabel("energy (eV)")
        ax_box.set_title(f"{names[b]}: {n_el[b]} π electrons in L = {L[b]:.2f} nm", fontsize=10)
        ax_box.plot([0, 0, L[b], L[b]], [e_top, 0, 0, e_top], color="#374151", lw=2)  # the box walls
        h = homo[b]
        for n in range(h + 2):  # filled levels and the LUMO
            ax_box.plot([0, L[b]], [E[b, n]] * 2, color="#9ca3af" if n <= h else "#dc2626", lw=1)
            if n < h:
                ax_box.plot([0.42 * L[b], 0.58 * L[b]], [E[b, n]] * 2, "o", ms=4, color="#2563eb")
        dE = E[b, h + 1] - E[b, h]
        for n, c in ((h, "#2563eb"), (h + 1, "#dc2626")):  # HOMO and LUMO wavefunctions, drawn at their energy
            ax_box.plot(x[b], E[b, n] + 0.35 * dE * psi[b, :, n] / np.abs(psi[b, :, n]).max(), color=c, lw=1.2)
        # one HOMO electron stays, the other jumps to the LUMO as f goes 0 → 1
        ax_box.plot([0.42 * L[b]], [E[b, h]], "o", ms=4, color="#2563eb")
        ax_box.plot([0.58 * L[b]], [E[b, h] + f * dE], "o", ms=5, color="#2563eb")
        if f > 0.05:
            ax_box.annotate(
                "", xy=(0.7 * L[b], E[b, h] + f * dE), xytext=(0.7 * L[b], E[b, h]),
                arrowprops={"arrowstyle": "->", "color": _rgb(lam_num[b]), "lw": 2},
            )  # fmt: skip
        ax_box.text(
            0.74 * L[b], E[b, h] + 0.5 * dE, f"{lam_num[b]:.0f} nm", fontsize=8, va="center", color=_rgb(lam_num[b])
        )
        ax_box.text(L[b] + 0.03, E[b, h], "HOMO", fontsize=7, va="center", color="#2563eb")
        ax_box.text(L[b] + 0.03, E[b, h + 1], "LUMO", fontsize=7, va="center", color="#dc2626")
        # the conjugated chain drawn as a zigzag inside the bottom of the box
        xs = end * bond + bond * np.arange(n_bonds[b] + 1)  # atom positions: one bond length apart
        ys = np.where(np.arange(n_bonds[b] + 1) % 2 == 0, -0.04, -0.09) * e_top
        ax_box.plot(xs, ys, "-o", ms=3, lw=1, color="#6b7280")

    def update(step: int):
        b, i = divmod(step, FRAMES_PER_MOLECULE)
        f = min(1.0, i / (0.6 * FRAMES_PER_MOLECULE))
        draw_box(b, f)
        mark.set_data([b], [lam_num[b]])
        for j, frame in enumerate(frames_sw):
            frame.set_visible(j == b)

    fig.tight_layout()
    n_steps = B * FRAMES_PER_MOLECULE
    return save_gif(fig, update, frame_indices(n_steps, n_steps), path, fps=8)


def _rainbow() -> np.ndarray:
    """A vertical rainbow image (380–780 nm) for the background of the λ panel."""
    from .equation import wavelength_rgb

    return wavelength_rgb(torch.linspace(380, 780, 120))[:, None, :].numpy()
