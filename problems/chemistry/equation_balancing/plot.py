"""Static figure: the signed composition matrix of one reaction, the singular values of all, and the balanced list."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from problems._common.plotting import save_png

from . import tensors

FLOOR = 1e-17  # exact zeros drawn here on the log axis


def figure(
    M: torch.Tensor,
    nu: torch.Tensor,
    elements: list[str],
    species: list[str],
    show: str,
    names: list[str],
    sv: list[torch.Tensor],
    balanced: list,
    plane: list,
    out_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(11, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.15, 1])
    ax_m = fig.add_subplot(grid[0, 0])
    ax_s = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, :])

    # --- (a) signed composition matrix with the coefficients underneath ---
    m = M.cpu().numpy() + 0.0  # + 0.0 turns -0 into 0
    lim = int(np.abs(m).max())
    levels = np.arange(-lim - 0.5, lim + 1.5)
    cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.1, 0.9, len(levels) - 1)))
    ax_m.imshow(m, cmap=cmap, norm=BoundaryNorm(levels, cmap.N), aspect="auto")
    for (e, s), value in np.ndenumerate(m):
        ax_m.text(s, e, f"{value:.0f}", ha="center", va="center", fontsize=10, color="#111827")
    ax_m.set_xticks(range(len(species)), [f"${tensors.latex(f)}$\nν = {int(n)}" for f, n in zip(species, nu)])
    ax_m.set_yticks(range(len(elements)), ["charge" if e == "q" else e for e in elements])
    ax_m.grid(False)
    ax_m.set_title(f"{show}: $M_{{es}} = C_{{es}}\\,\\sigma_s$ (products < 0),  $M\\nu = 0$", fontsize=10)

    # --- (b) singular values: a zero one ⇔ a null vector ⇔ a balanced reaction ---
    for r, s in enumerate(sv):
        s = s.cpu().numpy()
        zero = s < 1e-9 * s[0]
        ax_s.plot([r] * (~zero).sum(), s[~zero], "o", color="#2563eb", ms=5)
        dx = 0.15 * (np.arange(zero.sum()) - (zero.sum() - 1) / 2)  # side by side when there are several
        ax_s.plot(r + dx, np.maximum(s[zero], FLOOR), "o", color="#dc2626", ms=7)
        if zero.sum() > 1:
            ax_s.text(r, FLOOR * 1e2, f"{zero.sum()} zeros", ha="center", fontsize=8, color="#dc2626")
    ax_s.set_yscale("log")
    ax_s.set_ylim(FLOOR / 10, 100)
    ax_s.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax_s.set_ylabel("singular values of $M$")
    ax_s.set_title("red: (numerically) zero → null space", fontsize=10)

    # --- (c) the balanced equations ---
    ax_t.axis("off")
    rows = [(name, _latex_equation(nu_r, name)) for name, nu_r in balanced]
    rows += [("peroxide (basis)", _latex_equation(c, "peroxide")) for c in plane]
    for i, (name, eq) in enumerate(rows):
        y = 1 - (i + 0.5) / len(rows)
        ax_t.text(0.02, y, name, fontsize=10, color="#6b7280", va="center", transform=ax_t.transAxes)
        ax_t.text(0.24, y, f"${eq}$", fontsize=12, va="center", transform=ax_t.transAxes)
    ax_t.set_title("whole-number coefficients from the smallest singular vector", fontsize=10)

    fig.tight_layout()
    return save_png(fig, out_dir / "equation_balancing.png")


def _latex_equation(nu: torch.Tensor, name: str) -> str:
    reactants, products = tensors.REACTIONS[name]
    species = reactants + products

    def side(idx):
        terms = [(f"{int(nu[i])}\\," if nu[i] != 1 else "") + tensors.latex(species[i]) for i in idx if nu[i] != 0]
        return " + ".join(terms)

    return f"{side(range(len(reactants)))} \\rightarrow {side(range(len(reactants), len(species)))}"
