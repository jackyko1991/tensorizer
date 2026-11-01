"""Two panels: B/A along the valley of stability, and a heatmap of B/A over the (N, Z) chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm

from problems._common.plotting import save_png


def figure(
    z: torch.Tensor,
    n: torch.Tensor,
    b_per_a: torch.Tensor,
    mass: torch.Tensor,
    best: torch.Tensor,
    z_best: torch.Tensor,
    z_line: torch.Tensor,
    measured: list,
    path: Path,
) -> Path:
    fig, (ax_curve, ax_map) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})

    a_np, best_np = mass.cpu().numpy(), best.cpu().numpy()
    ax_curve.plot(a_np, best_np, color="#2563eb", lw=2, label="liquid-drop formula")
    offsets = {"⁵⁶Fe": (-22, 6), "⁶²Ni": (2, 6)}  # keep the two neighbouring labels apart
    for name, _z, a, ba in measured:
        ax_curve.plot(a, ba, "o", color="#dc2626", ms=5)
        offset = offsets.get(name, (4, -11))
        ax_curve.annotate(name, (a, ba), textcoords="offset points", xytext=offset, fontsize=8, color="#b91c1c")
    k = int(np.argmax(best_np))
    ax_curve.axvline(a_np[k], color="#f59e0b", lw=1, ls="--")
    ax_curve.text(a_np[k] + 4, 3.0, f"peak A = {a_np[k]:.0f}", color="#b45309", fontsize=9)
    ax_curve.plot([], [], "o", color="#dc2626", ms=5, label="measured")
    ax_curve.set_xlim(0, a_np.max())
    ax_curve.set_ylim(0, 9.5)
    ax_curve.set_xlabel("mass number A")
    ax_curve.set_ylabel("binding energy per nucleon B/A (MeV)")
    ax_curve.legend(loc="lower right", fontsize=8)

    ba = b_per_a.cpu().numpy()
    ba = np.where(ba > 0, ba, np.nan)  # unbound cells left blank
    levels = np.array([0.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0])  # finer bins near the top
    cmap = plt.get_cmap("viridis", len(levels) - 1)
    image = ax_map.imshow(
        ba,
        origin="lower",
        extent=(n.min().item() - 0.5, n.max().item() + 0.5, z.min().item() - 0.5, z.max().item() + 0.5),
        cmap=cmap,
        norm=BoundaryNorm(levels, cmap.N, clip=True),
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax_map, label="B/A (MeV)")
    z_np = z_best.cpu().numpy()
    ax_map.plot(a_np - z_np, z_np, color="#dc2626", lw=1.2, label="most bound isobar")
    z_line_np = z_line.cpu().numpy()
    ax_map.plot(a_np - z_line_np, z_line_np, color="white", lw=1, ls="--", label=r"$Z_0(A)$ formula")
    ax_map.plot([0, n.max().item()], [0, n.max().item()], color="#9ca3af", lw=1, ls=":", label="N = Z")
    ax_map.set_xlim(0, n.max().item())
    ax_map.set_ylim(0, z.max().item())
    ax_map.set_xlabel("neutrons N")
    ax_map.set_ylabel("protons Z")
    ax_map.grid(False)
    ax_map.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return save_png(fig, path)
