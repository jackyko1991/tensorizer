"""Animate training: loss curves, predicted vs true π energy for the GNN and the MLP, and one large test molecule."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, GREY, DARK = "#2563eb", "#dc2626", "#9ca3af", "#374151"


def animate(h_gnn: dict, h_mlp: dict, y: dict, var: float, ex: dict, path: Path) -> Path:
    y_s, y_l = y["small"].cpu().numpy(), y["large"].cpu().numpy()
    g_s, g_l = h_gnn["small"].numpy(), h_gnn["large"].numpy()  # (E+1, M)
    m_s, m_l = h_mlp["small"].numpy(), h_mlp["large"].numpy()
    loss_g, loss_m = h_gnn["loss"].numpy(), h_mlp["loss"].numpy()
    big_g = ((g_l - y_l) ** 2).mean(1)  # test MSE on the larger molecules, per epoch
    big_m = ((m_l - y_l) ** 2).mean(1)
    epochs = np.arange(len(loss_g))
    lo, hi = 0.0, max(y_l.max(), y_s.max()) * 1.08

    fig = plt.figure(figsize=(8.4, 4.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1])
    ax_e = fig.add_subplot(grid[0, 0])
    ax_m = fig.add_subplot(grid[1, 0])
    ax_g = fig.add_subplot(grid[:, 1])
    ax_p = fig.add_subplot(grid[:, 2])
    for ax in (ax_g, ax_p):
        ax.set_aspect("equal")

    # --- loss curves ---
    ax_e.set_yscale("log")
    ax_e.minorticks_off()
    ax_e.set_xlim(0, epochs[-1])
    ax_e.set_ylim(min(loss_g.min(), loss_m.min()) * 0.5, max(big_m[-1], var) * 60)  # legend room; clips epoch 0
    ax_e.axhline(var, color=GREY, ls="--", lw=1)
    ax_e.text(epochs[-1], var * 1.15, "predict the mean", ha="right", va="bottom", fontsize=6.5, color="#6b7280")
    lines = {
        "gt": ax_e.plot([], [], color=BLUE, lw=1.5, label="GNN train")[0],
        "gl": ax_e.plot([], [], color=BLUE, lw=1.2, ls=":", label="GNN larger mol.")[0],
        "mt": ax_e.plot([], [], color=RED, lw=1.5, label="MLP train")[0],
        "ml": ax_e.plot([], [], color=RED, lw=1.2, ls=":", label="MLP larger mol.")[0],
    }
    ax_e.set_title("mean squared error", fontsize=9)
    ax_e.tick_params(labelsize=7)
    ax_e.legend(fontsize=6, loc="upper right", ncol=2)

    # --- predicted vs true ---
    scat = {}
    for ax, name in ((ax_g, "message-passing GNN"), (ax_p, "MLP on adjacency")):
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("true π energy  y / |β|", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        scat[ax] = (
            ax.plot([], [], "o", color=BLUE, ms=2.2, alpha=0.6, label="4–10 C (as trained)")[0],
            ax.plot([], [], "o", color=RED, ms=2.2, alpha=0.6, label="11–16 C (larger)")[0],
        )
    ax_g.set_ylabel("predicted", fontsize=8)
    ax_g.legend(fontsize=6.5, loc="upper left", markerscale=2)
    mae_txt = {ax: ax.text(0.97, 0.03, "", transform=ax.transAxes, ha="right", fontsize=7) for ax in (ax_g, ax_p)}

    # --- one large test molecule, atoms coloured by their learned contribution e_i ---
    _draw_molecule(ax_m, ex)
    counter = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=9)

    def update(k: int):
        lines["gt"].set_data(epochs[: k + 1], loss_g[: k + 1])
        lines["gl"].set_data(epochs[: k + 1], big_g[: k + 1])
        lines["mt"].set_data(epochs[: k + 1], loss_m[: k + 1])
        lines["ml"].set_data(epochs[: k + 1], big_m[: k + 1])
        for ax, ps, pl in ((ax_g, g_s, g_l), (ax_p, m_s, m_l)):
            scat[ax][0].set_data(y_s, ps[k])
            scat[ax][1].set_data(y_l, pl[k])
            mae_txt[ax].set_text(f"MAE {np.abs(ps[k] - y_s).mean():.2f} (blue), {np.abs(pl[k] - y_l).mean():.2f} (red)")
        counter.set_text(f"epoch {k}   ·   GNN training MSE {loss_g[k]:.3f}   ·   MLP {loss_m[k]:.3f}")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, frame_indices(len(epochs), 60), path, fps=8)


def _draw_molecule(ax, ex: dict) -> None:
    mask = ex["mask"].cpu().numpy() > 0
    A = ex["A"].cpu().numpy()[np.ix_(mask, mask)]
    e = ex["e"].cpu().numpy()[mask]
    xy = _layout(A)
    for i, j in zip(*np.nonzero(np.triu(A))):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=DARK, lw=1.5, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=e, cmap="Blues", s=60, edgecolors="#1e40af", linewidths=0.8, zorder=2)
    cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("$e_i$", fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{mask.sum()} C: true {ex['y']:.2f}, GNN {ex['pred']:.2f}", fontsize=8)
    ax.text(0.5, -0.08, f"relabel atoms: GNN Δ ≤ {ex['gnn_perm']:.0e}, MLP Δ = {ex['mlp_perm']:.1f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5)  # fmt: skip


def _layout(A: np.ndarray) -> np.ndarray:
    """2D coordinates from graph distances by classical multidimensional scaling (bonded atoms end up close)."""
    n = len(A)
    D = np.where(A > 0, 1.0, np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(n):  # Floyd–Warshall shortest paths
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    J = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * J @ (D**2) @ J)
    return vecs[:, -2:] * np.sqrt(np.maximum(vals[-2:], 1e-9))
