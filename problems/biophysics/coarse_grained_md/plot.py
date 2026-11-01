"""Animate the runs: chains at three temperatures, Q(t) for every temperature, and the melting curve building up."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .solve import melting_curve

SPACING = 26.0  # Å between the three chains drawn side by side


def animate(
    native: torch.Tensor,
    times: torch.Tensor,
    Q: torch.Tensor,
    frame_t: torch.Tensor,
    frames: torch.Tensor,
    T_b: torch.Tensor,
    n_rep: int,
    Tm: float,
    path: Path,
) -> Path:
    """native (N, 3); Q (K, S) at times (K,); frames (F, S, N, 3) at frame_t (F,); S = B·R systems."""
    nat = native.cpu().numpy()
    nat = nat - nat.mean(0)
    t, q = times.cpu().numpy(), Q.cpu().numpy()
    ft, fr = frame_t.cpu().numpy(), frames.cpu().numpy()
    temps = T_b.cpu().numpy()
    B, half = len(temps), n_rep // 2
    colours = plt.get_cmap("coolwarm")(np.linspace(0, 1, B))
    q_br = q.reshape(len(t), B, n_rep)
    q_nat = _smooth(q_br[:, :, :half].mean(2))  # (K, B) started folded
    q_unf = _smooth(q_br[:, :, half:].mean(2))  # (K, B) started unfolded
    final_nat, final_unf, _ = (c.cpu().numpy() for c in melting_curve(times, Q, B, n_rep))

    # three chains that started unfolded: coldest, nearest T_m, hottest
    near_tm = int(np.nanargmin(np.abs(temps - Tm))) if np.isfinite(Tm) else B // 2
    shown = sorted({0, near_tm, B - 1})
    systems = [b * n_rep + half for b in shown]

    fig = plt.figure(figsize=(10.5, 4.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    ax3 = fig.add_subplot(grid[0], projection="3d")
    ax_q = fig.add_subplot(grid[1])
    ax_m = fig.add_subplot(grid[2])

    offsets = [np.array([(k - (len(shown) - 1) / 2) * SPACING, 0.0, 0.0]) for k in range(len(shown))]
    chains = []
    for k, b in enumerate(shown):
        ax3.plot(*(nat + offsets[k]).T, color="#d1d5db", lw=1)
        chains.append(ax3.plot([], [], [], "o-", color=colours[b], ms=3, lw=1.5)[0])
        ax3.text(*(offsets[k] + [0, 0, 15]), f"T = {temps[b]:.2f}", ha="center", fontsize=8)
    lim = SPACING * len(shown) / 2
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-13, 13)
    ax3.set_zlim(-13, 13)
    ax3.set_box_aspect((2 * lim, 26, 26), zoom=1.5)
    ax3.set_axis_off()
    title3 = ax3.set_title("", fontsize=10)

    ax_q.set_xlim(0, t[-1])
    ax_q.set_ylim(0, 1.02)
    ax_q.set_xlabel("time (τ)")
    ax_q.set_ylabel("fraction of native contacts Q")
    ax_q.set_title("solid: started folded, dashed: unfolded", fontsize=9)
    lines_n = [ax_q.plot([], [], color=colours[b], lw=1.2)[0] for b in range(B)]
    lines_u = [ax_q.plot([], [], "--", color=colours[b], lw=1.2)[0] for b in range(B)]

    ax_m.plot(temps, final_nat, color="#9ca3af", lw=1)
    ax_m.plot(temps, final_unf, "--", color="#9ca3af", lw=1)
    if np.isfinite(Tm):
        ax_m.axvline(Tm, color="#f59e0b", lw=1, ls=":")
        ax_m.text(Tm, 0.04, f" Tm ≈ {Tm:.2f}", color="#b45309", fontsize=8)
    pts_n = ax_m.scatter(temps, np.zeros(B), c=colours, s=28, edgecolors="#374151", zorder=3)
    pts_u = ax_m.scatter(temps, np.zeros(B), c=colours, s=28, marker="s", edgecolors="#374151", zorder=3)
    ax_m.set_xlim(temps[0] - 0.05, temps[-1] + 0.05)
    ax_m.set_ylim(0, 1.02)
    ax_m.set_xlabel("temperature T (ε / k_B)")
    ax_m.set_ylabel("⟨Q⟩, second half of the run so far")
    ax_m.set_title("melting curve (● folded, ■ unfolded start)", fontsize=9)

    native_t = torch.as_tensor(nat)

    def update(f: int):
        now = ft[f]
        k = int(np.searchsorted(t, now, side="right"))
        for b in range(B):
            lines_n[b].set_data(t[:k], q_nat[:k, b])
            lines_u[b].set_data(t[:k], q_unf[:k, b])
        for c, s in enumerate(systems):
            x = _superpose(fr[f, s], native_t) + offsets[c]
            chains[c].set_data_3d(x[:, 0], x[:, 1], x[:, 2])
        if now > 0:
            m_n, m_u, _ = melting_curve(times, Q, B, n_rep, float(now))
            pts_n.set_offsets(np.c_[temps, m_n.cpu().numpy()])
            pts_u.set_offsets(np.c_[temps, m_u.cpu().numpy()])
        ax3.view_init(elev=15, azim=-80 + 40 * f / max(len(ft) - 1, 1))
        title3.set_text(f"chains started unfolded, t = {now:.0f} τ")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(ft), 72), path, fps=8)


def _smooth(y: np.ndarray, width: int = 15) -> np.ndarray:
    """Running mean along time (axis 0), so the thermal jitter of Q does not hide the trend."""
    kernel = np.ones(width) / width
    pad = np.pad(y, ((width // 2, width - 1 - width // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, b], kernel, mode="valid") for b in range(y.shape[1])], 1)


def _superpose(x: np.ndarray, ref: torch.Tensor) -> np.ndarray:
    """Kabsch: rotate the centred chain x (N, 3) onto the centred reference, for display only."""
    P = torch.as_tensor(x - x.mean(0))
    H = P.T @ ref  # (3, 3) = Σ_i p_ia q_ib
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.T @ U.T))
    R = Vh.T @ torch.diag(torch.tensor([1.0, 1.0, d.item()], dtype=P.dtype)) @ U.T
    return (P @ R.T).numpy()
