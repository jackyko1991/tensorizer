"""Animate the structure module: the chain folding into place (3D, rotating) next to the distance maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import kabsch, pair_distances


def animate(
    target: torch.Tensor,
    traj: torch.Tensor,
    d_used: torch.Tensor,
    far: torch.Tensor,
    attn: torch.Tensor | None,
    rms_traj: torch.Tensor,
    rms_starts: torch.Tensor,
    at: torch.Tensor,
    path: Path,
) -> Path:
    """target (N, 3); traj (T, N, 3) snapshots of one start, taken at steps at (T,); d_used, far (N, N);
    attn (N, N) or None; RMSD of the shown start (T,) and of every start (T, B)."""
    tgt = target.cpu().numpy()
    aligned = kabsch(traj, target).cpu().numpy()  # (T, N, 3) every snapshot superposed on the target
    maps = pair_distances(traj).cpu().numpy()  # (T, N, N)
    pred = np.where(far.cpu().numpy(), 22.0, d_used.cpu().numpy())
    rt, rs = rms_traj.cpu().numpy(), rms_starts.cpu().numpy()
    T, N = aligned.shape[:2]
    it = at.cpu().numpy()
    n_steps = int(it[-1])
    cmap = plt.get_cmap("viridis", 11)  # discrete colours keep the GIF small

    fig = plt.figure(figsize=(10.5, 4.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax_p = fig.add_subplot(grid[0, 1])
    ax_r = fig.add_subplot(grid[0, 2])
    ax_a = fig.add_subplot(grid[1, 1])
    ax_c = fig.add_subplot(grid[1, 2])

    centre = tgt.mean(0)
    half = np.abs(tgt - centre).max() + 1
    ax3.plot(*tgt.T, color="#9ca3af", lw=1.2, alpha=0.8, label="target")
    (chain,) = ax3.plot([], [], [], color="#374151", lw=1.5)
    beads = ax3.scatter(*aligned[0].T, c=np.arange(N), cmap=plt.get_cmap("rainbow", 8), s=22, depthshade=False)
    for a, setter in zip(range(3), (ax3.set_xlim, ax3.set_ylim, ax3.set_zlim)):
        setter(centre[a] - half, centre[a] + half)
    ax3.set_box_aspect((1, 1, 1), zoom=1.5)
    ax3.set_axis_off()  # no panes or ticks: fewer colours and faster frames; the rotation shows the depth
    ax3.legend(loc="upper left", fontsize=8)
    title3 = ax3.set_title("")

    for ax, _, title in (
        (ax_p, pred, r"predicted $\hat d_{ij}$ (Å)"),
        (ax_r, maps[0], r"model distances $r_{ij}$ (Å)"),
    ):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=9)
    ax_p.imshow(pred, cmap=cmap, vmin=0, vmax=22)
    im_r = ax_r.imshow(np.minimum(maps[0], 22), cmap=cmap, vmin=0, vmax=22)
    fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.grid(False)
    if attn is not None:
        ax_a.imshow(attn.cpu().numpy(), cmap=plt.get_cmap("Blues", 9))
        ax_a.set_title("attention $A_{ij}$ (row i → j)", fontsize=9)
    else:
        ax_a.text(0.5, 0.5, "refinement off", ha="center", va="center", transform=ax_a.transAxes)

    for b in range(rs.shape[1]):
        ax_c.plot(it, rs[:, b], color="#d1d5db", lw=0.8)
    ax_c.plot(it, rt, color="#93c5fd", lw=1)
    (trace,) = ax_c.plot([], [], color="#2563eb", lw=1.8)
    (dot,) = ax_c.plot([], [], "o", color="#2563eb", ms=4)
    ax_c.set_yscale("log")
    ax_c.set_yticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])  # plain labels: no mathtext per frame
    ax_c.minorticks_off()
    ax_c.set_xlim(0, n_steps)
    ax_c.set_xlabel("gradient step", fontsize=8, labelpad=1)
    ax_c.set_ylabel("RMSD (Å)", fontsize=8)
    ax_c.set_title("all starts (grey), shown (blue)", fontsize=9)
    ax_c.tick_params(labelsize=7)

    def update(t: int):
        x = aligned[t]
        chain.set_data_3d(x[:, 0], x[:, 1], x[:, 2])
        beads._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        ax3.view_init(elev=18, azim=-60 + 200 * t / max(T - 1, 1))
        im_r.set_data(np.minimum(maps[t], 22))
        trace.set_data(it[: t + 1], rt[: t + 1])
        dot.set_data(it[t : t + 1], rt[t : t + 1])
        title3.set_text(f"step {it[t]:.0f}: RMSD {rt[t]:.2f} Å")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(T, 72), path, fps=8)
