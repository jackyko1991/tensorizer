"""The two views with their matches, the RANSAC panorama next to the naive one, and the reprojection errors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import save_png


def figure(imgs, x1, x2, is_out, inliers, pano_r, pano_a, e_ransac, e_all, threshold, corner_err, path: Path):
    imgs, x1, x2 = imgs.cpu().numpy(), x1.cpu().numpy(), x2.cpu().numpy()
    is_out, inliers = is_out.cpu().numpy(), inliers.cpu().numpy()
    h, w = imgs.shape[1:3]
    gap = 8
    pair = np.ones((h, 2 * w + gap, 3))
    pair[:, :w], pair[:, w + gap :] = imgs[0], imgs[1]

    fig = plt.figure(figsize=(10, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, :2])
    ax.imshow(pair, interpolation="nearest")
    for a, b, good in zip(x1, x2, inliers):
        ax.plot(
            [a[0], b[0] + w + gap],
            [a[1], b[1]],
            color="#2563eb" if good else "#dc2626",
            lw=0.6 if good else 0.5,
            alpha=0.75 if good else 0.6,
        )
    ax.scatter(x1[:, 0], x1[:, 1], s=6, c=np.where(inliers, "#1e40af", "#dc2626"), zorder=3)
    ax.scatter(x2[:, 0] + w + gap, x2[:, 1], s=6, c=np.where(inliers, "#1e40af", "#dc2626"), zorder=3)
    ax.set_title(
        f"view 1 | view 2: {inliers.sum()} RANSAC inliers (blue), {(~inliers).sum()} rejected (red)", fontsize=10
    )
    ax.set_axis_off()

    ax_e = fig.add_subplot(grid[0, 2])
    order = np.argsort(e_ransac.cpu().numpy())
    idx = np.arange(len(order))
    er, ea = e_ransac.cpu().numpy()[order], e_all.cpu().numpy()[order]
    ax_e.semilogy(idx, ea, ".", ms=3, color="#9ca3af", label="DLT on all matches")
    ax_e.semilogy(idx[~is_out[order]], er[~is_out[order]], ".", ms=4, color="#2563eb", label="RANSAC: true match")
    ax_e.semilogy(idx[is_out[order]], er[is_out[order]], ".", ms=4, color="#dc2626", label="RANSAC: wrong match")
    ax_e.axhline(threshold, color="#f59e0b", lw=1.2, ls="--", label=f"threshold {threshold:g} px")
    ax_e.set_xlabel("match (sorted by RANSAC error)")
    ax_e.set_ylabel("reprojection error |x₂ − H x₁| (px)")
    ax_e.set_title("reprojection error", fontsize=10)
    ax_e.legend(fontsize=7, loc="upper left")

    for k, (pano, label) in enumerate(
        [
            (pano_r, f"panorama, RANSAC + DLT (corners off by {corner_err[0]:.1f} px)"),
            (pano_a, f"panorama, DLT on all matches ({corner_err[1]:.0f} px)"),
        ]
    ):
        axp = fig.add_subplot(grid[1, 0:2] if k == 0 else grid[1, 2])
        axp.imshow(pano.clamp(0, 1).cpu().numpy(), interpolation="nearest")
        axp.set_title(label, fontsize=10)
        axp.set_axis_off()
    fig.tight_layout()
    return save_png(fig, path)
