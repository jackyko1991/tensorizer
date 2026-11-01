"""Animate photon pairs reaching the ring, the sinogram filling up, then MLEM sharpening iteration by iteration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from problems._common.plotting import frame_indices, save_gif

from . import tensors

FILL_STEPS = 40  # animation steps spent collecting data before the reconstruction starts
N_SHOWN = 160  # coincidences drawn as lines on the ring


def animate(
    events: dict,
    kept: torch.Tensor,
    y: torch.Tensor,
    activity: torch.Tensor,
    images: torch.Tensor,
    recon_fbp: torch.Tensor,
    err: torch.Tensor,
    err_fbp: float,
    n: int,
    path: Path,
) -> Path:
    K, L, s_max = tensors.sinogram_bins(n)
    n_iter = images.shape[0] - 1
    half, R = tensors.WIDTH / 2, tensors.RING_RADIUS
    # events shown on the ring: the first N_SHOWN detected coincidences
    idx = torch.nonzero(kept).flatten()
    shown = idx[:N_SHOWN]
    a = events["annihilation"][shown].cpu().numpy()  # (S, 2)
    hits = events["crystals"][shown].cpu().numpy()  # (S, 2, 2)
    # cumulative sinograms while the data come in (detected events in the order they were simulated)
    bins = events["bin"][idx]
    background = (y - torch.bincount(bins, minlength=K * L).float()).clamp(min=0)
    cuts = [round(len(bins) * (i + 1) / FILL_STEPS) for i in range(FILL_STEPS)]
    partial = [
        (torch.bincount(bins[:c], minlength=K * L).float() + background * c / len(bins)).view(K, L).T.cpu().numpy()
        for c in cuts
    ]
    imgs = images.view(-1, n, n).cpu().numpy()
    fbp_img = recon_fbp.view(n, n).cpu().numpy()
    truth = (activity / activity.sum() * images[0].sum()).cpu().numpy()  # true image on the same count scale
    vmax = float(truth.max()) * 1.1
    cmap = plt.get_cmap("inferno", 16)
    img_kw = {"origin": "lower", "cmap": cmap, "vmin": 0, "vmax": vmax, "extent": [-half, half, -half, half]}

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8))
    (ax_ring, ax_sino, ax_true), (ax_fbp, ax_mlem, ax_err) = axes
    for ax in (ax_ring, ax_true, ax_fbp, ax_mlem):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- ring with decays and lines of response ---------------------------------------------------------
    body = np.ma.masked_equal(activity.cpu().numpy(), 0)  # transparent outside the body
    ax_ring.imshow(body, alpha=0.35, **{**img_kw, "vmax": float(activity.max())})
    ring = np.linspace(0, 2 * np.pi, 200)
    ax_ring.plot(R * np.cos(ring), R * np.sin(ring), color="#374151", lw=4)
    ax_ring.set_xlim(-R * 1.08, R * 1.08)
    ax_ring.set_ylim(-R * 1.08, R * 1.08)
    old = LineCollection([], colors="#2563eb", linewidths=0.5, alpha=0.35)
    new = LineCollection([], colors="#dc2626", linewidths=1.4)
    ax_ring.add_collection(old)
    ax_ring.add_collection(new)
    flash = ax_ring.plot([], [], "o", color="#dc2626", ms=4)[0]
    origin = ax_ring.plot([], [], "*", color="#f59e0b", ms=8)[0]
    ring_title = ax_ring.set_title("coincidences: 0")

    # --- sinogram ---------------------------------------------------------------------------------------
    sino = ax_sino.imshow(
        partial[0], origin="lower", aspect="auto", cmap=plt.get_cmap("viridis", 16), vmin=0,
        vmax=float(partial[-1].max()), extent=[0, 180, -s_max, s_max],
    )  # fmt: skip
    ax_sino.set_xlabel("θ (°)")
    ax_sino.set_ylabel("s (cm)")
    ax_sino.grid(False)
    sino_title = ax_sino.set_title("sinogram (LOR counts)")

    ax_true.imshow(truth, **img_kw)
    ax_true.set_title("true activity")
    fbp_im = ax_fbp.imshow(np.zeros_like(fbp_img), **img_kw)
    fbp_title = ax_fbp.set_title("FBP (waiting for data)")
    mlem_im = ax_mlem.imshow(imgs[0], **img_kw)
    mlem_title = ax_mlem.set_title("MLEM, iteration 0")

    # --- convergence --------------------------------------------------------------------------------------
    e = err.cpu().numpy()
    ax_err.set_xlim(0, max(n_iter, 1))
    ax_err.set_ylim(0, max(float(e.max()), err_fbp) * 1.1)
    ax_err.axhline(err_fbp, ls="--", color="#16a34a", lw=1.2, label="FBP")
    err_line = ax_err.plot([], [], color="#dc2626", lw=1.8, label="MLEM")[0]
    ax_err.set_xlabel("iteration")
    ax_err.set_title("error vs. true activity")
    ax_err.legend(fontsize=8)

    def update(step: int):
        if step < FILL_STEPS:
            m = round(N_SHOWN * (step + 1) / FILL_STEPS)
            m_old = round(N_SHOWN * step / FILL_STEPS)
            old.set_segments(list(hits[:m_old]))
            # the photon pairs that just arrived: two back-to-back paths from the annihilation point to the ring
            new.set_segments([np.stack([a[e], h]) for e in range(m_old, m) for h in hits[e]])
            flash.set_data(hits[m_old:m, :, 0].ravel(), hits[m_old:m, :, 1].ravel())
            origin.set_data(a[m_old:m, 0], a[m_old:m, 1])
            sino.set_data(partial[step])
            ring_title.set_text(f"coincidences: {cuts[step]:,}")
            return
        # all data are in: final ring picture and sinogram, FBP appears, MLEM iterates
        k = step - FILL_STEPS + 1
        old.set_segments(list(hits))
        new.set_segments([])
        flash.set_data([], [])
        origin.set_data([], [])
        sino.set_data(partial[-1])
        ring_title.set_text(f"coincidences: {cuts[-1]:,}")
        sino_title.set_text("sinogram (complete)")
        fbp_im.set_data(fbp_img)
        fbp_title.set_text("FBP (ramp × Hann)")
        mlem_im.set_data(imgs[k])
        mlem_title.set_text(f"MLEM, iteration {k}")
        err_line.set_data(np.arange(k + 1), e[: k + 1])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(FILL_STEPS + n_iter, 100), path, fps=10)
