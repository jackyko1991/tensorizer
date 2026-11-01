"""Animate: the gantry turns and the material maps form, then a virtual monoenergetic image sweeps the energy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import tensors

SPIN = 32  # logical steps of phase 1 (gantry rotation)


def animate(
    x: torch.Tensor,  # (N,)
    theta: torch.Tensor,  # (A,)
    e: torch.Tensor,  # (E,) keV
    detected: torch.Tensor,  # (B, E) open-beam spectrum per bin
    mu: torch.Tensor,  # (E, M)
    Q: torch.Tensor,  # (4, A, N, N) per-angle filtered back-projections: water, bone, iodine, conventional
    images: torch.Tensor,  # (4, N, N) final images
    maps: torch.Tensor,  # (3, N, N) true basis maps
    vmi_e: torch.Tensor,  # (K,) keV
    vmi: torch.Tensor,  # (K, N, N) HU
    vmi_rois: torch.Tensor,  # (K, 5) HU in each insert
    labels: list[str],
    doses: list[float],
    errors: list[float],
    path: Path,
) -> Path:
    n_angles = len(theta)
    spin_idx = [round(k * (n_angles - 1) / (SPIN - 1)) for k in range(SPIN)]
    running = torch.cumsum(Q, dim=1)[:, spin_idx] * (np.pi / n_angles)  # (4, SPIN, N, N) partial FBP sums
    running = running.float().cpu().numpy()
    images, vmi, vmi_rois = images.float().cpu().numpy(), vmi.float().cpu().numpy(), vmi_rois.cpu().numpy()
    e, detected, vmi_e = e.cpu().numpy(), detected.cpu().numpy(), vmi_e.cpu().numpy()
    mu_iodine = mu[:, 2].cpu().numpy() * 10.0  # 10 mg/mL iodine, 1/cm
    theta = theta.cpu().numpy()
    i70 = int(np.argmin(np.abs(vmi_e - 70)))
    mu70 = tensors.basis_attenuation(torch.tensor([vmi_e[i70]], dtype=torch.float64))[0].numpy()
    n_vmi = len(vmi_e)
    r = tensors.FOV_HALF_WIDTH
    ext = (-r, r, -r, r)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4))
    (ax_sp, ax_cv, ax_io, ax_bo), (ax_wa, ax_vmi, ax_hu, ax_dose) = axes

    # spectrum seen by each energy bin and the iodine K-edge
    for b in range(detected.shape[0]):
        ax_sp.fill_between(e, detected[b] / detected.max(), color=ACCENT[b], alpha=0.5, lw=0, label=f"bin {b + 1}")
    ax_sp.set_xlabel("photon energy (keV)")
    ax_sp.set_ylabel("counts (open beam)")
    ax_sp.set_title("energy bins")
    ax_sp.plot(e, mu_iodine / mu_iodine.max(), color="#374151", lw=1.2, ls="--", label=r"iodine $\mu$ (scaled)")
    ax_sp.legend(fontsize=7, loc="upper right")

    def image(ax, data, cmap, vmin, vmax, title, unit):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        im = ax.imshow(data, cmap=plt.get_cmap(cmap, 24), vmin=vmin, vmax=vmax, origin="lower", extent=ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=unit)
        ax.set_title(title, fontsize=10)
        return im

    im_cv = image(ax_cv, images[3], "gray", 0.17, 0.30, "energy-integrating CT", r"$\mu_{\rm eff}$ (1/cm)")
    im_io = image(ax_io, images[2], "magma", -2, 16, "iodine map", "mg/mL")
    im_bo = image(ax_bo, 100 * images[1], "bone", -5, 25, "bone map", "% bone")
    im_wa = image(ax_wa, images[0], "Blues_r", 0.0, 1.2, "water map", "fraction")
    im_vmi = image(ax_vmi, vmi[i70], "gray", -150, 900, "", "HU")
    for k, lab in enumerate(labels):  # name the inserts on the conventional image
        phi = np.pi / 2 + 2 * np.pi * k / 5
        ax_cv.text(9.4 * np.cos(phi), 9.4 * np.sin(phi), lab, fontsize=6.5, ha="center", va="center", color=ACCENT[1])
    (gantry,) = ax_cv.plot([], [], color=ACCENT[0], lw=3)

    for k in (0, 1, 2, 3, 4):
        style = dict(color="#374151", lw=2) if k == 4 else dict(color=ACCENT[k], lw=1.3)
        ax_hu.plot(vmi_e, vmi_rois[:, k], label=labels[k], **style)
    marker = ax_hu.axvline(vmi_e[i70], color="#f59e0b", lw=1.5)
    ax_hu.set_xlabel("virtual monoenergetic energy (keV)")
    ax_hu.set_ylabel("insert CT number (HU)")
    ax_hu.set_title("inserts in the VMI")
    ax_hu.legend(fontsize=7, loc="upper right")

    ax_dose.loglog(doses, errors, "o-", color=ACCENT[0], ms=4, label="measured")
    ref = errors[0] * np.sqrt(doses[0] / np.asarray(doses))
    ax_dose.loglog(doses, ref, "--", color="#9ca3af", lw=1, label=r"$\propto 1/\sqrt{N_0}$")
    ax_dose.set_xlabel(r"photons per ray $N_0$")
    ax_dose.set_ylabel("iodine RMS error (mg/mL)")
    ax_dose.set_title("quantification vs dose")
    ax_dose.legend(fontsize=7)
    ax_dose.xaxis.set_minor_locator(NullLocator())
    ax_dose.yaxis.set_minor_locator(NullLocator())
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16) if min(errors) / 1.5 <= t <= max(errors) * 1.5]
    ax_dose.set_yticks(ticks, [f"{t:g}" for t in ticks])

    def update(step: int):
        if step < SPIN:  # phase 1: the gantry turns, the partial back-projections fill in
            frame = running[:, step]
            im_cv.set_data(frame[3])
            im_io.set_data(frame[2])
            im_bo.set_data(100 * frame[1])
            im_wa.set_data(frame[0])
            v = 1000 * (np.einsum("m,myx->yx", mu70, frame[:3]) - mu70[0]) / mu70[0]
            im_vmi.set_data(v)
            k_vmi = i70
            th = theta[spin_idx[step]]
            d, ray = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
            end = -11.5 * ray
            gantry.set_data([end[0] - 9 * d[0], end[0] + 9 * d[0]], [end[1] - 9 * d[1], end[1] + 9 * d[1]])
            ax_cv.set_title(f"energy-integrating CT ({spin_idx[step] + 1}/{n_angles})", fontsize=10)
        else:  # phase 2: sweep the virtual monoenergetic energy
            for im, data in ((im_cv, images[3]), (im_io, images[2]), (im_bo, 100 * images[1]), (im_wa, images[0])):
                im.set_data(data)
            gantry.set_data([], [])
            ax_cv.set_title("energy-integrating CT", fontsize=10)
            k_vmi = min(n_vmi - 1, step - SPIN)
            im_vmi.set_data(vmi[k_vmi])
        marker.set_xdata([vmi_e[k_vmi], vmi_e[k_vmi]])
        ax_vmi.set_title(f"virtual mono {vmi_e[k_vmi]:.0f} keV", fontsize=10)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(SPIN + n_vmi, 60), path, fps=8)
