"""A slice of the plaquette field, ⟨plaquette⟩(β) against strong/weak coupling, the Wilson-loop area law and the
glueball effective mass, as the Monte Carlo statistics accumulate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import strong_coupling_u, weak_coupling


def animate(group, dim, betas, i0, out, plaq_err, u, running, n_therm, path: Path) -> Path:
    b = betas.cpu().numpy()
    snaps = out["snapshots"][:, i0].cpu().numpy()  # (K, L, L)
    sweeps = out["snapshot_sweeps"]
    rp, rw = running["plaquette"].cpu().numpy(), running["wilson"].cpu().numpy()
    rm, re = running["mass"].cpu().numpy(), running["mass_err"].cpu().numpy()
    perr = plaq_err.cpu().numpy()
    u0 = float(u[i0])

    fig, (ax_f, ax_p, ax_w, ax_m) = plt.subplots(
        1, 4, figsize=(12.5, 3.5), gridspec_kw={"width_ratios": [1, 1.2, 1.1, 1.1]}
    )
    img = ax_f.imshow(snaps[0].T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_f.grid(False)
    ax_f.set_xlabel("$x_0$ (time)")
    ax_f.set_ylabel("$x_1$")
    fig.colorbar(img, ax=ax_f, fraction=0.046, pad=0.03)
    f_title = ax_f.set_title("", fontsize=9)

    fine = torch.linspace(0.05, float(b.max()) * 1.05, 200, dtype=torch.float64)
    ax_p.plot(
        fine, strong_coupling_u(fine, group), color="#9ca3af", ls="--", lw=1.2, label="strong coupling $u(\\beta)$"
    )
    weak = fine[weak_coupling(fine, group, dim) > 0.35]
    ax_p.plot(weak, weak_coupling(weak, group, dim), color="#9ca3af", ls=":", lw=1.4, label=r"weak $1 - n/(D\beta)$")
    pts = ax_p.errorbar(b, rp[-1], yerr=perr, fmt="o", ms=4, color=ACCENT[0], label="Monte Carlo")
    ax_p.set_ylim(0, 1)
    ax_p.set_xlabel(r"$\beta$")
    ax_p.set_title(r"$\langle\frac{1}{2}\mathrm{Tr}\,P\rangle$" + f", {group}, D = {dim}", fontsize=9)
    ax_p.legend(fontsize=7, loc="lower right")

    r_max = rw.shape[-1]
    area = np.array([[r * t for t in range(1, r_max + 1)] for r in range(1, r_max + 1)], dtype=float)
    a_line = np.linspace(0, area.max() + 0.5, 50)
    ax_w.plot(a_line, -np.log(u0) * a_line, color="#9ca3af", ls="--", lw=1.2, label=r"area law $-RT\ln u$")
    scat = ax_w.scatter(area.ravel(), np.zeros(area.size), s=22, color=ACCENT[1], zorder=3, label="$-\\ln W(R,T)$")
    ax_w.set_xlabel("area $RT$")
    ax_w.legend(fontsize=7, loc="upper left")
    ax_w.set_title(rf"Wilson loops, $\beta = {b[i0]:g}$", fontsize=9)
    ax_w.set_ylim(0, -np.log(u0) * area.max() * 1.15)

    n_delta = rm.shape[-1]
    lead = -4 * np.log(u0)
    ax_m.axhline(lead, color="#9ca3af", ls="--", lw=1.2, label=r"strong coupling $-4\ln u$")
    m_bars = [ax_m.errorbar([], [], yerr=[], fmt="s", color=ACCENT[2], ms=5, label=r"$a\,m_\mathrm{eff}(\Delta)$")]
    ax_m.set_xlim(-0.5, n_delta - 0.5)
    ax_m.set_xticks(range(n_delta))
    ax_m.set_ylim(0, lead * 1.8 + 0.5)
    ax_m.set_xlabel(r"$\Delta$ (time slices)")
    ax_m.legend(fontsize=7, loc="upper right")
    if out["operator"] is None:
        ax_m.text(
            0.5,
            0.5,
            "no glueball operator in 2D:\nno propagating gluons",
            ha="center",
            transform=ax_m.transAxes,
            fontsize=8,
        )
    m_title = ax_m.set_title("", fontsize=9)

    def update(f: int):
        img.set_data(snaps[f].T)
        phase = "thermalising" if sweeps[f] < n_therm else "measuring"
        f_title.set_text(rf"$\frac{{1}}{{2}}$Tr$P_{{01}}$, sweep {sweeps[f] + 1} ({phase})")
        y = rp[f]
        for line in pts.lines[2]:
            line.set_alpha(1.0 if f == len(sweeps) - 1 else 0.0)  # error bars once all data are in
        pts.lines[0].set_data(b, y)
        w = rw[f]
        loops = np.where(w.ravel() > 3e-3, w.ravel(), np.nan)  # loops below ~3e-3 are lost in the noise
        scat.set_offsets(np.c_[area.ravel(), -np.log(loops)])
        m_bars[0].remove()
        m_bars[0] = ax_m.errorbar(np.arange(n_delta), rm[f], yerr=re[f], fmt="s", color=ACCENT[2], ms=5, capsize=3)
        ok = np.isfinite(rm[f, 0])
        m_title.set_text(
            rf"glueball: $a\,m_\mathrm{{eff}}(0) = {rm[f, 0]:.2f} \pm {re[f, 0]:.2f}$"
            if ok
            else "glueball correlator (collecting)"
        )

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(sweeps), len(sweeps)), path, fps=8)
