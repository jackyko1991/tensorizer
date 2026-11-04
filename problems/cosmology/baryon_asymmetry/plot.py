"""η_B over the (K, ε) plane, the abundances evolving with z = M/T, and the efficiency κ_f(K)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation


def animate(grid, one, kappa_zero, eps_pick, K_pick, eta_obs, path: Path) -> Path:
    logK = np.log10(grid["K"][0].numpy())
    loge = np.log10(grid["eps"][:, 0].numpy())
    eta = grid["eta"].numpy()  # (E, K)
    z = one["z"].numpy()
    n_n, n_eq = one["n_n"][0, 0].numpy(), equation.n_eq(one["z"]).numpy()
    kappa_z = one["n_bl"][0, 0].numpy() / (0.75 * eps_pick)  # running efficiency N_{B−L}/(¾ε)
    eta_z = equation.baryon_to_photon(one["n_bl"][0, 0]).numpy()

    fig = plt.figure(figsize=(9.4, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[:, 0])
    levels = np.arange(-14, -3.5, 1.0)
    cf = ax.contourf(logK, loge, np.log10(np.clip(eta, 1e-14, None)), levels=levels, cmap="Blues")
    fig.colorbar(cf, ax=ax, label=r"$\log_{10}\eta_B$", shrink=0.85)
    ax.contour(logK, loge, eta, levels=[eta_obs], colors=[ACCENT[1]], linewidths=2)
    ax.plot([], [], color=ACCENT[1], lw=2, label=r"observed $\eta_B = 6.1\times10^{-10}$")
    ax.plot(np.log10(K_pick), np.log10(eps_pick), "o", color=ACCENT[4], ms=8, mec="white", label="model shown right")
    ax.axvline(0, color="#6b7280", lw=0.8, ls=":")
    ax.text(-1.9, loge[-1] - 0.3, "weak washout", fontsize=7, color="white")
    ax.text(0.2, loge[-1] - 0.3, "strong washout", fontsize=7, color="white")
    ax.set_xlabel(r"$\log_{10} K$  (decay rate / expansion rate)")
    ax.set_ylabel(r"$\log_{10}\varepsilon$  (CP asymmetry per decay)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"{eta.size} models solved as one batched ODE", fontsize=8.5)

    ax_z = fig.add_subplot(gs[0, 1])
    ax_z.loglog(z, n_eq, color="#6b7280", lw=1, ls="--", label=r"$N_N^{eq}$")
    line_n = ax_z.loglog([], [], color=ACCENT[0], lw=2, label=r"$N_N$ heavy neutrinos")[0]
    line_k = ax_z.loglog([], [], color=ACCENT[4], lw=2, label=r"$N_{B-L}\,/\,(3\varepsilon/4)$")[0]
    cursor = ax_z.axvline(z[0], color="#9ca3af", lw=0.8)
    ax_z.set_xlim(z[0], z[-1])
    ax_z.set_ylim(1e-5, 2)
    ax_z.set_xlabel(r"$z = M/T$ (time →)")
    ax_z.legend(fontsize=6.5, loc="lower left")
    title = ax_z.set_title("", fontsize=8)

    ax_k = fig.add_subplot(gs[1, 1])
    Ks = grid["K"][0].numpy()
    ax_k.loglog(Ks, grid["kappa"].numpy(), color=ACCENT[0], lw=2, label="thermal start (solver)")
    ax_k.loglog(Ks, kappa_zero.numpy(), color=ACCENT[2], lw=1.5, label="zero start (solver)")
    ax_k.loglog(Ks, equation.kappa_strong_washout(grid["K"][0]).numpy(), color="#111827", lw=1, ls="--",
                label="analytic fit (BDP 2005)")  # fmt: skip
    ax_k.axvline(K_pick, color=ACCENT[4], lw=1)
    ax_k.set_xlabel(r"$K$")
    ax_k.set_ylabel(r"efficiency $\kappa_f$")
    ax_k.set_ylim(1e-5, 2)
    ax_k.legend(fontsize=6.5, loc="lower left")

    def update(step: int):
        line_n.set_data(z[: step + 1], n_n[: step + 1])
        k = np.clip(kappa_z[: step + 1], 1e-12, None)
        line_k.set_data(z[: step + 1], k)
        cursor.set_xdata([z[step], z[step]])
        title.set_text(f"ε = {eps_pick:.1e}, K = {K_pick:.3g}: T = M/{z[step]:.2f}, η_B so far = {eta_z[step]:.2e}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(z), 45), path)
