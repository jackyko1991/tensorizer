"""Animate a dilution sweep of the Zn²⁺ half-cell: EMF, activity coefficients, Debye length, the ion's cloud."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import equation


def animate(
    log_Q, T_batch, E_tq, m_zn, m_cu, E_ideal, E_lim, E_ext, I_zn, lg_pb, A, B, a, z, T, r, phi, kap, volts, path: Path
) -> Path:
    lq, Tb, E_tq = log_Q.cpu().numpy(), T_batch.cpu().numpy() - 273.15, E_tq.cpu().numpy()
    m, ion, lg_pb = m_zn.cpu().numpy(), I_zn.cpu().numpy(), lg_pb.cpu().numpy()
    x_ratio = np.log10(m / m_cu.item())
    dE_lim = 1e3 * (E_lim - E_ideal).cpu().numpy()  # mV
    dE_ext = 1e3 * (E_ext - E_ideal).cpu().numpy()
    E_ext_np = E_ext.cpu().numpy()
    r_np, phi_mV, kinv = r.cpu().numpy(), 1e3 * volts.item() * phi.cpu().numpy(), 1 / kap.cpu().numpy()
    V = 1e3 * volts.item()  # mV per unit of φ̃
    C = len(m)

    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 3)
    ax_n, ax_d, ax_g = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_k, ax_p = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])

    # --- (a) Nernst: E against log Q for a batch of temperatures ---
    cmap = plt.get_cmap("coolwarm")
    for t in range(len(Tb)):
        ax_n.plot(lq, E_tq[t], color=cmap(t / (len(Tb) - 1)), lw=1.3, label=f"{Tb[t]:.0f} °C")
    ax_n.plot(x_ratio, E_ext_np, "k--", lw=1.2, label=r"with γ, x = log$_{10}$(m$_{Zn}$/m$_{Cu}$)")
    (mark_n,) = ax_n.plot([], [], "o", color="#f59e0b", ms=7, zorder=5)
    ax_n.set_xlabel(r"$\log_{10}\,Q = \log_{10}(a_{Zn}/a_{Cu})$", labelpad=1)
    ax_n.set_ylabel("EMF (V)")
    ax_n.set_title("Nernst: slope 2.303RT/nF", fontsize=10)
    ax_n.legend(fontsize=6.5, loc="lower left", ncol=2)

    # --- (b) how much the activities change the EMF ---
    ax_d.semilogx(m, dE_lim, ":", color="#dc2626", lw=1.5, label="limiting law")
    ax_d.semilogx(m, dE_ext, "-", color="#2563eb", lw=1.5, label="extended law")
    ax_d.axhline(0, color="#9ca3af", lw=1)
    (mark_d,) = ax_d.plot([], [], "o", color="#f59e0b", ms=7, zorder=5)
    ax_d.set_xlabel("m(Zn²⁺)  (mol/kg)", labelpad=1)
    ax_d.set_ylabel(r"$E - E_{ideal}$ (mV)")
    ax_d.set_title(f"activity correction, m(Cu²⁺) = {m_cu.item():g}", fontsize=10)
    ax_d.legend(fontsize=7)

    # --- (c) activity coefficient of Zn²⁺ against √I ---
    s = np.linspace(0, np.sqrt(ion.max()) * 1.05, 100)
    I_s = torch.tensor(s**2, dtype=A.dtype)
    ax_g.plot(
        s, 10 ** equation.log10_gamma_limiting(z, I_s, A.cpu()).numpy(), ":", color="#dc2626", lw=1.5, label="limiting"
    )
    ax_g.plot(
        s,
        10 ** equation.log10_gamma_extended(z, I_s, A.cpu(), B.cpu(), a).numpy(),
        color="#2563eb",
        lw=1.5,
        label="extended",
    )
    (dots_g,) = ax_g.plot([], [], "o", ms=3, mfc="none", color="#16a34a", label="from PB cloud")
    (mark_g,) = ax_g.plot([], [], "o", color="#f59e0b", ms=7, zorder=5)
    ax_g.set_ylim(0, 1.05)
    ax_g.set_xlabel(r"$\sqrt{I}$  (mol/kg)$^{1/2}$", labelpad=1)
    ax_g.set_ylabel(r"$\gamma(\mathrm{Zn^{2+}})$")
    ax_g.set_title("Debye–Hückel", fontsize=10)
    ax_g.legend(fontsize=7, loc="lower left")

    # --- (d) Debye length against concentration ---
    l_11 = 1 / equation.kappa(torch.tensor(m), T.cpu()).numpy()  # a 1:1 salt, I = m
    ax_k.loglog(m, l_11, color="#9ca3af", lw=1.3, label="1:1 salt (NaCl)")
    ax_k.loglog(m, kinv, color="#2563eb", lw=1.5, label="2:2 salt (ZnSO₄)")
    (mark_k,) = ax_k.plot([], [], "o", color="#f59e0b", ms=7, zorder=5)
    ax_k.set_xlabel("molality (mol/kg)", labelpad=1)
    ax_k.set_ylabel(r"Debye length $\kappa^{-1}$ (nm)")
    ax_k.legend(fontsize=7)

    # --- (e) the potential around one Zn²⁺ ion ---
    r_view = np.linspace(a, 6.0, 300)
    ax_p.plot(r_view, V / r_view, ":", color="#6b7280", lw=1.3, label="bare Coulomb  q/4πεr")
    (line_y,) = ax_p.plot([], [], color="#2563eb", lw=1.6, label=r"Yukawa $e^{-\kappa r}/r$ (analytic)")
    (dots_p,) = ax_p.plot([], [], "o", ms=3.5, mfc="none", color="#dc2626", label="linearised PB (linalg.solve)")
    line_k = ax_p.axvline(a, color="#f59e0b", lw=1.5, ls="--")
    text_k = ax_p.text(0, 0.9, "", color="#b45309", fontsize=9, transform=ax_p.get_xaxis_transform())
    ax_p.axvspan(0, a, color="#93c5fd", alpha=0.5)
    ax_p.text(
        a / 2, 0.5, "ion", ha="center", rotation=90, fontsize=8, color="#1e40af", transform=ax_p.get_xaxis_transform()
    )
    ax_p.set_xlim(0, 6.0)
    ax_p.set_ylim(0, V / a * 1.05)
    ax_p.set_xlabel("distance from the ion centre r (nm)", labelpad=1)
    ax_p.set_ylabel("φ (mV)")
    ax_p.legend(fontsize=7.5, loc="upper right")
    title = ax_p.set_title("screened potential around Zn²⁺", fontsize=10)

    def update(c: int):
        mark_n.set_data([x_ratio[c]], [E_ext_np[c]])
        mark_d.set_data([m[c]], [dE_ext[c]])
        dots_g.set_data(np.sqrt(ion[: c + 1]), 10 ** lg_pb[: c + 1])
        mark_g.set_data([np.sqrt(ion[c])], [10 ** lg_pb[c]])
        mark_k.set_data([m[c]], [kinv[c]])
        k = 1 / kinv[c]
        line_y.set_data(r_view, V * np.exp(-k * (r_view - a)) / ((1 + k * a) * r_view))
        keep = r_np[c] <= 6.0
        stride = max(1, keep.sum() // 30)
        dots_p.set_data(r_np[c][keep][::stride], phi_mV[c][keep][::stride])
        x_k = min(a + kinv[c], 5.95)
        line_k.set_xdata([x_k, x_k])
        text_k.set_position((x_k + 0.05, 0.55))
        text_k.set_text(f"a + κ⁻¹ = {a + kinv[c]:.2f} nm")
        title.set_text(f"screened potential around Zn²⁺   ·   m = {m[c]:.1e} mol/kg, I = {ion[c]:.3g}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(C, 45), path, fps=8)
