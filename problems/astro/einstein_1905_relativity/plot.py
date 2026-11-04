"""Light clocks racing along, γ(β) with Frisch & Smith's muons, and binding energies from E = mc²."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation

AMBER, GREY = "#f59e0b", "#9ca3af"
ROW = 1.6  # vertical spacing of the clocks


def animate(beta, t, x, y, nuclei, mu, path: Path) -> Path:
    beta_np, t_np, x_np, y_np = beta.numpy(), t.numpy(), x.numpy(), y.numpy()
    B = len(beta_np)
    gam = equation.gamma(beta).numpy()
    names, Bind, A = (v.numpy() if isinstance(v, torch.Tensor) else v for v in nuclei)

    fig, (ax_c, ax_g, ax_m) = plt.subplots(1, 3, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # --- light clocks
    ax_c.set_xlim(-0.4, t_np[-1] + 0.4)
    ax_c.set_ylim(-0.4, ROW * (B - 1) + 1.5)
    ax_c.set_xlabel("lab position $x$ (mirror spacings, $c = 1$)")
    ax_c.set_yticks([])
    ax_c.grid(False)
    rows = []
    for i in range(B):
        y0 = ROW * i
        rows.append(
            {
                "y0": y0,
                "mirrors": [ax_c.plot([], [], color="#374151", lw=2.5)[0] for _ in range(2)],
                "trail": ax_c.plot([], [], color=AMBER, lw=1.2)[0],
                "photon": ax_c.plot([], [], "o", color=AMBER, ms=5)[0],
                "label": ax_c.text(-0.3, y0 + 1.15, "", fontsize=7.5, color="#374151"),
            }
        )

    # --- γ(β), length contraction and the muons
    one_minus = np.logspace(0, -3.3, 600)  # 1 − β, so speeds near c get room
    b = 1 - one_minus
    ax_g.plot(one_minus, 1 / np.sqrt(1 - b**2), color=ACCENT[0], lw=1.8, label="time dilation $\\gamma$")
    ax_g.plot(one_minus, np.sqrt(1 - b**2), color=ACCENT[2], lw=1.5, label="length $L/L_0 = 1/\\gamma$")
    ax_g.plot(1 - beta_np, gam, "o", color=AMBER, ms=6, zorder=3, label="the clocks")
    beta_mu = float(mu["flight"]["beta"].mean())
    ax_g.errorbar(
        [1 - beta_mu],
        [mu["gamma_fit"]],
        yerr=[mu["gamma_err"]],
        fmt="s",
        color=ACCENT[1],
        ms=6,
        capsize=3,
        label=f"muon counts: {mu['gamma_fit']:.1f} ± {mu['gamma_err']:.1f}",
    )
    ax_g.plot(
        [1 - beta_mu], [mu["gamma_pred"]], "D", color=ACCENT[3], ms=5, label=f"muon speeds: {mu['gamma_pred']:.1f}"
    )
    ax_g.set_yscale("log")
    ax_g.set_xscale("log")
    ax_g.set_xlim(1.0, 5e-4)  # reversed: faster to the right
    ax_g.set_ylim(0.03, 40)
    ax_g.set_xlabel("$1 - v/c$  (faster →)")
    ax_g.set_title(
        "Frisch & Smith 1963: 563/h on Mt Washington,\n408/h at sea level (30/h without dilation)", fontsize=8
    )
    ax_g.legend(fontsize=6.5, loc="upper left")

    # --- E = mc²: binding energy per nucleon from atomic masses
    per = Bind / A
    bars = ax_m.bar(range(len(names)), per, color=ACCENT[0], alpha=0.8)
    he = list(names).index("He-4")
    bars[he].set_color(AMBER)
    ax_m.set_xticks(range(len(names)), names, fontsize=7, rotation=40)
    ax_m.set_ylabel("binding energy per nucleon (MeV)")
    ax_m.set_ylim(0, 10.5)
    ax_m.text(he, per[he] + 0.3, f"{Bind[he]:.2f} MeV\n= 0.0304 u·c²", ha="center", fontsize=7, color="#b45309")
    ax_m.set_title("$E = mc^2$: missing mass $Zm_H + Nm_n - M$\ne⁺e⁻ → 2 × 0.511 MeV photons", fontsize=8)

    footer = fig.text(0.5, 0.01, "", ha="center", fontsize=8.5)

    def update(k: int):
        for i, r in enumerate(rows):
            xc = x_np[i, k]
            r["mirrors"][0].set_data([xc - 0.3, xc + 0.3], [r["y0"], r["y0"]])
            r["mirrors"][1].set_data([xc - 0.3, xc + 0.3], [r["y0"] + 1, r["y0"] + 1])
            r["trail"].set_data(x_np[i, : k + 1], r["y0"] + y_np[i, : k + 1])
            r["photon"].set_data([xc], [r["y0"] + y_np[i, k]])
            n_ticks = int(np.sum((np.diff(y_np[i, : k + 1])[:-1] < 0) & (np.diff(y_np[i, : k + 1])[1:] >= 0)))
            r["label"].set_text(f"β = {beta_np[i]:.2f}   γ = {gam[i]:.2f}   ticks: {n_ticks}")
        ax_c.set_title(f"light clocks, lab time t = {t_np[k]:.1f}: one tick takes γ × 2D/c", fontsize=9)
        footer.set_text(
            "Postulates: same laws in every inertial frame, light always at c  ⇒  the photon's diagonal path makes"
            " moving clocks slow by γ, rods short by 1/γ"
        )

    update(0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    frames = frame_indices(len(t_np), 80) + [len(t_np) - 1] * 10
    return save_gif(fig, update, frames, path, fps=12)
