"""Four panels: tachyon signals in a moving frame, the antitelephone loop, neutrino-speed data and condensation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, GREY, AMBER, GREEN = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b", "#16a34a"


def animate(speeds, betas, seen, u0, loop, data, mu, field, v, params, path: Path) -> Path:
    speeds, betas, seen = speeds.cpu().numpy(), betas.cpu().numpy(), seen.cpu().numpy()  # (U,), (F,), (F, U, 2)
    t_rec, x_rec = loop["t_receive"].cpu().numpy(), loop["x_receive"].cpu().numpy()
    t_back = loop["t_back"].cpu().numpy()
    phi, x = field["phi"].cpu().numpy(), field["x"].cpu().numpy()
    pot = field["potential"].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
    ax_m, ax_a, ax_d, ax_c = axes.ravel()

    # A. Minkowski diagram of the signal fan in the moving frame
    lim = 1.6
    ax_m.plot([-lim, lim], [-lim, lim], color=GREY, ls="--", lw=1)
    ax_m.plot([-lim, lim], [lim, -lim], color=GREY, ls="--", lw=1)
    ax_m.set_xlim(-lim, lim)
    ax_m.set_ylim(-lim, lim)
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("x′")
    ax_m.set_ylabel("t′")
    lines = [ax_m.plot([], [], lw=2)[0] for _ in speeds]
    dots = ax_m.scatter([], [], s=18, zorder=3)
    labels = [ax_m.text(0, 0, f"{u:g}c", fontsize=7) for u in speeds]
    title_m = ax_m.set_title("signals seen at β\nred: received before sent", fontsize=9)

    # B. antitelephone in Alice's frame
    t_top = min(max(1.6, float(np.max(t_rec)) * 1.1), 8.0)
    t_bottom = min(-0.3, float(np.min(t_back)) - 0.2)
    ax_a.set_xlim(-0.2, max(1.0, float(np.max(x_rec[t_rec < t_top])) * 1.15))
    ax_a.set_ylim(max(t_bottom, -3.0), t_top)
    ax_a.axvline(0, color="#374151", lw=2)
    ax_a.axhline(1.0, color=AMBER, lw=1, ls=":")
    ax_a.text(0.02, 1.02, "message sent", color="#b45309", fontsize=7, transform=ax_a.get_yaxis_transform())
    (bob,) = ax_a.plot([], [], color="#374151", lw=1.5, ls="--")
    (msg,) = ax_a.plot([], [], color=BLUE, lw=2)
    (reply,) = ax_a.plot([], [], color=RED, lw=2)
    back_dot = ax_a.scatter([0], [1], s=30, color=RED, zorder=4)
    status = ax_a.text(0.97, 0.04, "", transform=ax_a.transAxes, ha="right", fontsize=8)
    title_a = ax_a.set_title("antitelephone\n", fontsize=9)
    ax_a.set_xlabel("x (Alice's frame)")
    ax_a.set_ylabel("t")
    loop_fill = []

    # C. neutrino speed data (static)
    names = ["OPERA (first result)", "OPERA (corrected)", "ICARUS", "SN1987A"]
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        row = data[name]
        colour = RED if name == "OPERA (first result)" else (GREEN if row["kind"] == "bound" else BLUE)
        ax_d.errorbar(
            1e6 * row["dv_over_c"],
            y,
            xerr=[[1e6 * row["err_minus"]], [1e6 * row["err_plus"]]],
            fmt="o",
            color=colour,
            capsize=3,
        )
        ax_d.text(-14, y + 0.18, f"{name} {int(row['year'])}", fontsize=7)
    ax_d.axvline(0, color="#374151", lw=1)
    ax_d.set_xlim(-15, 32)
    ax_d.set_ylim(-0.6, len(names) - 0.3)
    ax_d.set_yticks([])
    ax_d.set_xlabel(r"$(v-c)/c$  ($10^{-6}$)")
    ax_d.set_title("measured neutrino speeds: all consistent with c", fontsize=9)
    ax_d.text(
        0.98,
        0.04,
        f"OPERA 2011 as a tachyon: |m| ≈ {1000 * mu:.0f} MeV\n"
        f"→ 10 MeV SN1987A ν at ≈ {(1 + (mu / 0.01) ** 2) ** 0.5:.0f}c: ruled out",
        transform=ax_d.transAxes,
        ha="right",
        fontsize=7,
        color=RED,
    )

    # D. tachyon condensation
    ax_c.axhline(v, color=GREEN, ls="--", lw=1)
    ax_c.axhline(-v, color=GREEN, ls="--", lw=1)
    ax_c.axhline(0, color=GREY, lw=0.8)
    (field_line,) = ax_c.plot([], [], color=BLUE, lw=1.5)
    ax_c.set_xlim(x[0], x[-1])
    ax_c.set_ylim(-1.8 * v, 1.8 * v)
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("φ")
    title_c = ax_c.set_title("tachyon condensation", fontsize=9)

    n_steps = phi.shape[0]

    def update(f: int):
        beta = betas[f]
        # A
        flipped = seen[f, :, 0] < 0
        scale = lim * 0.9 / np.maximum(np.abs(seen[f]).max(-1), 1e-9)  # draw each signal to the same length
        ends = seen[f] * np.minimum(scale, 1.0)[:, None]
        for k, line in enumerate(lines):
            line.set_data([0, ends[k, 1]], [0, ends[k, 0]])
            line.set_color(RED if flipped[k] else BLUE)
            line.set_linestyle("--" if flipped[k] else "-")
            labels[k].set_position((ends[k, 1] + 0.03, ends[k, 0]))
        dots.set_offsets(np.column_stack([ends[:, 1], ends[:, 0]]))
        dots.set_color([RED if s else BLUE for s in flipped])
        title_m.set_text(f"signals from the origin seen at β = {beta:.2f}\nred: received before sent (β > 1/u)")
        # B
        for item in loop_fill:
            item.remove()
        loop_fill.clear()
        bob.set_data([0, beta * t_top], [0, t_top])
        msg.set_data([0, x_rec[f]], [1.0, t_rec[f]])
        reply.set_data([x_rec[f], 0], [t_rec[f], t_back[f]])
        back_dot.set_offsets([[0, t_back[f]]])
        closed = t_back[f] < 1.0
        loop_fill.append(
            ax_a.fill([0, x_rec[f], 0], [1.0, t_rec[f], t_back[f]], color=RED if closed else BLUE, alpha=0.12)[0]
        )
        status.set_text(
            f"reply {1 - t_back[f]:.2f} before the message!" if closed else f"reply arrives at t = {t_back[f]:.2f}"
        )
        status.set_color(RED if closed else "#111827")
        title_a.set_text(f"antitelephone: Bob recedes at β = {beta:.2f}\nAlice and Bob signal at u = {u0:g}c")
        # D
        step = int(round(f / max(len(betas) - 1, 1) * (n_steps - 1)))
        field_line.set_data(x, phi[step])
        title_c.set_text(f"tachyon condensation (m² = {params['m2']:g}): ⟨V⟩ = {pot[step]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(betas), 90), path)
