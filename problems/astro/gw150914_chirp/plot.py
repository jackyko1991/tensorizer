"""Animate the detection: whitened strain revealing the chirp, the Q-map and its fit, the bank scan, the overlay."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter

from problems._common.plotting import frame_indices, save_gif

from .equation import T_SUN, chirp_frequency

N_STEPS = 60


def animate(res: dict, path: Path) -> Path:
    t_e = res["t_event"]
    t = res["t"].numpy() - t_e  # seconds relative to the merger
    view = (t > -0.32) & (t < 0.06)
    tv = t[view]
    h1 = res["white_bp"][0].numpy()[view]
    fs = 1.0 / float(res["t"][1] - res["t"][0])
    shift = int(round(res["dt_cc_ms"] * 1e-3 * fs))
    l1 = -np.roll(res["white_bp"][1].numpy(), shift)[view]  # L1 moved onto H1's clock and sign-flipped
    tmpl = res["template_white"].numpy()[view]
    freqs = res["freqs"].numpy()
    cols = np.nonzero(view)[0][::4]
    qmap = res["qmap"].numpy()[:, cols]  # (Fq, T') H1 normalised energy
    tq = t[cols]
    bank, snr = res["bank"].numpy(), res["snr_bank"].numpy()
    net = res["network"].numpy()
    t_pts = res["t_pts"].numpy() - t_e
    f_pts = res["f_pts"].numpy()
    kept = res["kept"].numpy()
    tau = np.linspace(0.004, 0.32, 300)
    f_fit = chirp_frequency(torch.tensor(tau), res["mc_track"] * T_SUN).numpy()
    t_fit = res["tc_track"] - t_e - tau

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
    (ax_s, ax_q), (ax_b, ax_o) = axes
    lim = 1.1 * max(np.abs(h1).max(), np.abs(l1).max())

    (line_h,) = ax_s.plot([], [], color="#2563eb", lw=1.1, label="H1 Hanford")
    (line_l,) = ax_s.plot(
        [], [], color="#dc2626", lw=1.1, alpha=0.8, label=f"L1, shifted {res['dt_cc_ms']:.1f} ms, inverted"
    )
    ax_s.set_xlim(tv[0], tv[-1])
    ax_s.set_ylim(-lim, lim)
    ax_s.set_ylabel("whitened strain (σ)")
    ax_s.legend(loc="upper left", fontsize=7)
    ax_s.set_title("LIGO strain, whitened and band-passed 35–350 Hz", fontsize=9)

    masked = np.ma.masked_all_like(qmap)
    mesh = ax_q.pcolormesh(tq, freqs, masked, cmap="viridis", vmin=0, vmax=min(40, qmap.max()), shading="auto")
    ax_q.set_yscale("log")
    ax_q.set_xlim(tv[0], tv[-1])
    ax_q.set_ylim(freqs[0], freqs[-1])
    ax_q.set_yticks([30, 50, 100, 200, 400])
    ax_q.set_yticklabels(["30", "50", "100", "200", "400"])
    ax_q.yaxis.set_minor_formatter(NullFormatter())
    ax_q.set_ylabel("frequency (Hz)")
    ax_q.grid(False)
    pts = ax_q.scatter([], [], s=10, color="white", edgecolors="#111827", linewidths=0.5, zorder=3)
    (track,) = ax_q.plot([], [], color="#f59e0b", lw=2, zorder=4, label=f"Newtonian fit M_c = {res['mc_track']:.1f} M☉")
    ax_q.legend(loc="upper left", fontsize=7)
    ax_q.set_title("H1 Q-map (normalised energy) and ridge fit", fontsize=9)
    fig.colorbar(mesh, ax=ax_q, pad=0.01).ax.tick_params(labelsize=7)

    ax_b.set_xlim(bank[0], bank[-1])
    ax_b.set_ylim(0, 1.15 * net.max())
    (b_h,) = ax_b.plot([], [], color="#2563eb", lw=1.2, label="H1")
    (b_l,) = ax_b.plot([], [], color="#dc2626", lw=1.2, label="L1")
    (b_n,) = ax_b.plot([], [], color="#111827", lw=2, label="network √(ρ_H² + ρ_L²)")
    ax_b.axvspan(28.0, 32.0, color="#f59e0b", alpha=0.2, lw=0, label="published ≈ 30 M☉ (detector frame)")
    best_mark = ax_b.axvline(np.nan, color="#16a34a", ls="--", lw=1.2)
    ax_b.set_xlabel("template chirp mass M_c (M☉)")
    ax_b.set_ylabel("peak matched-filter SNR")
    ax_b.legend(loc="upper right", fontsize=7)
    title_b = ax_b.set_title("template bank, filtered in one batch", fontsize=9)

    ax_o.plot(tv, h1, color="#9ca3af", lw=1, label="H1 data (whitened)")
    (o_t,) = ax_o.plot([], [], color="#dc2626", lw=1.6, label=f"best template, M_c = {res['mc_bank']:.1f} M☉")
    ax_o.set_xlim(-0.2, 0.05)
    ax_o.set_ylim(-lim, lim)
    ax_o.set_xlabel("time from merger (s)")
    ax_o.set_ylabel("whitened strain (σ)")
    ax_o.legend(loc="upper left", fontsize=7)
    ax_o.set_title(f"template overlay: SNR {res['snr'][0]:.1f} (H1), {res['snr'][1]:.1f} (L1)", fontsize=9)
    for ax in (ax_s, ax_q):
        ax.set_xlabel("time from merger (s)")
    for ax in axes.flat:
        ax.tick_params(labelsize=7)

    def update(step: int):
        frac = step / (N_STEPS - 1)
        cursor = tv[0] + frac * (tv[-1] - tv[0])
        shown = tv <= cursor
        line_h.set_data(tv[shown], h1[shown])
        line_l.set_data(tv[shown], l1[shown])
        mesh.set_array(np.ma.masked_where(np.broadcast_to(tq > cursor, qmap.shape), qmap).ravel())
        seen = t_pts <= cursor
        pts.set_offsets(
            np.column_stack([t_pts[seen & kept], f_pts[seen & kept]]) if (seen & kept).any() else np.empty((0, 2))
        )
        on = t_fit <= cursor
        track.set_data(t_fit[on], f_fit[on])
        m = max(1, int(round(frac * len(bank))))
        b_h.set_data(bank[:m], snr[0, :m])
        b_l.set_data(bank[:m], snr[1, :m])
        b_n.set_data(bank[:m], net[:m])
        j = int(net[:m].argmax())
        best_mark.set_xdata([bank[j], bank[j]])
        title_b.set_text(f"template bank: best so far M_c = {bank[j]:.1f} M☉, network SNR {net[j]:.1f}")
        o_t.set_data(tv[shown], tmpl[shown])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(N_STEPS, N_STEPS), path, fps=10)
