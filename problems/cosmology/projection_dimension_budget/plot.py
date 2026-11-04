"""Animate the (n, R) map filling in bound by bound, R_max(n), the force law and the quantum dimension count."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data, frame_indices
from problems._common.progress import report

from . import tensors

COLORS = ["#dcfce7", "#93c5fd", "#fca5a5", "#fcd34d", "#c4b5fd"]  # allowed, then one colour per bound family


def animate(n, R, em, rb, fc, qb, R_star, M_star, path: Path) -> Path:
    nv, lR = n.cpu().numpy(), np.log10(R.cpu().numpy())
    first = em["first"].cpu().numpy()
    per_stage = np.log10(np.clip(rb["per_stage"].cpu().numpy(), 1e-40, 1.0))  # (stages, N), inf -> 1 m
    n_st = len(tensors.STAGES)

    fig = plt.figure(figsize=(11, 7.2))
    ax_m, ax_b = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
    ax_f, ax_q = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)

    # --- the map ---
    cmap = ListedColormap(COLORS)
    extent = (lR[0], lR[-1], nv[0] - 0.5, nv[-1] + 0.5)
    img = ax_m.imshow(np.zeros_like(first, dtype=float), origin="lower", aspect="auto", extent=extent, cmap=cmap,
                      vmin=-0.5, vmax=n_st + 0.5, interpolation="nearest")  # fmt: skip
    for k, c in enumerate(COLORS):
        label = "allowed" if k == 0 else tensors.LABELS[tensors.STAGES[k - 1]]
        ax_m.plot([], [], "s", color=c, ms=8, label=label)
    ax_m.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    ax_m.set_xlabel("radius of the extra dimensions $\\log_{10}(R / 1\\,\\mathrm{m})$ (dimensionless)")
    ax_m.set_ylabel("extra dimensions $n$ (dimensionless)")
    ax_m.set_yticks(nv)
    for x, txt in ((math.log10(tensors.L_PLANCK_M), "Planck\nlength"), (-15, "1 fm"), (-6, "1 μm")):
        ax_m.axvline(x, color="#6b7280", lw=0.6, ls=":")
        ax_m.text(x + 0.3, nv[-1] - 0.2, txt, fontsize=6, color="#374151", va="top")
    if nv[-1] >= 7:
        ax_m.axhspan(6.5, nv[-1] + 0.5, xmin=0.25, xmax=0.85, fill=False, hatch="//", ec="#9ca3af", lw=0)
        ax_m.text(-22, 6.75, r"$n \geq 7$: only torsion bounds bundled", fontsize=7, color="#374151")
    star = ax_m.plot([], [], "k--", lw=1.2, label=f"ADD with $M_D$ = {M_star:g} TeV")[0]
    rmax_dots = ax_m.plot([], [], "o", color="#111827", ms=4)[0]
    title_m = ax_m.set_title("", fontsize=9)

    # --- R_max(n) ---
    bars = ax_b.bar(nv, np.zeros_like(nv) - 36, bottom=0, color="#86efac", edgecolor="#166534")
    ax_b.set_xlim(nv[0] - 0.6, nv[-1] + 0.6)
    ax_b.set_ylim(-36, 0.5)
    ax_b.axhline(math.log10(tensors.L_PLANCK_M), color="#6b7280", lw=0.8, ls=":")
    ax_b.set_xticks(nv)
    ax_b.set_xlabel("extra dimensions $n$ (dimensionless)")
    ax_b.set_ylabel("largest allowed $\\log_{10}(R_{max} / 1\\,\\mathrm{m})$ (dimensionless)")
    ax_b.set_title("dimension budget: every $n$ survives, but only if small", fontsize=9)
    bar_txt = [ax_b.text(k, 0, "", fontsize=6, ha="center", va="bottom") for k in nv]

    # --- force law ---
    r = fc["r"].cpu().numpy()
    for j, (k, c) in enumerate(fc["curves"].items()):
        ax_f.loglog(r, c.cpu().numpy(), color=ACCENT[j], lw=1.8, label=f"$n = {k}$: $1/r^{{{k + 2}}}$ → $1/r^2$")
    ax_f.axhline(1, color="#111827", lw=0.8, ls="--")
    ax_f.axvline(1, color="#9ca3af", lw=0.8, ls=":")
    ax_f.text(1.1, 3e3, "$r = R$", fontsize=8, color="#374151")
    ax_f.set_xlabel("distance $r / R$ (dimensionless)")
    ax_f.set_ylabel("$r^2 g(r) / G_4 M$, Newton = 1 (dimensionless)")
    ax_f.legend(fontsize=7, loc="upper right")
    ax_f.set_title("gravity spreads into $n$ dimensions below $R$ (image sums);\n"
                   "torsion balances see Newton down to 52 μm, so $R$ < 38.6 μm", fontsize=9)  # fmt: skip

    # --- quantum side ---
    N = qb["N"].cpu().numpy()
    ax_q.plot(N, qb["log10_hilbert"].cpu().numpy(), color=ACCENT[3], lw=2, label="Hilbert space $D = 2^N$ (step 1)")
    ax_q.plot(N, qb["log10_spatial"].cpu().numpy(), color=ACCENT[4], lw=1.4, ls="--",
              label="if spatial: $2^{N+1} - 3$ extra real dims")  # fmt: skip
    ax_q.axhspan(0, math.log10(6), color="#fca5a5", alpha=0.5, lw=0, label="$n$ = 1..6 with collider bounds")
    ax_q.plot([qb["qubits"]], [math.log10(qb["hilbert_q"])], "o", color="#111827", ms=6)
    ax_q.set_xlabel("entangled two-level systems $N$ (dimensionless)")
    ax_q.set_ylabel("$\\log_{10}$ dimensions needed (dimensionless)")
    ax_q.legend(fontsize=7, loc="upper left")
    ax_q.set_title(f"quantum side: {qb['qubits']} qubits need $D \\geq$ {qb['hilbert_q']}; race $d + K$ = {qb['race']}"
                   "\nthese need not be spatial (Hilbert-space reading)", fontsize=9)  # fmt: skip

    pre, per, hold = 4, 14, 14
    frames = list(range(pre + n_st * per + hold))
    frames = [frames[i] for i in frame_indices(len(frames), 72)]
    R_star_np = np.log10(R_star.cpu().numpy())
    rmax = np.log10(np.clip(rb["rmax"].cpu().numpy(), 1e-40, 1.0))

    def update(f: int):
        stage = (f - pre) // per if f >= pre else -1
        stage = min(stage, n_st)
        shown = np.zeros_like(first, dtype=float)
        for k in range(min(stage, n_st)):  # finished stages
            shown[first == k] = k + 1
        if 0 <= stage < n_st:  # the current stage sweeps in from large R
            frac = ((f - pre) % per + 1) / per
            cols = np.nonzero((first == stage).any(0))[0]
            lo = lR[cols[0]] if len(cols) else lR[-1]
            cut = lR[-1] - frac * (lR[-1] - lo)
            shown[(first == stage) & (lR[None, :] >= cut)] = stage + 1
            title_m.set_text(f"adding: {tensors.LABELS[tensors.STAGES[stage]]}")
        img.set_data(shown)
        done = min(stage, n_st - 1)
        if stage >= n_st:
            star.set_data(np.r_[R_star_np], nv[1:])
            rmax_dots.set_data(rmax[1:], nv[1:])
            title_m.set_text("all bounds: excluded unless $R < R_{max}(n)$ (dots); dashed: $M_D$ = "
                             f"{M_star:g} TeV")  # fmt: skip
        top = per_stage[done] if stage >= 0 else np.zeros_like(nv)
        for k, (b, t) in enumerate(zip(bars, bar_txt)):
            h = top[k] if nv[k] > 0 else 0.0
            b.set_height(h + 36)
            b.set_y(-36)
            b.set_color("#e5e7eb" if nv[k] >= 7 else "#86efac")
            b.set_edgecolor("#166534")
            t.set_position((nv[k], h + 0.3))
            t.set_text("any" if nv[k] == 0 else f"{10**h:.0e}" if h < -0.01 else "")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.9, wspace=0.28, hspace=0.48)
    return save_gif_shared_palette(fig, update, frames, path, fps=8)


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 8) -> Path:
    """Like plotting.save_gif, but every frame uses ONE palette built from a strip of all frames, so colours stay
    fixed through the animation (Pillow otherwise picks a new 256-colour palette per frame)."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)
