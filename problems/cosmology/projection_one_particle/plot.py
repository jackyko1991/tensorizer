"""Animate a time slice sweeping one zig-zagging worldline, the counters, and the asymmetry the data demand."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from problems._common.plotting import export_plot_data, frame_indices
from problems._common.progress import report

BLUE, RED, GREEN = "#2563eb", "#dc2626", "#16a34a"


def _cuts(t: np.ndarray, x: np.ndarray, u: float):
    """x positions where the closed loop crosses t = u, split into forward and backward crossings."""
    a, b = t, np.roll(t, -1)
    xa, xb = x, np.roll(x, -1)
    up = (a < u) & (u <= b)
    down = (b < u) & (u <= a)
    frac = (u - a) / np.where(b != a, b - a, 1)
    xc = xa + frac * (xb - xa)
    return xc[up], xc[down]


def animate(rep, bt, ob, levels, dims, path: Path) -> Path:
    X = rep["X"].cpu().numpy()
    t, x = X[:, 0], X[:, 1]
    lv = levels.cpu().numpy()
    up_r, down_r = rep["up"].cpu().numpy(), rep["down"].cpu().numpy()

    fig = plt.figure(figsize=(10.5, 6.8))
    ax_w, ax_c = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
    ax_f, ax_a = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)

    # --- the worldline, coloured by the direction it runs in time ---
    pts = np.stack([x, t], 1)
    seg = np.stack([pts, np.roll(pts, -1, 0)], 1)
    fwd = np.roll(t, -1) > t
    ax_w.add_collection(LineCollection(seg, colors=np.where(fwd, BLUE, RED), linewidths=1.6))
    births, deaths = rep["births"].cpu().numpy(), rep["deaths"].cpu().numpy()
    ax_w.plot(x[births], t[births], "^", color=GREEN, ms=7, label="pair creation (minimum of $t$)")
    ax_w.plot(x[deaths], t[deaths], "v", color="#111827", ms=7, label="annihilation (maximum of $t$)")
    pad = 0.15 * (x.max() - x.min())
    ax_w.set_xlim(x.min() - pad, x.max() + pad)
    ax_w.set_ylim(lv[0], lv[-1])
    ax_w.set_xlabel("space $x$, one axis of the projection (a.u.)")
    ax_w.set_ylabel("time $t$ (a.u.)")
    ax_w.legend(fontsize=7, loc="upper left")
    slice_line = ax_w.axhline(lv[0], color="#f59e0b", lw=2)
    dots_p = ax_w.plot([], [], "o", color=BLUE, ms=8, mec="white")[0]
    dots_a = ax_w.plot([], [], "o", color=RED, ms=8, mec="white")[0]
    counter = ax_w.text(0.98, 0.03, "", transform=ax_w.transAxes, ha="right", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "#d1d5db"})  # fmt: skip
    ax_w.set_title(f"one closed worldline in $D = {dims}$, seen in $(x, t)$:\nblue forward = particle, "
                   "red backward = antiparticle", fontsize=9)  # fmt: skip

    # --- counters per slice ---
    ax_c.step(lv, up_r, where="mid", color=BLUE, lw=1.6, label="particles $N_+$ (this loop)")
    ax_c.step(lv, down_r + 0.06, where="mid", color=RED, lw=1.2, ls="--", label="antiparticles $N_-$ (this loop)")
    ax_c.plot(lv, bt["mean_N"].cpu().numpy() / 2, "o", color="#6b7280", ms=3, label="batch mean $N/2$")
    ax_c.plot(lv, bt["rice"].cpu().numpy() / 2, color="#111827", lw=1, label="Rice formula / 2")
    ax_c.set_xlabel("slice time $u$ (a.u.)")
    ax_c.set_ylabel(r"count on the slice $N_\pm$ (dimensionless)")
    ax_c.legend(fontsize=7, loc="upper left")
    ax_c.set_ylim(0, max(up_r.max(), bt["mean_N"].max().item() / 2) * 1.35 + 0.5)
    cursor = ax_c.axvline(lv[0], color="#f59e0b", lw=1.5)
    ax_c.set_title(f"$N_+ = N_-$ on every slice of every closed loop (max $|Q|$ = {bt['max_abs_Q']})", fontsize=9)

    # --- AMS-02 positron fraction ---
    E, f, sg = ob["E"].cpu().numpy(), ob["fraction"].cpu().numpy(), ob["sigma"].cpu().numpy()
    ax_f.errorbar(E, f, yerr=sg, fmt="o", ms=3, color=BLUE, ecolor="#93c5fd", label="AMS-02 2019 (cited data)")
    ax_f.axhline(0.5, color=RED, lw=2, label="one closed worldline: $f = 1/2$")
    ax_f.set_xscale("log")
    ax_f.set_ylim(0, 0.6)
    ax_f.set_xlabel("kinetic energy $E$ (GeV)")
    ax_f.set_ylabel("positron fraction $f = e^+/(e^+ + e^-)$\n(dimensionless)")
    ax_f.legend(fontsize=7, loc="center left")
    ax_f.set_title("cosmic rays: positrons are only 5 to 16% of $e^\\pm$", fontsize=9)

    # --- asymmetry: what one worldline allows against what is seen ---
    N = np.logspace(0, 90, 200)
    ax_a.loglog(N, 1 / N, color=RED, lw=2, label="one open worldline: $|Q|/N \\leq 1/N$")
    ax_a.fill_between(N, 1e-95, 1 / N, color="#fecaca", alpha=0.6, lw=0)
    ax_a.axhspan(ob["A_min"], ob["A_max"], color="#bfdbfe", alpha=0.8, lw=0)
    ax_a.text(1e40, 1e-6, "blue band: cosmic-ray $e^\\pm$ (AMS-02)", fontsize=7, color="#1e40af")
    ax_a.plot(
        [ob["N_gamma"]], [ob["eta"]], "*", color="#111827", ms=12, label=f"early universe: $\\eta$ = {ob['eta']:.1e}"
    )
    ax_a.plot([ob["N_B"]], [1.0], "s", color="#9333ea", ms=8, label=f"today: {ob['N_B']:.0e} baryons, no antimatter")
    ax_a.set_xlim(1, 1e91)
    ax_a.set_ylim(1e-92, 10)
    ax_a.set_xlabel("crossings on a slice $N$ (dimensionless)")
    ax_a.set_ylabel("asymmetry $|N_+ - N_-|/N$ (dimensionless)")
    ax_a.legend(fontsize=7, loc="lower left")
    ax_a.set_title(f"net charge short by {ob['N_B']:.0e}: ruled out in its simplest form", fontsize=9)

    n = len(lv)
    sweep = list(range(n)) + list(range(n - 2, 0, -1))
    sweep = [sweep[i] for i in frame_indices(len(sweep), 64)]

    def update(i: int):
        u = lv[i]
        slice_line.set_ydata([u])
        cursor.set_xdata([u])
        xp, xa = _cuts(t, x, u)
        dots_p.set_data(xp, np.full_like(xp, u))
        dots_a.set_data(xa, np.full_like(xa, u))
        counter.set_text(f"particles {len(xp)}   antiparticles {len(xa)}   net {len(xp) - len(xa)}")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.92, wspace=0.25, hspace=0.42)
    return save_gif_shared_palette(fig, update, sweep, path, fps=8)


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
