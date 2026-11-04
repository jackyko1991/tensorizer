"""Animate: measurement directions sweeping, E(theta) and S(theta) of the models, beyond-Born races, the data."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data
from problems._common.progress import report

N_FRAMES = 40
T2 = 2 * math.sqrt(2)
LOCAL_COLORS = ["#ea580c", "#9333ea", "#0891b2"]


def animate(co, rc, bb, rows, D, path: Path) -> Path:
    th = co["theta"].cpu().numpy()
    deg = np.degrees(th)
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0))
    ax_d, ax_e, ax_s = axs[0]
    ax_h, ax_m, ax_x = axs[1]

    # --- measurement directions ---
    circ = np.linspace(0, 2 * np.pi, 200)
    ax_d.plot(np.cos(circ), np.sin(circ), color="#d1d5db", lw=1)
    ax_d.set_aspect("equal")
    ax_d.set_xlim(-1.35, 1.35)
    ax_d.set_ylim(-1.35, 1.35)
    ax_d.set_axis_off()
    arrows = {}
    for name, col in (("a", ACCENT[0]), ("a'", ACCENT[0]), ("b", ACCENT[1]), ("b'", ACCENT[1])):
        line = ax_d.plot([], [], color=col, lw=2.2 if "'" not in name else 1.4, ls="-" if "'" not in name else "--")[0]
        label = ax_d.text(0, 0, name, color=col, fontsize=10, ha="center", va="center")
        arrows[name] = (line, label)
    td = ax_d.set_title("", fontsize=9)

    # --- E(theta) ---
    ax_e.plot(deg, co["E_qm"].cpu(), color="#111827", lw=2.2, label="quantum $-\\cos\\theta$")
    ax_e.plot(np.degrees(co["th_mc"].cpu()), co["E_mc"].cpu(), "o", color=ACCENT[2], ms=4,
              label=f"(i) joint race, $\\Psi\\in\\mathbb{{C}}^{{{D}}}$ (Monte Carlo)")  # fmt: skip
    m_max = co["E_joint"].shape[0]
    if m_max > 1:
        ax_e.plot(deg, co["E_joint"][-1].cpu(), color=ACCENT[1], lw=1.2, ls="--", label=f"joint race $m = {m_max}$")
    for j, (E, lab) in enumerate(zip(co["E_local"].cpu(), ["$m=1$", "$m=2$", "$m\\to\\infty$ (Bell 1964)"])):
        ax_e.plot(deg, E, color=LOCAL_COLORS[j], lw=1.3, label=f"(ii) local race {lab}")
    ax_e.set_xlabel("angle between settings $\\theta$ (degrees)")
    ax_e.set_ylabel("correlation $E(a, b)$ (dimensionless)")
    ax_e.set_xlim(0, 180)
    ax_e.set_ylim(-1.1, 1.1)
    ax_e.legend(fontsize=6, loc="upper left")
    cur_e = ax_e.axvline(0, color="#f59e0b", lw=1.2)
    ax_e.set_title("correlations", fontsize=9)

    # --- S(theta) ---
    ax_s.axhspan(0, 2, color="#fee2e2", lw=0)
    ax_s.axhspan(2, T2, color="#dcfce7", lw=0)
    ax_s.axhline(T2, color="#166534", lw=0.8, ls=":")
    ax_s.plot(deg, co["S_qm"].cpu(), color="#111827", lw=2.2, label="quantum = (i), $m = 1$")
    if m_max > 1:
        ax_s.plot(deg, co["S_joint"][-1].cpu(), color=ACCENT[1], lw=1.2, ls="--", label=f"joint race $m = {m_max}$")
    for j, S in enumerate(co["S_local"].cpu()):
        ax_s.plot(deg, S, color=LOCAL_COLORS[j], lw=1.3)
    ax_s.text(178, 1.9, "local bound 2", fontsize=7, ha="right", va="top", color="#991b1b")
    ax_s.text(178, T2 + 0.04, "Tsirelson $2\\sqrt{2}$", fontsize=7, ha="right", va="bottom", color="#166534")
    ax_s.set_xlim(0, 180)
    ax_s.set_ylim(0, 4)
    ax_s.set_xlabel("$\\theta$ (degrees): $a = 0$, $b = \\theta$, $a' = 2\\theta$, $b' = 3\\theta$")
    ax_s.set_ylabel("CHSH $S$ (dimensionless)")
    ax_s.legend(fontsize=6, loc="upper right")
    dots = ax_s.plot([], [], "o", color="#f59e0b", ms=6)[0]
    ts = ax_s.set_title("", fontsize=9)

    # --- S over random settings ---
    bins = np.linspace(0, 3.0, 61)
    ax_h.hist(rc["S_joint"].cpu().numpy(), bins=bins, color="#111827", alpha=0.6, label="(i) joint, $m=1$")
    for j, S in enumerate(rc["S_local"].cpu().numpy()):
        ax_h.hist(S, bins=bins, histtype="step", color=LOCAL_COLORS[j], lw=1.3, label=f"(ii) local, max {S.max():.3f}")
    ax_h.axvline(2, color="#991b1b", lw=1, ls=":")
    ax_h.axvline(T2, color="#166534", lw=1, ls=":")
    ax_h.set_xlabel("$S$ for random settings $a, a', b, b'$ (dimensionless)")
    ax_h.set_ylabel("settings per bin (count)")
    ax_h.legend(fontsize=6, loc="upper left")
    ax_h.set_title(f"{rc['S_joint'].shape[0]} random settings: local $\\leq 2$, joint $\\leq 2\\sqrt{{2}}$", fontsize=9)

    # --- beyond Born ---
    m = bb["m"].cpu().numpy()
    ax_m.plot(m, bb["S"].cpu(), color=ACCENT[1], lw=2, label="joint race $S$ (optimal settings)")
    ax_m.axhline(T2, color="#166534", lw=0.8, ls=":")
    ax_m.axvline(1, color="#16a34a", lw=0.8)
    ax_m.set_xlabel("hidden complex dimensions per outcome $m$ (dimensionless)")
    ax_m.set_ylabel("CHSH $S$ (dimensionless)", color=ACCENT[1])
    ax_m2 = ax_m.twinx()
    ax_m2.plot(m, bb["signal"].cpu(), color=ACCENT[5], lw=1.5, ls="--", label="signalling (partly entangled)")
    ax_m2.set_ylabel("signalling $\\Delta P_A(+)$ (dimensionless)", color=ACCENT[5])
    ax_m.set_title("$m \\neq 1$: post-quantum $S$ and faster-than-light signals", fontsize=9)
    h1, l1 = ax_m.get_legend_handles_labels()
    h2, l2 = ax_m2.get_legend_handles_labels()
    ax_m.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")

    # --- data ---
    ax_x.axhspan(1.6, 2, color="#fee2e2", lw=0)
    ax_x.axhspan(2, T2, color="#dcfce7", lw=0)
    ax_x.axhspan(T2, 3.3, color="#f3f4f6", lw=0)
    ax_x.axhline(2, color="#991b1b", lw=1, ls=":")
    ax_x.axhline(T2, color="#166534", lw=1, ls=":")
    data = [r for r in rows if isinstance(r["S"], float)]
    y = [r["S"] for r in data]
    e = [r["sigma"] for r in data]
    x = np.arange(len(data))
    ax_x.errorbar(x, y, yerr=e, fmt="o", color="#111827", capsize=3, ms=5)
    ax_x.set_xticks(x, [f"{r['experiment'].split(' ')[0]}\n{r['year']}" for r in data], fontsize=7)
    ax_x.text(len(data) - 0.6, 1.65, "(ii) local races: $S \\leq 2$", fontsize=7, ha="right", color="#991b1b")
    ax_x.text(len(data) - 0.6, 2.75, "(i) joint race = QM", fontsize=7, ha="right", color="#166534", va="top")
    ax_x.text(len(data) - 0.6, 3.25, "joint race, $m > 1$", fontsize=7, ha="right", color="#6b7280", va="top")
    ax_x.set_xlim(-0.6, len(data) - 0.4)
    ax_x.set_ylim(1.6, 3.3)
    ax_x.set_ylabel("measured CHSH $S$ (dimensionless)")
    ax_x.set_xlabel("loophole-free experiment")
    others = [r for r in rows if not isinstance(r["S"], float)]
    note = "; ".join(f"{r['experiment'].split(' ')[0]} $p$ = {r['p_value']}" for r in others)
    ax_x.set_title(f"loophole-free tests (CH-Eberhard: {note})", fontsize=7)

    th_f = np.linspace(0, np.pi / 2, N_FRAMES)

    def update(f: int):
        t = th_f[f]
        for name, phi in (("a", 0.0), ("b", t), ("a'", 2 * t), ("b'", 3 * t)):
            line, label = arrows[name]
            line.set_data([0, np.sin(phi)], [0, np.cos(phi)])
            label.set_position((1.18 * np.sin(phi), 1.18 * np.cos(phi)))
        i = int(round(t / np.pi * (len(th) - 1)))
        Sq = co["S_qm"][i].item()
        td.set_text(f"settings in one plane, $\\theta$ = {np.degrees(t):.0f}°\nAlice $a, a'$ (blue), Bob $b, b'$ (red)")
        cur_e.set_xdata([np.degrees(t)])
        ys = [Sq] + [S[i].item() for S in co["S_local"]]
        dots.set_data([np.degrees(t)] * len(ys), ys)
        ts.set_text(f"CHSH: quantum {Sq:.3f}, local {max(ys[1:]):.3f}")

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.92, wspace=0.6, hspace=0.45)
    return save_gif_shared_palette(fig, update, list(range(N_FRAMES)), path, fps=8)


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 8) -> Path:
    """Like plotting.save_gif, but every frame is quantised to ONE palette built from a strip of frames, so a colour
    never drifts from frame to frame."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]  # a strip of frames spanning the run sets the palette
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)
