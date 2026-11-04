"""Animate the criteria: growing matter/antimatter domains and their gamma rays, the worldline across the CPT mirror,
and the neutrino-mass and tensor-to-scalar axes with their bounds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from PIL import Image

from problems._common.plotting import export_plot_data
from problems._common.progress import report

from . import equation, solve

BLUE, RED, AMBER, GREEN, GREY, INK = "#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#6b7280", "#111827"
LIGHT_BLUE, LIGHT_RED = "#bfdbfe", "#fecaca"


def _wrapped(eta: np.ndarray) -> np.ndarray:
    """Conformal time folded back onto the circle [-1, 1)."""
    return (eta + 1) % 2 - 1


def _cuts(e: np.ndarray, x: np.ndarray, u: float):
    """x positions where the wrapped worldline crosses eta = u, split by direction in eta (wrap jumps skipped)."""
    a, b = e[:-1], e[1:]
    ok = np.abs(b - a) < 1
    up = ok & (a < u) & (u <= b)
    down = ok & (b < u) & (u <= a)
    frac = (u - a) / np.where(b != a, b - a, 1)
    xc = x[:-1] + frac * (x[1:] - x[:-1])
    return xc[up], xc[down]


def animate(c, dom, maps, mi, dims, path: Path) -> Path:
    n = len(dom["d"])
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.38, wspace=0.26)
    ax_m, ax_g, ax_w = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    sub = gs[1, 1].subgridspec(2, 1, hspace=1.0)
    ax_s, ax_r = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])

    # --- domains of matter and antimatter, glowing walls, our horizon ---
    L = solve.BOX_GPC
    cmap = ListedColormap([LIGHT_RED, AMBER, LIGHT_BLUE])  # -1 antimatter, 0 wall, +1 matter

    def picture(i: int) -> np.ndarray:
        s = maps[i].numpy()
        wall = np.zeros_like(s, dtype=bool)
        for sh, ax in ((1, 0), (1, 1)):  # one-sided: walls one pixel thick
            wall |= s != np.roll(s, sh, ax)
        return np.where(wall, 0, s)

    img = ax_m.imshow(picture(0), cmap=cmap, vmin=-1.5, vmax=1.5, extent=(-L / 2, L / 2, -L / 2, L / 2),
                      origin="lower", interpolation="nearest")  # fmt: skip
    th = np.linspace(0, 2 * np.pi, 200)
    R = c["horizon_gpc"]
    ax_m.plot(R * np.cos(th), R * np.sin(th), "--", color=INK, lw=1.2)
    ax_m.text(0, -R + 1.2, "our horizon (14.3 Gpc)", ha="center", fontsize=7, color=INK,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})  # fmt: skip
    ax_m.set_xlim(-L / 2, L / 2)
    ax_m.set_ylim(-L / 2, L / 2)
    ax_m.set_xlabel("comoving $x$ (Gpc)")
    ax_m.set_ylabel("comoving $y$ (Gpc)")
    for col, lab in ((LIGHT_BLUE, "matter domain"), (LIGHT_RED, "antimatter domain"), (AMBER, "annihilation wall")):
        ax_m.plot([], [], "s", color=col, ms=8, label=lab)
    ax_m.legend(fontsize=7, loc="upper right", framealpha=0.9)
    title_m = ax_m.set_title("", fontsize=9)

    # --- the diffuse gamma-ray background: predicted for this d against the data ---
    E = dom["E"].numpy()
    e2 = E**2
    ax_g.loglog(E, e2 * dom["comptel"].numpy(), color=INK, lw=3, label="COMPTEL fit (0.8-30 MeV)")
    ax_g.loglog(E, e2 * dom["fermi"].numpy(), color=GREY, lw=3, label="Fermi-LAT IGRB fit (0.1-820 GeV)")
    J = dom["J"].numpy()
    curve = ax_g.loglog(E, np.where(J[0] > 0, e2 * J[0], np.nan), color=AMBER, lw=2,
                        label="wall annihilation, $z$ = 20 to recombination")[0]  # fmt: skip
    ax_g.set_xlim(E[0], E[-1])
    ax_g.set_ylim(1e-6, 3.0)
    ax_g.set_xlabel("photon energy today $E$ (MeV)")
    ax_g.set_ylabel(r"$E^2\,dJ/dE$ (MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax_g.legend(fontsize=7, loc="upper right")
    title_g = ax_g.set_title("", fontsize=9)

    # --- one worldline winding through the Big Bang into the CPT mirror ---
    eta, x = mi["eta"].numpy(), mi["x"].numpy()
    e = _wrapped(eta)
    pts = np.stack([x, e], 1)
    seg = np.stack([pts[:-1], pts[1:]], 1)
    jump = np.abs(np.diff(e)) > 1
    fwd = np.diff(eta) > 0
    ours = (e[:-1] + e[1:]) / 2 > 0
    particle = fwd == ours  # forward in eta reads as matter on our side, as antimatter in the mirror
    colors = np.where(particle, BLUE, RED)
    lc = LineCollection(seg[~jump], colors=colors[~jump], linewidths=1.5)
    ax_w.axhspan(0, 1, color="#eff6ff", lw=0)
    ax_w.axhspan(-1, 0, color="#fef2f2", lw=0)
    ax_w.add_collection(lc)
    ax_w.axhline(0, color=INK, lw=1.5)
    pad = 0.1 * (x.max() - x.min())
    ax_w.set_xlim(x.min() - pad, x.max() + pad)
    ax_w.set_ylim(-1, 1)
    ax_w.text(x.min(), 0.88, "our universe: arrow of time up", fontsize=7, color="#1e40af")
    ax_w.text(x.min(), -0.95, "CPT mirror: arrow of time down", fontsize=7, color="#b91c1c")
    ax_w.text(x.max(), 0.03, "Big Bang $\\eta = 0$", fontsize=7, ha="right")
    ax_w.set_xlabel("space $x$, one of $D-1$ axes (a.u.)")
    ax_w.set_ylabel(r"conformal time $\eta$ (a.u.; $\pm1$ glued)")
    sl_o = ax_w.axhline(0.5, color=AMBER, lw=1.6)
    sl_m = ax_w.axhline(-0.5, color=AMBER, lw=1.6, ls="--")
    dots_b = ax_w.plot([], [], "o", color=BLUE, ms=6, mec="white")[0]
    dots_r = ax_w.plot([], [], "o", color=RED, ms=6, mec="white")[0]
    count = ax_w.text(0.99, 0.6, "", transform=ax_w.transAxes, ha="right", fontsize=7,
                      bbox={"facecolor": "white", "edgecolor": "#d1d5db"})  # fmt: skip
    ax_w.set_title(f"one worldline in $D = {dims}$ winding $w = {mi['w']}$ times: blue = matter, red = antimatter,\n"
                   "each side read with its own arrow of time (author's alignment)", fontsize=9)  # fmt: skip

    # --- the sum of neutrino masses ---
    ax_s.set_xscale("log")
    ax_s.set_xlim(0.03, 6.0)
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    bounds = [(c["desi_dr2_sum_mnu"], "DESI DR2", 0.9), (c["desi_2024_sum_mnu"], "DESI 2024", 0.62),
              (c["planck_sum_mnu"], "Planck 2018", 0.9), (3 * c["katrin_2025"], "KATRIN 2025\n($3m_\\beta$)", 0.62),
              (3 * c["katrin_2022"], "KATRIN 2022\n($3m_\\beta$)", 0.9)]  # fmt: skip
    for b, lab, yy in bounds:
        ax_s.axvline(b, color=GREY, lw=1)
        ax_s.text(b * 1.03, yy, lab, fontsize=6, va="top", color=GREY)
    ax_s.axvspan(c["desi_dr2_sum_mnu"], 6.0, color="#f3f4f6", lw=0)
    ax_s.plot([mi["sum_no0"]], [0.22], "^", color=GREEN, ms=9)
    ax_s.plot([mi["sum_io0"]], [0.22], "v", color=AMBER, ms=9)
    nl = chr(10)
    ax_s.text(mi["sum_no0"], 0.02, f"normal{nl}{mi['sum_no0']:.3f} eV", fontsize=6, ha="right", color=GREEN)
    ax_s.text(mi["sum_io0"] * 1.04, 0.02, f"inverted{nl}{mi['sum_io0']:.3f} eV", fontsize=6, color="#b45309")
    ax_s.text(0.031, 0.62, f"massless{nl}lightest " + r"$\nu$:", fontsize=6, color=INK)
    toy_s = ax_s.plot([], [], "D", ms=7, mec=INK)[0]
    ax_s.set_xlabel(r"sum of neutrino masses $\Sigma m_\nu$ (eV)")
    title_s = ax_s.set_title("", fontsize=8)

    # --- the tensor-to-scalar ratio ---
    ax_r.set_xscale("log")
    ax_r.set_xlim(1e-4, 1.0)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([])
    ax_r.axvspan(c["bicep_keck_r"], 1.0, color="#f3f4f6", lw=0)
    ax_r.axvline(c["bicep_keck_r"], color=GREY, lw=1)
    ax_r.text(c["bicep_keck_r"] * 1.1, 0.85, f"BICEP/Keck $r < {c['bicep_keck_r']}$", fontsize=6, va="top", color=GREY)
    ax_r.axvline(2 * c["future_sigma_r"], color=GREY, lw=1, ls="--")
    ax_r.text(2 * c["future_sigma_r"] * 1.1, 0.85, "future $2\\sigma(r)$", fontsize=6, va="top", color=GREY)
    ax_r.annotate("CPT universe: $r = 0$", xy=(1e-4, 0.3), xytext=(2.2e-4, 0.3), fontsize=7, va="center",
                  arrowprops={"arrowstyle": "->", "color": GREEN}, color=GREEN)  # fmt: skip
    toy_r = ax_r.plot([], [], "D", ms=7, mec=INK)[0]
    ax_r.set_xlabel("tensor-to-scalar ratio $r$ (dimensionless)")
    title_r = ax_r.set_title("", fontsize=8)

    levels = mi["levels"].numpy()
    toy_m = np.concatenate([[0.0], np.logspace(-3, np.log10(0.5), n - 1)])
    toy_rr = np.logspace(-4, np.log10(0.3), n)
    toy_sum = equation.sum_mnu(  # the toy points use the same functions as the tests
        torch.tensor(toy_m), "normal", c
    ).numpy()

    def update(i: int):
        d = dom["d"][i].item()
        img.set_data(picture(i))
        title_m.set_text(f"domain size $d$ = {d / 1e3:.2f} Gpc")
        curve.set_ydata(np.where(J[i] > 0, e2 * J[i], np.nan))
        ratio = dom["ratio"][i].item()
        verdict = "excluded" if ratio > 1 else "allowed"
        title_g.set_text(f"worst predicted / observed = {ratio:.2g}: {verdict}\n"
                         f"($d_{{min}}$ = {dom['d_min_mpc'] / 1e3:.0f} Gpc; flux $\\propto 1/d$)")  # fmt: skip
        k = (i * 3) % (2 * len(levels) - 2)
        u = levels[k if k < len(levels) else 2 * len(levels) - 2 - k]
        sl_o.set_ydata([u])
        sl_m.set_ydata([-u])
        bo, ro = _cuts(e, x, u)  # ours: forward = matter
        fm, bm = _cuts(e, x, -u)  # mirror: forward in eta = antimatter
        dots_b.set_data(np.concatenate([bo, bm]), np.concatenate([np.full_like(bo, u), np.full_like(bm, -u)]))
        dots_r.set_data(np.concatenate([ro, fm]), np.concatenate([np.full_like(ro, u), np.full_like(fm, -u)]))
        count.set_text(f"our slice: {len(bo)} matter, {len(ro)} anti, net {len(bo) - len(ro):+d}\n"
                       f"mirror slice: {len(bm)} matter, {len(fm)} anti, net {len(bm) - len(fm):+d}\n"
                       f"total {len(bo) - len(ro) + len(bm) - len(fm):+d}")  # fmt: skip
        s_ok = toy_sum[i] < c["desi_dr2_sum_mnu"]
        toy_s.set_data([toy_sum[i]], [0.55])
        toy_s.set_color(GREEN if s_ok else RED)
        title_s.set_text(f"toy point: lightest mass {toy_m[i]:.3f} eV (normal) -> "
                         f"{'allowed' if s_ok else 'excluded'} by DESI DR2")  # fmt: skip
        r_ok = toy_rr[i] < c["bicep_keck_r"]
        toy_r.set_data([toy_rr[i]], [0.3])
        toy_r.set_color(GREEN if r_ok else RED)
        title_r.set_text(f"toy point: $r$ = {toy_rr[i]:.1e} -> {'allowed' if r_ok else 'excluded'} by BICEP/Keck")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.94)
    return save_gif_shared_palette(fig, update, list(range(n)), path, fps=6)


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
