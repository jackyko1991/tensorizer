"""Animate the telescope tests: GW amplitude against distance for a sweep of D, the posterior of D, the screening
bound, the supernova Hubble diagram, the shadow ring for a moving fold against the EHT bands, and the summary."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data, frame_indices
from problems._common.progress import report

from . import equation, solve

F64 = torch.float64
GREY, INK = "#6b7280", "#111827"
EM_COLOR = {"sbf": ACCENT[0], "shoes": ACCENT[4], "planck": ACCENT[3]}
EM_NAME = {"sbf": "SBF (Cantiello 2018)", "shoes": "Hubble flow, SH0ES $H_0$", "planck": "Hubble flow, Planck $H_0$"}


def summary_rows(posts, sh, sn, pardo) -> list[dict]:
    """One row per constraint: its deviation from GR in 3+1 in standard deviations, and the value as text."""
    rows = []
    for k in ("sbf", "shoes", "planck"):
        m, (lo, hi) = posts[k]["map"], posts[k]["interval"]
        s = (hi - m) if m < 4 else (m - lo)
        text = f"D = {m:.2f} (+{hi - m:.2f}/-{m - lo:.2f})"
        rows.append({"label": f"GW170817 D, {k}", "pull": (m - 4) / s, "text": text, "group": "gw"})
    for k, (m, lo, hi) in pardo.items():
        s = hi if m < 4 else lo
        rows.append({"label": f"Pardo 2018 D, {k}", "pull": (m - 4) / s, "text": f"D = {m:.2f} (+{hi:.2f}/-{lo:.2f})",
                     "group": "pub"})  # fmt: skip
    for r in sh["rows"]:
        name = "M87*" if r["key"] == "m87" else f"Sgr A* ({r['key'].split('_')[1]})"
        rows.append({"label": f"EHT shadow, {name}", "pull": r["pull"],
                     "text": f"delta = {r['delta']:+.2f}, r_f <= {r['r_bound']:.2f} r_s", "group": "eht"})  # fmt: skip
    rows.append({"label": "Pantheon+ leakage alpha", "pull": sn["mean"] / sn["sd"],
                 "text": f"alpha = {sn['mean']:.2f} +- {sn['sd']:.2f}", "group": "sn"})  # fmt: skip
    return rows


def _interval(x: np.ndarray, pdf: np.ndarray, q: float) -> float:
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))])
    return float(np.interp(q * cdf[-1], cdf, x))


def animate(gw, table, posts, key, scr, i_n, sh, sn, summary, path: Path) -> Path:
    fig, axs = plt.subplots(2, 4, figsize=(14.5, 7.0))
    (ax_h, ax_p, ax_r, ax_s), (ax_i, ax_d, ax_a, ax_f) = axs

    # ---- (a1) amplitude against distance, relative to 4D gravity ----
    d = np.logspace(0, 3, 200)
    dt = torch.tensor(d, dtype=F64)

    def ratio(D):  # h / h_4D = d / d_GW
        return (equation.strain_amplitude(dt, torch.tensor(D, dtype=F64)) * dt).numpy()

    for D, ls in ((3.8, ":"), (4.2, "--"), (4.5, "-.")):
        ax_h.semilogx(d, ratio(D), color=GREY, lw=1, ls=ls)
        ax_h.text(d[-1] * 0.9, ratio(D)[-1] + 0.04, f"D = {D:g}", fontsize=7, color=GREY, ha="right")
    ax_h.axhline(1, color=GREY, lw=1)
    ax_h.text(1.2, 1.04, "D = 4 (GR)", fontsize=7, color=GREY)
    x, L = solve.D_TABLE.numpy(), table.numpy()
    g_med, g_lo, g_hi = (_interval(x, L, q) for q in (0.5, 0.16, 0.84))
    from . import tensors

    for j, k in enumerate(("sbf", "shoes", "planck")):
        em = tensors.em_distance(k)
        y = em["d"] / g_med
        ax_h.errorbar(em["d"] * (1 + 0.04 * (j - 1)), y, xerr=em["sigma"],
                      yerr=[[y - em["d"] / g_hi], [em["d"] / g_lo - y]],
                      fmt="o", color=EM_COLOR[k], ms=5, capsize=2, lw=1.2, label=EM_NAME[k])  # fmt: skip
    sweep = ax_h.semilogx(d, ratio(4.0), color=INK, lw=2)[0]
    ax_h.set_xlim(1, 1000)
    ax_h.set_ylim(0, 2.2)
    ax_h.set_xlabel("distance $d_{EM}$ (Mpc)")
    ax_h.set_ylabel("$h / h_{4D} = d_{EM}/d_{GW}$ (dimensionless)")
    ax_h.legend(fontsize=6.5, loc="upper left")
    th = ax_h.set_title("", fontsize=9)

    # ---- (a2) posterior of D ----
    Ds = posts[key]["D"].numpy()
    for k in ("sbf", "shoes", "planck"):
        ax_p.plot(Ds, posts[k]["pdf"].numpy(), color=EM_COLOR[k], lw=2.2 if k == key else 1.2, label=EM_NAME[k])
    top = max(float(p["pdf"].max()) for p in posts.values())
    for j, (k, (m, lo, hi)) in enumerate((("shoes", (4.02, 0.10, 0.07)), ("planck", (3.98, 0.09, 0.07)))):
        y = top * (1.08 + 0.07 * j)
        ax_p.errorbar(m, y, xerr=[[lo], [hi]], fmt="s", color=EM_COLOR[k], ms=4, capsize=2, mfc="white")
        ax_p.text(m - lo - 0.03, y, f"Pardo 2018 ({k})", fontsize=6.5, va="center", ha="right", color=INK)
    lo, hi = posts[key]["interval"]
    ax_p.axvspan(lo, hi, color="#dbeafe", lw=0, zorder=0)
    ax_p.axvline(4, color=INK, lw=0.8, ls=":")
    ax_p.set_xlim(3.3, 4.6)
    ax_p.set_ylim(0, top * 1.45)
    ax_p.set_xlabel("spacetime dimensions gravity sees, $D$ (dimensionless)")
    ax_p.set_ylabel("posterior density (per unit $D$)")
    ax_p.legend(fontsize=6.5, loc="center left")
    ax_p.set_title(f"$D$ = {posts[key]['map']:.2f} (+{hi - posts[key]['map']:.2f}/-{posts[key]['map'] - lo:.2f}), "
                   "68% shaded", fontsize=9)  # fmt: skip
    cur_p = ax_p.axvline(4, color=INK, lw=1.5)

    # ---- (a3) screening scale ----
    nD = len(scr["D"]) - 1
    Dv = scr["D"][:nD].numpy()
    for j, n in enumerate(scr["n"].tolist()):
        bnd = scr["bound"][:nD, j].numpy()
        ax_r.semilogy(Dv, bnd, color=ACCENT[j % len(ACCENT)], lw=2.0 if j == i_n else 1.2, label=f"n = {n:g}")
    ax_r.axhline(20, color=GREY, ls="--", lw=1)
    ax_r.text(7.9, 23, "Pardo 2018: $R_c \\gtrsim 20$ Mpc", fontsize=6.5, ha="right", color=GREY)
    ax_r.axhline(float(scr["Rc"][-1]), color=GREY, lw=0.6)
    ax_r.set_ylim(5, 2e4)
    ax_r.set_xlabel("$D$ beyond the screening scale (dimensionless)")
    ax_r.set_ylabel("smallest allowed $R_c$ (Mpc)")
    ax_r.legend(fontsize=6.5, loc="upper right", title="steepness", title_fontsize=6.5)
    ax_r.set_title("screened leakage: excluded below the curves\n(curves at $10^4$ Mpc: grid edge)", fontsize=9)

    # ---- (c1) supernova Hubble residuals ----
    z, mu, sig = (sn[k].numpy() for k in ("z", "mu", "sigma"))
    w = sig**-2

    def offset_model(shape):
        s = shape.numpy()
        return s + np.sum(w * (mu - s)) / np.sum(w)

    base = offset_model(sn["shape"]["lcdm"])
    bins = np.logspace(np.log10(z.min()), np.log10(z.max() * 1.001), 16)
    idx = np.digitize(z, bins) - 1
    zc = np.array([np.sum(w[idx == i] * z[idx == i]) / np.sum(w[idx == i]) for i in range(15)])
    rc = np.array([np.sum(w[idx == i] * (mu - base)[idx == i]) / np.sum(w[idx == i]) for i in range(15)])
    re = np.array([np.sum(w[idx == i]) ** -0.5 for i in range(15)])
    ax_s.scatter(z, mu - base, s=2, color="#d1d5db", zorder=1)
    ax_s.errorbar(zc, rc, yerr=re, fmt="o", color=INK, ms=4, capsize=2, zorder=3, label="Pantheon+ (binned)")
    ax_s.axhline(0, color=ACCENT[0], lw=1.6, label=f"$\\Lambda$CDM, $\\Omega_m$ = {sn['lcdm'][1]:.2f}")
    ax_s.plot(z, offset_model(sn["shape"]["dgp"]) - base, color=ACCENT[1], lw=1.6,
              label=f"DGP leakage, $\\Omega_m$ = {sn['dgp'][1]:.2f}")  # fmt: skip
    ax_s.plot(z, offset_model(sn["shape"]["fit"]) - base, color=ACCENT[2], lw=1.2, ls="--",
              label=f"best $\\alpha$ = {sn['a_fit']:.2f}")  # fmt: skip
    ax_s.set_xscale("log")
    ax_s.set_ylim(-0.3, 0.3)
    ax_s.set_xlabel("redshift $z$ (dimensionless)")
    ax_s.set_ylabel("$\\mu - \\mu_{\\Lambda CDM}$ (mag)")
    ax_s.legend(fontsize=6.5, loc="lower left")
    ax_s.set_title(f"supernovae: $\\chi^2$ $\\Lambda$CDM {sn['lcdm'][0]:.1f}, DGP {sn['dgp'][0]:.1f}", fontsize=9)

    # ---- (b1) shadow ring image for Sgr A* (VLTI theta_g) ----
    row = next(r for r in sh["rows"] if r["key"] == "sgra_vlti")
    tg = row["thetag_dyn"]  # uas
    ext = 60.0
    g = np.linspace(-ext, ext, 161)
    rho = np.hypot(*np.meshgrid(g, g))
    beam = 20.0 / 2.355  # EHT resolution ~ 20 uas FWHM (schematic blur)
    img = ax_i.imshow(np.zeros_like(rho), extent=(-ext, ext, -ext, ext), origin="lower", cmap="afmhot", vmin=0,
                      vmax=1.05)  # fmt: skip
    ax_i.grid(False)
    t = np.linspace(0, 2 * np.pi, 200)
    r_sch = math.sqrt(27) * tg
    bands = []  # drawn over the (moving) image every frame
    for dd, ls, lab in ((row["delta"] - row["delta_lo"], ":", None), (row["delta"] + row["delta_hi"], ":", "68%"),
                        (row["delta_up"], "--", "95% upper")):  # fmt: skip
        bands += ax_i.plot(
            (1 + dd) * r_sch * np.cos(t), (1 + dd) * r_sch * np.sin(t), color="white", lw=1.2, ls=ls, label=lab
        )
    edge_line = ax_i.plot([], [], color="#22d3ee", lw=1.6, label="shadow edge")[0]
    ax_i.set_xlabel("$\\Delta\\alpha$ ($\\mu$as)")
    ax_i.set_ylabel("$\\Delta\\delta$ ($\\mu$as)")
    leg_i = ax_i.legend(fontsize=6, loc="lower right", facecolor="black", labelcolor="white", framealpha=0.6)
    ti = ax_i.set_title("", fontsize=9)

    # ---- (b2) delta against the fold radius ----
    rf = np.linspace(1.0, 3.2, 300)
    dl = equation.fold_delta(torch.tensor(rf, dtype=F64)).numpy()
    ax_d.plot(rf, dl, color=INK, lw=2, label="fold model $\\delta(r_f)$")
    kmin = min(r["kerr_min"] for r in sh["rows"])
    ax_d.axhspan(kmin, 0, color="#e5e7eb", lw=0, label="Kerr spin band")
    cols = {"m87": ACCENT[5], "sgra_keck": ACCENT[4], "sgra_vlti": ACCENT[3]}
    for r in sh["rows"]:
        c = cols[r["key"]]
        name = "M87*" if r["key"] == "m87" else f"Sgr A* {r['key'].split('_')[1]}"
        ax_d.errorbar(1.15 + 0.1 * list(cols).index(r["key"]), r["delta"], yerr=[[r["delta_lo"]], [r["delta_hi"]]],
                      fmt="o", color=c, ms=4, capsize=2)  # fmt: skip
        ax_d.axhline(r["allowed"], color=c, lw=1, ls="--")
        ax_d.plot([r["r_bound"]], [r["allowed"]], "v", color=c, ms=7, label=f"{name}: $r_f \\leq$ {r['r_bound']:.2f}")
    ax_d.axvline(1.5, color=GREY, lw=0.8, ls=":")
    ax_d.text(1.52, 0.42, "photon\nsphere", fontsize=6.5, color=GREY)
    ax_d.set_xlim(1.0, 3.2)
    ax_d.set_ylim(-0.25, 0.5)
    ax_d.set_xlabel("fold radius $r_f / r_s$ (dimensionless)")
    ax_d.set_ylabel("shadow change $\\delta$ (dimensionless)")
    ax_d.legend(fontsize=6, loc="upper left")
    ax_d.set_title("dots: measured $\\delta$; dashed: growth a fold may add", fontsize=9)
    dot_d = ax_d.plot([], [], "o", color="#f59e0b", ms=8)[0]

    # ---- (c2) posterior of the leakage index ----
    a, pa = sn["alpha"].numpy(), sn["pa"].numpy()
    ax_a.fill_between(a, pa, color="#bbf7d0", lw=0)
    ax_a.plot(a, pa, color=ACCENT[2], lw=2)
    for v, lab, c in ((0.0, "$\\Lambda$CDM", ACCENT[0]), (1.0, "DGP", ACCENT[1])):
        ax_a.axvline(v, color=c, lw=1.2, ls="--")
        ax_a.text(v + 0.03, pa.max() * 1.02, lab, fontsize=7, color=c)
    ax_a.set_ylim(0, pa.max() * 1.15)
    ax_a.set_xlabel("leakage index $\\alpha$ (dimensionless)")
    ax_a.set_ylabel("posterior density (per unit $\\alpha$)")
    ax_a.set_title(f"Pantheon+: $\\alpha$ = {sn['mean']:.2f} $\\pm$ {sn['sd']:.2f} (both allowed)", fontsize=9)

    # ---- summary ----
    group_c = {"gw": ACCENT[0], "pub": GREY, "eht": ACCENT[3], "sn": ACCENT[2]}
    ys = np.arange(len(summary))[::-1]
    ax_f.axvspan(-2, 2, color="#f3f4f6", lw=0)
    ax_f.axvspan(-1, 1, color="#e5e7eb", lw=0)
    ax_f.axvline(0, color=INK, lw=1)
    for y, r in zip(ys, summary):
        ax_f.plot([r["pull"]], [y], "o", color=group_c[r["group"]], ms=7)
        ax_f.text(-3.9, y + 0.22, f"{r['label']}: {r['text']}", fontsize=6.2, color=INK)
    ax_f.set_xlim(-4, 4)
    ax_f.set_ylim(-0.8, len(summary) - 0.2)
    ax_f.set_yticks([])
    ax_f.set_xlabel("deviation from GR in 3+1 (standard deviations)")
    ax_f.set_title("every test: consistent with $D = 4$ (bands 1, 2 $\\sigma$)", fontsize=9)

    # ---- frames: sweep D, then sweep the fold ----
    n1, n2, hold = 36, 36, 6
    D_seq = np.concatenate([np.linspace(3.4, 4.6, n1 // 2), np.linspace(4.6, 4.0, n1 - n1 // 2)])
    r_seq = np.concatenate([np.linspace(1.0, 3.0, n2 * 3 // 4), np.linspace(3.0, 1.5, n2 - n2 * 3 // 4)])
    pdf_sel = posts[key]["pdf"].numpy()
    total = n1 + n2 + hold

    def update(f: int):
        D = D_seq[min(f, n1 - 1)]
        sweep.set_ydata(ratio(D))
        th.set_text(f"GW170817: $h\\propto d^{{-(D-2)/2}}$, D = {D:.2f}\n"
                    f"posterior density {np.interp(D, Ds, pdf_sel):.2f}")  # fmt: skip
        cur_p.set_xdata([D])
        r = r_seq[min(max(f - n1, 0), n2 - 1)]
        b = float(equation.fold_shadow_radius(torch.tensor([r], dtype=F64))[0])
        R = b / (math.sqrt(27) / 2) * r_sch  # shadow edge in uas
        ring = 1.1 * R  # emission ring just outside the edge (schematic)
        img.set_data(np.exp(-0.5 * ((rho - ring) / math.hypot(2.0, beam)) ** 2))
        edge_line.set_data(R * np.cos(t), R * np.sin(t))
        dlt = b / (math.sqrt(27) / 2) - 1
        ti.set_text(f"Sgr A* (VLTI $\\theta_g$), fold at $r_f$ = {r:.2f} $r_s$\n$\\delta$ = {dlt:+.3f}")
        dot_d.set_data([r], [dlt])

    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.08, top=0.9, wspace=0.33, hspace=0.42)
    fig.suptitle("Projection Hypothesis step 6 (speculative): do telescopes see extra dimensions or a fold?",
                 fontsize=11)  # fmt: skip
    moving = [sweep, th, cur_p, img, *bands, edge_line, leg_i, ti, dot_d]
    return save_gif_shared_palette(fig, update, frame_indices(total, 78), path, fps=7, moving=moving)


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 8, moving=()) -> Path:
    """Render every frame, then quantise all of them with ONE palette built from a strip of frames, so colours do
    not drift between frames (Pillow otherwise picks a new 256-colour palette per frame).

    The static parts are drawn once; each frame restores that background and redraws only the moving artists."""
    fig.set_dpi(80)
    for art in moving:
        art.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    shots = []
    for i, f in enumerate(frames):
        fig.canvas.restore_region(background)
        update(f)
        for art in moving:
            fig.draw_artist(art)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)
