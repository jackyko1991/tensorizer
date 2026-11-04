"""Animate: the Flamm surface turning, a sheet tilting until its shadow folds, fold exponents, GR vs fold curvature."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data
from problems._common.progress import report

N_FRAMES = 40
CAT_NAMES = {2: "fold", 3: "cusp", 4: "swallowtail", 5: "butterfly"}


def animate(em, cu, fo, R, a, D, path: Path) -> Path:
    fig = plt.figure(figsize=(11.5, 7.2))
    ax_f = fig.add_subplot(2, 3, 1, projection="3d")
    ax_s = fig.add_subplot(2, 3, 2, projection="3d")
    ax_h = fig.add_subplot(2, 3, 3)
    ax_e, ax_g, ax_j = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)

    # --- Flamm paraboloid (both sheets) ---
    X, Y, Z = (m.cpu().numpy() for m in em["flamm_mesh"])
    for sgn in (1, -1):
        ax_f.plot_surface(X, Y, sgn * Z, color=ACCENT[0], alpha=0.55, linewidth=0.2, edgecolor="#1e3a8a")
    th = np.linspace(0, 2 * np.pi, 60)
    ax_f.plot(np.cos(th), np.sin(th), 0 * th, color="#dc2626", lw=2)
    ax_f.set_axis_off()
    ax_f.set_title("known: Flamm paraboloid $z = 2\\sqrt{r_s(r-r_s)}$\n(red: horizon $r = r_s$)", fontsize=9)

    # --- tilted wavy sheet and its shadow ---
    uu, vv = np.meshgrid(np.linspace(-2 * np.pi, 2 * np.pi, 48), np.linspace(-2, 2, 4))
    alpha = fo["alpha"].cpu().numpy()
    xf = fo["x_fold"].cpu().numpy()
    uf = fo["u_fold"].cpu().numpy()
    ZS = 2 * np.pi + a
    lim = 2 * np.pi + a
    ax_h.set_xlim(-lim, lim)
    centres = 0.5 * (fo["edges"][1:] + fo["edges"][:-1]).cpu().numpy()
    dens = fo["density"].cpu().numpy()
    ax_h.set_ylim(0.05, dens.max() * 1.5)
    ax_h.set_yscale("log")
    hist = ax_h.step(centres, dens[0], where="mid", color=ACCENT[0], lw=1.3)[0]
    fold_lines = [ax_h.axvline(0, color="#dc2626", lw=0.8, ls=":", visible=False) for _ in range(4)]
    ax_h.set_xlabel("shadow coordinate $x$ (a.u.)")
    ax_h.set_ylabel("projected density $\\rho(x)$ (a.u.)")
    th_h = ax_h.set_title("", fontsize=9)

    # --- universal exponents ---
    for j, ((c, rho), n, al) in enumerate(zip(fo["cat"], fo["n"].tolist(), fo["alpha_cat"].tolist())):
        c, rho = c.numpy(), rho.numpy()
        ax_e.loglog(c, rho / rho[-1], "o", ms=3, color=ACCENT[j])
        ax_e.loglog(c, (c / c[-1]) ** (1 / n - 1), color=ACCENT[j], lw=1,
                    label=f"{CAT_NAMES[n]} $x=s^{n}$: {al:+.3f} (exact {1 / n - 1:+.3f})")  # fmt: skip
    for j, ((c, rho), k, al) in enumerate(zip(fo["sph"], fo["k"].tolist(), fo["alpha_sph"].tolist())):
        c, rho = c.numpy(), rho.numpy()
        ok = rho > 0
        y = rho[ok] / rho[ok][-1] * 10 ** (-1.5 - j)
        ax_e.loglog(c[ok], y, "s-", ms=2.5, lw=0.8, color="#6b7280")
        ax_e.text(c[ok][-1] * 1.3, y[-1], f"$S^{k}\\to\\mathbb{{R}}^3$: {al:+.2f}", fontsize=6, color="#374151",
                  va="center")  # fmt: skip
    ax_e.set_xlabel("distance to the caustic $|x|$ or $u = 1-|Px|^2$ (a.u.)")
    ax_e.set_ylabel("density $\\rho$, rescaled (a.u.)")
    ax_e.legend(fontsize=6, loc="upper right")
    ax_e.set_title(f"P4 exponents: fold $-1/2$, cusp $-2/3$; $S^k\\to\\mathbb{{R}}^3$ in $\\mathbb{{R}}^{{{D}}}$",
                   fontsize=9)  # fmt: skip

    # --- GR vs fold shell: curvature against r ---
    r = cu["r"].cpu().numpy()
    toy = cu["toy"]
    ax_g.loglog(r, cu["K_exact"].cpu().numpy(), color="#111827", lw=2, label="GR $K = 12r_s^2/r^6$")
    ax_g.loglog(r[::4], cu["K"].cpu().numpy()[::4], "o", ms=3, mfc="none", color="#111827", label="autograd $K$")
    pk = toy["proxy_K"].cpu().numpy()
    ax_g.loglog(r, pk, color=ACCENT[1], lw=1.6, label="fold shell, $8\\sum E^2$ (weak field)")
    ax_g.axvline(1.0, color="#9ca3af", ls="--", lw=0.8)
    ax_g.text(1.05, 1e-2, "$r_s$", fontsize=8, color="#6b7280")
    ax_g.axvline(R, color=ACCENT[1], ls=":", lw=0.8)
    ax_g.set_ylim(1e-3, 1e11)
    ax_g.set_xlabel("radius $r / r_s$ (dimensionless)")
    ax_g.set_ylabel("curvature $K$ ($r_s^{-4}$)")
    ax_g.legend(fontsize=6, loc="upper right")
    probe = ax_g.plot([], [], "o", color="#f59e0b", ms=7)[0]
    probe_t = ax_g.plot([], [], "o", color=ACCENT[1], ms=6)[0]
    tg = ax_g.set_title("", fontsize=9)

    # --- Jacobian rank of Fronsdal's embedding ---
    rl = em["r_line"].cpu().numpy()
    ax_j.semilogy(
        rl, em["sv_schw"].cpu().numpy(), color="#6b7280", lw=1.4, label="coordinates $(t, r, \\theta, \\phi)$"
    )
    ax_j.semilogy(rl, em["sv_krus"].cpu().numpy(), color=ACCENT[2], lw=2, label="Kruskal $(T, X, \\theta, \\phi)$")
    ax_j.axvline(1.0, color="#9ca3af", ls="--", lw=0.8)
    ax_j.set_xlabel("radius $r / r_s$ (dimensionless)")
    ax_j.set_ylabel("$\\sigma_{min}(\\partial z/\\partial x)$ (dimensionless)")
    err = max(em["errors"].values())
    ax_j.set_title(f"Fronsdal: Schwarzschild in $\\mathbb{{R}}^{{1,5}}$, $|\\Delta g|/g$ = {err:.0e}\n"
                   "rank 4: no fold at $r_s$; $z_2\\to-\\infty$ as $r\\to0$", fontsize=9)  # fmt: skip
    ax_j.legend(fontsize=7, loc="lower right")
    cursor = ax_j.axvline(rl[-1], color="#f59e0b", lw=1.2)

    r_probe = np.geomspace(5.0, 0.03, N_FRAMES)

    def update(f: int):
        az = 30 + 9 * f
        ax_f.view_init(elev=18, azim=az)
        ax_s.cla()
        al = alpha[f]
        c, s = np.cos(al), np.sin(al)
        xs, zs = uu * c + a * np.sin(uu) * s, -uu * s + a * np.sin(uu) * c  # the sheet rotated about the v axis
        ax_s.plot_surface(xs, vv, zs, color=ACCENT[0], alpha=0.5, linewidth=0.2, edgecolor="#1e3a8a")
        ax_s.plot([-lim, lim], [0, 0], [-ZS, -ZS], color="#f59e0b", lw=1.5)  # the screen the shadow falls on
        if fo["n_fold"][f] > 0:
            for x0, u0 in zip(xf[f], uf[f]):
                z0 = -u0 * s + a * np.sin(u0) * c  # the fold line on the sheet, where the view grazes it
                ax_s.plot([x0, x0], [-2, 2], [z0, z0], color="#dc2626", lw=1.5)
                ax_s.plot([x0, x0], [0, 0], [z0, -ZS], color="#dc2626", lw=0.8, ls=":")
        ax_s.set_xlim(-lim, lim)
        ax_s.set_ylim(-2, 2)
        ax_s.set_zlim(-ZS, ZS)
        ax_s.view_init(elev=22, azim=-70 + 0.75 * f)
        ax_s.set_axis_off()
        ax_s.set_title(f"P4: a smooth sheet tilted by $\\alpha$ = {al:.2f}, seen along $z$\n"
                       f"{'folds (red)' if fo['n_fold'][f] > 0 else 'no fold yet'}: "
                       f"$a\\tan\\alpha$ = {a * np.tan(al):.2f}", fontsize=9)  # fmt: skip
        hist.set_ydata(dens[f])
        on = fo["n_fold"][f] > 0
        for line, x0 in zip(fold_lines, xf[f]):
            line.set_visible(bool(on))
            line.set_xdata([x0])
        th_h.set_text("shadow density: spikes $\\propto|x-x_f|^{-1/2}$ at folds" if on else "shadow density: smooth")
        rp = r_probe[f]
        i = np.argmin(np.abs(r - rp))
        probe.set_data([r[i]], [cu["K_exact"][i].item()])
        probe_t.set_data([r[i]], [max(pk[i], 1e-3)])
        tg.set_text(f"GR vs a fold shell at $R$ = {R:.2f} $r_s$ (same outside $R$)\n"
                    f"$r$ = {r[i]:.3f}: GR $K$ = {cu['K_exact'][i].item():.2e}, fold {pk[i]:.2e}")  # fmt: skip
        cursor.set_xdata([np.clip(rp, rl[0], rl[-1])])

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.9, wspace=0.3, hspace=0.42)
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
