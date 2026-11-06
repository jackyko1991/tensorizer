"""The animation: for each target, the data evolving next to a re-simulation of the equation found, the STRidge
tolerance sweep pruning the library (term shares shrinking to zero), the noise study, and the equation as it is
written. One GIF per target; a static summary of the four equations found from 1 % noisy data."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data
from problems._common.progress import report
from problems._common.runtime import fast_mode

from . import solve
from . import tensors as T

METHODS = {  # colour, line style, label
    "fd": (ACCENT[1], "-", "finite differences"),
    "sg": (ACCENT[4], "-", "Savitzky–Golay"),
    "weak": (ACCENT[0], "-", "weak form"),
    "gate-weak": (ACCENT[2], "--", "annealed gates (weak)"),
}
TITLES = {
    "lorenz": "Lorenz system",
    "heat": "heat equation",
    "burgers": "viscous Burgers",
    "maxwell": "Maxwell (2D TM)",
}
SHARE_LIM = (1e-5, 3.0)
ERR_LIM = (1e-7, 300.0)
SWEEP, HOLD = 14, 6  # frames per GIF: tolerance sweep, then the final equation (last frame held 4 s)


def save_gif_shared_palette(fig, update, frames: list, path: Path, fps: int = 5, durations=None) -> Path:
    """Every frame quantised with ONE palette built from a strip of frames, so colours do not drift between frames."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=durations or int(1000 / fps), loop=0)
    return tmp.replace(path)


def _short(line: str, n_terms: int = 6) -> str:
    """Keep the first n terms of a long written equation."""
    lhs, rest = line.split(" = ", 1)
    parts = re.split(r" (?=[+-] )", rest)
    if len(parts) <= n_terms:
        return line
    return f"{lhs} = " + " ".join(parts[:n_terms]) + f" + \\ldots\\ ({len(parts) - n_terms}\\ more)"


def _wrap(line: str, per_line: int) -> list[str]:
    """Break a written equation after every per_line terms (continuation lines start with the sign)."""
    lhs, rest = line.split(" = ", 1)
    parts = re.split(r" (?=[+-] )", rest)
    chunks = [" ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)]
    return [f"{lhs} = {chunks[0]}"] + [f"\\qquad {c}" for c in chunks[1:]]


def constants(name: str, terms, xi) -> str:
    """The physical constants read off the found coefficients."""
    n = [tm.latex for tm in terms]
    x = xi.numpy()
    if name == "lorenz":
        return (f"$\\sigma$ = {(x[0, n.index('y')] - x[0, n.index('x')]) / 2:.4f} (10),  "
                f"$\\rho$ = {x[1, n.index('x')]:.3f} (28),  $\\beta$ = {-x[2, n.index('z')]:.4f} (8/3)")  # fmt: skip
    if name == "heat":
        return f"$D$ = {x[0, n.index('u_{xx}')]:.5f} cm$^2$/s  (true {T.HEAT_D})"
    if name == "burgers":
        return f"$\\nu$ = {x[0, n.index('u_{xx}')]:.5f} m$^2$/s  (true {T.BURGERS_NU})"
    c2 = (x[0, n.index("\\partial_x B_y")] - x[0, n.index("\\partial_y B_x")]) / 2
    return f"$c$ = {np.sqrt(max(c2, 0)):.3f} cm/ns  (true {T.C_LIGHT:.3f})"


def _data_panel(fig, spec, ds, sim):
    """Build the data panel once; returns update(k) that moves it to time index k."""
    name, U, S = ds["name"], ds["U"].numpy(), sim.numpy()
    t = ds["coords"][-1].numpy()
    if name == "maxwell":
        sub = spec.subgridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.15)
        vmax = float(np.abs(U[0]).max())
        x = ds["coords"][0].numpy()
        ext = [x[0], x[-1], x[0], x[-1]]
        ims, axes = [], []
        for j, lab in enumerate(("data (Yee FDTD)", "found equation")):
            ax = fig.add_subplot(sub[j])
            ims.append(ax.imshow(U[0, :, :, 0].T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax))
            ax.set_xlabel("$x$ (cm)", fontsize=10)
            if j == 0:
                ax.set_ylabel("$y$ (cm)", fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=10)
            axes.append((ax, lab))
        cb = fig.colorbar(ims[0], cax=fig.add_subplot(sub[2]))
        cb.set_label("$E_z$ (a.u.)", fontsize=10)
        cb.ax.tick_params(labelsize=9)

        def update(k):
            for im, A, (ax, lab) in zip(ims, (U, S), axes):
                im.set_data(A[0, :, :, k].T)
                ax.set_title(f"{lab}, t = {t[k] * 1000:.0f} ps", fontsize=11)

        return update
    ax = fig.add_subplot(spec)
    ax.tick_params(labelsize=10)
    if name == "lorenz":
        ax.plot(U[0, 0], U[2, 0], color="#d1d5db", lw=0.8)
        ld = ax.plot([], [], color=ACCENT[0], lw=1.6, label="data")[0]
        lf = ax.plot([], [], color=ACCENT[4], lw=1.4, ls="--", label="found equation")[0]
        both = np.concatenate([U[[0, 2], 0], S[[0, 2], 0]], 1)
        lo, hi = both.min(1), both.max(1)
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel("$x$ (a.u.)", fontsize=10)
        ax.set_ylabel("$z$ (a.u.)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

        def update(k):
            ld.set_data(U[0, 0, : k + 1], U[2, 0, : k + 1])
            lf.set_data(S[0, 0, : k + 1], S[2, 0, : k + 1])
            ax.set_title(f"one trajectory, $t$ = {t[k]:.2f} (a.u.)", fontsize=11)

        return update
    x = ds["coords"][0].numpy()
    unit, xu = ("K", "cm") if name == "heat" else ("m/s", "m")
    ax.plot(x, U[0, :, 0], color="#d1d5db", lw=1.0, label="$t = 0$")
    ld = ax.plot(x, U[0, :, 0], color=ACCENT[0], lw=2.0, label="data")[0]
    lf = ax.plot(x, S[0, :, 0], color=ACCENT[4], lw=1.6, ls="--", label="found equation")[0]
    lo, hi = float(U.min()), float(U.max())
    pad = 0.08 * (hi - lo)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"$x$ ({xu})", fontsize=10)
    ax.set_ylabel(f"$u$ ({unit})", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    def update(k):
        ld.set_ydata(U[0, :, k])
        lf.set_ydata(S[0, :, k])
        ax.set_title(f"$t$ = {t[k]:.2f} s", fontsize=11)

    return update


def _bars_panel(fig, spec, ds, r):
    """Build the bar panel once; returns update(i) that shows tolerance step i of the STRidge sweep."""
    E, K = r["share"].shape[1:]
    sub = spec.subgridspec(E, 1, hspace=0.12)
    names = [f"${tm.latex}$" for tm in ds["terms"]]
    truth = ds["truth"].numpy() != 0
    base = SHARE_LIM[0] * 0.5
    bars, top = [], None
    for e in range(E):
        ax = fig.add_subplot(sub[e])
        bars.append(ax.bar(np.arange(K), np.zeros(K), bottom=base, width=0.7, color=ACCENT[0]))
        ax.scatter(np.where(truth[e])[0], np.full(truth[e].sum(), SHARE_LIM[0] * 1.8), marker="^", color="#111827",
                   s=22, zorder=3)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(*SHARE_LIM)
        ax.set_xlim(-0.6, K - 0.4)
        ax.set_yticks([1e-4, 1e-2, 1])
        ax.minorticks_off()
        ax.tick_params(labelsize=9)
        if E == 1:
            ax.set_ylabel(f"share in ${ds['lhs'][e]}$ (dimensionless)", fontsize=10)
        else:  # one shared axis label for the stack, the equation named inside each row
            ax.text(0.99, 0.9, f"${ds['lhs'][e]}$", transform=ax.transAxes, ha="right", va="top", fontsize=12,
                    bbox={"fc": "white", "ec": "#9ca3af", "boxstyle": "round,pad=0.2"})  # fmt: skip
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(names if e == E - 1 else [], rotation=90, fontsize=9 if K > 14 else 10)
        top = top or ax
    if E > 1:
        box = spec.get_position(fig)
        fig.text(box.x0 - 0.042, (box.y0 + box.y1) / 2, "share (dimensionless)", rotation=90, ha="center",
                 va="center", fontsize=10)  # fmt: skip

    def update(i):
        kept = 0
        for e in range(E):
            sh = r["share"][min(i, int(r["pick"][e])), e].numpy()
            kept += int((sh != 0).sum())
            for rect, v in zip(bars[e], sh):
                rect.set_height(abs(v) - base if v != 0 else 0.0)
                rect.set_color(ACCENT[0] if v >= 0 else ACCENT[1])
        tol = float(solve.TOLS[min(i, int(r["pick"].max()))])
        top.set_title(f"term share $|\\xi_k|\\,\\|\\Theta_k\\|/\\|\\partial_t u\\|$ (dimensionless): STRidge "
                      f"$\\tau$ = {tol:.0e}, {kept} terms kept", fontsize=11)  # fmt: skip

    return update


def _noise_panel(fig, spec, rows, name: str, levels):
    ax = fig.add_subplot(spec)
    xs = np.arange(len(levels))
    for m, (col, ls, lab) in METHODS.items():
        rr = [r for r in rows if r["target"] == name and r["method"] == m]
        if not rr:
            continue
        err = np.array([max(r["coef_err"] * 100, ERR_LIM[0]) for r in rr])
        ok = np.array([r["success"] for r in rr])
        ax.plot(xs, err, color=col, ls=ls, lw=1.6, label=lab)
        ax.scatter(xs[ok], err[ok], color=col, s=40, zorder=3)
        ax.scatter(xs[~ok], err[~ok], color=col, marker="x", s=55, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*ERR_LIM)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lv * 100:g}" for lv in levels])
    ax.set_xlim(-0.3, len(levels) - 0.7)
    ax.set_xlabel("noise (% of the field's std)", fontsize=10)
    ax.set_ylabel("largest coefficient error (%)", fontsize=10)
    ax.set_title("noise study: dot = right terms, × = wrong terms", fontsize=11)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)


def _summary(fig, datasets, rows, levels):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.95, "Equations found from data with 1 % noise (weak form + STRidge, ± bootstrap)",
            ha="center", fontsize=15, weight="bold")  # fmt: skip
    y = 0.86
    for tgt in datasets:
        r = next((q for q in rows if q["target"] == tgt and q["method"] == "weak" and q["noise"] == 0.01),
                 None) or next(q for q in rows if q["target"] == tgt and q["method"] == "weak")  # fmt: skip
        ax.text(0.04, y, f"{TITLES[tgt]}  ({'right terms' if r['success'] else 'wrong terms'};  "
                f"{constants(tgt, r['terms'], r['xi'])})", fontsize=12, color="#374151")  # fmt: skip
        y -= 0.055
        for line in r["lines"]:
            ax.text(0.07, y, f"${_short(line, 4)}$", fontsize=12.5)
            y -= 0.05
        y -= 0.02
    verdict = []
    for m in ("fd", "sg", "weak", "gate-weak"):
        ok = [tgt for tgt in datasets if any(q["target"] == tgt and q["method"] == m and q["noise"] == max(levels)
                                             and q["success"] for q in rows)]  # fmt: skip
        verdict.append(f"{METHODS[m][2]}: {', '.join(ok) if ok else 'none'}")
    ax.text(0.04, y - 0.01, f"right terms at {max(levels) * 100:g} % noise:", fontsize=12, color=ACCENT[1])
    ax.text(0.07, y - 0.06, ";   ".join(verdict), fontsize=12, color=ACCENT[1])


def animate_target(name: str, ds: dict, rows: list[dict], r: dict, sim, path: Path) -> Path:
    """One GIF for one target: the data evolving next to the re-simulated found equation, the STRidge sweep pruning
    the library, the noise study, and (bottom right) the equation written by the finder; the final frame is held."""
    levels = sorted({q["noise"] for q in rows})
    sweep, final_frames = (3, 1) if fast_mode() else (SWEEP, HOLD)
    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1.1, 1], left=0.06, right=0.98,
                          top=0.91, bottom=0.07, wspace=0.3, hspace=0.5)  # fmt: skip
    fig.suptitle(f"{TITLES[name]}: find $\\partial_t u = \\Theta(u)\\,\\xi$ with sparse $\\xi$ "
                 f"(clean data, finite differences)", fontsize=14, weight="bold")  # fmt: skip
    data = _data_panel(fig, gs[0, 0], ds, sim)
    bars = _bars_panel(fig, gs[0, 1], ds, r)
    _noise_panel(fig, gs[1, 0], rows, name, levels)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_t, n_f = ds["U"].shape[-1], sweep + final_frames
    truth = [solve.write_equation(lhs, ds["terms"], ds["truth"][e], digits=4) for e, lhs in enumerate(ds["lhs"])]

    def update(f):
        k = round(f / (n_f - 1) * (n_t - 1))
        i = round(min(f, sweep - 1) / max(sweep - 1, 1) * int(r["pick"].max()))
        data(k)
        bars(i)
        for txt in list(ax.texts):
            txt.remove()
        final = f >= sweep
        ax.set_title("the equation found" + (" (final, ± bootstrap)" if final else " (pruning…)"), fontsize=12,
                     weight="bold" if final else "normal")  # fmt: skip
        if final:
            lines = r["lines"]
        else:
            lines = [solve.write_equation(lhs, ds["terms"], r["path"][min(i, int(r["pick"][e])), e], digits=3)
                     for e, lhs in enumerate(ds["lhs"])]  # fmt: skip
        y = 0.98
        for line in lines:
            for part in _wrap(_short(line, 5), 2 if final else 3):
                ax.text(0.0, y, f"${part}$", fontsize=12.5, va="top", transform=ax.transAxes)
                y -= 0.11
            y -= 0.02
        if final:
            ax.text(0.0, y, constants(name, ds["terms"], r["xi"]), fontsize=12, va="top", color=ACCENT[2],
                    transform=ax.transAxes)  # fmt: skip
            y -= 0.12
            ax.text(0.0, y, "true equation:", fontsize=10.5, va="top", color="#6b7280", transform=ax.transAxes)
            for t in truth:
                y -= 0.085
                ax.text(0.03, y, f"${t}$", fontsize=10.5, va="top", color="#6b7280", transform=ax.transAxes)

    durations = [200] * (n_f - 1) + [4000]  # the finished equation stays 4 s before the loop
    return save_gif_shared_palette(fig, update, list(range(n_f)), path, durations=durations)


def summary_png(datasets: dict, rows: list[dict], path: Path) -> Path:
    """Static summary: the four equations found from 1 % noisy data, and who survives 5 % noise."""
    fig = plt.figure(figsize=(13.5, 7.6), dpi=80)
    _summary(fig, datasets, rows, sorted({q["noise"] for q in rows}))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def animate(datasets: dict, rows: list[dict], show: dict, sims: dict, out_dir: Path, main: str = "maxwell") -> Path:
    """results/<target>.gif for every target except the main one, which is the problem's main GIF; plus
    results/summary.png."""
    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    main_path = out_dir / "equation_finder.gif"
    for name in datasets:
        path = main_path if name == main else res / f"{name}.gif"
        animate_target(name, datasets[name], rows, show[name], sims[name], path)
    summary_png(datasets, rows, res / "summary.png")
    return main_path
