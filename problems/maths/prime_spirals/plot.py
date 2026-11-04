"""Ulam and Sacks spirals growing, with the Hardy–Littlewood checks (quadratics, twins, Goldbach, gaps) alongside."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import twin_prediction

N_FRAMES = 45


def animate(d: dict, path: Path) -> Path:
    side, half = d["side"], d["side"] // 2
    ulam = d["ulam"].cpu().numpy() + half
    prime = d["prime_mask"].cpu().numpy()
    diag = d["diag_mask"].cpu().numpy()
    sacks = d["sacks"].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    (ax_u, ax_s, ax_f), (ax_t, ax_g, ax_h) = axes

    # Ulam spiral: 0 not yet drawn, 1 composite, 2 prime, 3 prime on the highlighted diagonal
    cmap = ListedColormap(["#ffffff", "#f3f4f6", "#1e3a8a", "#dc2626"])
    code = np.where(prime, np.where(diag, 3, 2), 1)
    img = np.zeros((side, side))
    im = ax_u.imshow(img, cmap=cmap, vmin=0, vmax=3, origin="lower", interpolation="nearest")
    ax_u.set_xticks([])
    ax_u.set_yticks([])
    ax_u.grid(False)
    ax_u.set_title(f"Ulam spiral from {d['start']}: red = $k^2+k+{d['start']}$", fontsize=9)

    pr = sacks[prime]
    on_diag = sacks[prime & diag]
    sc_s = ax_s.scatter(pr[:1, 0], pr[:1, 1], s=1.2, color="#1e3a8a", lw=0)
    sc_d = ax_s.scatter(on_diag[:1, 0], on_diag[:1, 1], s=4, color="#dc2626", lw=0)
    lim = math.sqrt(d["start"] + side * side)
    ax_s.set_xlim(-lim, lim)
    ax_s.set_ylim(-lim, lim)
    ax_s.set_aspect("equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.grid(False)
    ax_s.set_title(r"Sacks spiral: $n$ at $r=\sqrt{n}$, $\theta = 2\pi\sqrt{n}$", fontsize=9)
    idx_prime = np.flatnonzero(prime)
    idx_diag = np.flatnonzero(prime & diag)

    fit = d["fit"]
    alive = d["alive"].cpu().numpy()
    const = fit["constant"].cpu().numpy()[alive]
    obs = fit["observed"].cpu().numpy()[alive].astype(float)
    pred = fit["predicted"].cpu().numpy()[alive]
    ks = fit["k"].cpu().numpy()
    emp = obs / (pred / const[:, None])  # observed / Σ 1/ln f = empirical constant
    sc_f = ax_f.scatter(const, emp[:, 0], s=6, color=ACCENT[0], alpha=0.6, lw=0)
    top = const.max() * 1.1
    ax_f.plot([0, top], [0, top], color=ACCENT[1], lw=1.2, label="Conjecture F: equal")
    ax_f.set_xlim(0, top)
    ax_f.set_ylim(0, top * 1.15)
    ax_f.set_xlabel("predicted constant $C(f)$")
    ax_f.set_ylabel(r"primes / $\sum 1/\ln f(k)$")
    ax_f.legend(fontsize=7, loc="upper left")
    title_f = ax_f.set_title("", fontsize=9)

    x_max = d["x_max"]
    xs = np.unique(np.logspace(3, math.log10(x_max), 300).astype(np.int64))
    hl_t = twin_prediction(torch.tensor(xs, dtype=torch.float64), d["c2"]).numpy()
    real = d["twins"].cpu().numpy()[xs] / hl_t
    model = d["twins_cramer"].cpu().numpy()[xs] / hl_t
    (ln_real,) = ax_t.semilogx(xs[:1], real[:1], color=ACCENT[0], lw=1.5, label=r"real $\pi_2(x)$")
    (ln_model,) = ax_t.semilogx(xs[:1], model[:1], color=ACCENT[4], lw=1.2, label="Cramér model")
    ax_t.axhline(1, color=ACCENT[1], lw=1, ls="--")
    ax_t.axhline(1 / (2 * d["c2"]), color="#9ca3af", lw=1, ls=":", label=r"$1/2C_2$")
    ax_t.set_xlim(xs[0], xs[-1])
    ax_t.set_ylim(0.5, 1.3)
    ax_t.set_xlabel("$x$")
    ax_t.set_ylabel(r"count / $2C_2\int dt/\ln^2 t$")
    ax_t.legend(fontsize=7, loc="lower right")
    ax_t.set_title("twin primes vs Hardy–Littlewood", fontsize=9)

    ne, re_, hl = d["n_even"].cpu().numpy(), d["r_even"].cpu().numpy(), d["hl"].cpu().numpy()
    keep = np.unique(np.linspace(0, len(ne) - 1, min(len(ne), 4000)).astype(int))
    ne, re_, hl = ne[keep], re_[keep], hl[keep]
    band = np.where(ne % 3 == 0, ACCENT[2], "#6b7280")
    sc_g = ax_g.scatter(ne[:1], re_[:1], s=1.5, c=band[:1], lw=0)
    sc_p = ax_g.scatter(ne[:1], hl[:1], s=1.0, color=ACCENT[1], lw=0)
    ax_g.set_xlim(0, ne[-1])
    ax_g.set_ylim(0, re_.max() * 1.05)
    ax_g.set_xlabel("even $N$")
    ax_g.set_ylabel("$r(N)$ = #{p + q = N}")
    title_g = ax_g.set_title("", fontsize=9)

    primes = d["primes"].cpu().numpy()
    norm, norm_c = d["norm"].cpu().numpy(), d["norm_cramer"].cpu().numpy()
    bins = np.linspace(0, 6, 31)
    centres = 0.5 * (bins[1:] + bins[:-1])
    (st_r,) = ax_h.semilogy(
        centres, np.ones_like(centres), drawstyle="steps-mid", color=ACCENT[0], lw=1.5, label="primes"
    )
    (st_c,) = ax_h.semilogy(
        centres, np.ones_like(centres), drawstyle="steps-mid", color=ACCENT[4], lw=1, label="Cramér model"
    )
    ax_h.semilogy(centres, np.exp(-centres), color=ACCENT[1], lw=1, ls="--", label="$e^{-s}$")
    ax_h.set_ylim(1e-4, 2)
    ax_h.set_xlabel(r"normalised gap $s = g/\ln p$")
    ax_h.set_ylabel("density")
    ax_h.legend(fontsize=7, loc="upper right")
    title_h = ax_h.set_title("", fontsize=9)
    cramer_primes = d["cramer_primes"].cpu().numpy()

    def hist(v: np.ndarray) -> np.ndarray:
        h, _ = np.histogram(v, bins=bins, density=True)
        return np.where(h > 0, h, np.nan)

    def update(f: int):
        s = (f + 1) / N_FRAMES
        ring = max(1, round(half * s))
        count = (2 * ring + 1) ** 2  # numbers in the first `ring` rings
        img[:] = 0
        img[ulam[:count, 1], ulam[:count, 0]] = code[:count]
        im.set_data(img)
        sc_s.set_offsets(sacks[idx_prime[idx_prime < count]])
        sc_d.set_offsets(sacks[idx_diag[idx_diag < count]] if (idx_diag < count).any() else np.empty((0, 2)))

        j = min(len(ks) - 1, max(0, round((len(ks) - 1) * s**2)))
        sc_f.set_offsets(np.column_stack([const, emp[:, j]]))
        title_f.set_text(f"{len(const)} quadratics $4k^2+bk+c$, $k \\leq {ks[j]}$")

        m = max(2, round(len(xs) * s))
        ln_real.set_data(xs[:m], real[:m])
        ln_model.set_data(xs[:m], model[:m])

        g = max(2, round(len(ne) * s))
        sc_g.set_offsets(np.column_stack([ne[:g], re_[:g]]))
        sc_g.set_color(band[:g])
        sc_p.set_offsets(np.column_stack([ne[:g], hl[:g]]))
        title_g.set_text(f"Goldbach's comet to {ne[g - 1]:,} (red: HL prediction)")

        x_f = 10 ** (3 + (math.log10(x_max) - 3) * s)
        n_r = max(2, int(np.searchsorted(primes, x_f)))
        st_r.set_ydata(hist(norm[: n_r - 1]))
        st_c.set_ydata(hist(norm_c[: max(1, int(np.searchsorted(cramer_primes, x_f)) - 1)]))
        title_h.set_text(f"prime gaps up to $x$ = {x_f:,.0f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=5)
