"""Animate the LQC bounce next to classical collapse, the effective Friedmann curve and the Wheeler–DeWitt packet."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import G, classical_volume


def animate(sol, p, lam, scales, rho_c, H, rho, wdw, path: Path) -> Path:
    t = sol["t"].cpu().numpy()
    y = sol["y"].cpu().numpy()  # (T, B, 3)
    V, phi = y[..., 0], y[..., 2]
    Hn, rhon = H.cpu().numpy(), rho.cpu().numpy()
    rc = rho_c.cpu().numpy()
    V_cl = classical_volume(sol["t"], sol["y"][0, 0, 0], p[0]).cpu().numpy()
    with np.errstate(divide="ignore"):
        rho_cl = float(p[0]) ** 2 / (2 * V_cl**2)
    alive = V_cl > 0
    x, phi_grid, psi, x_b, phi_b = wdw
    prob = (psi.abs() ** 2).cpu().numpy()
    prob /= prob.max()
    labels = [f"LQC, λ × {s:.2g}" for s in scales.cpu().numpy()]
    shades = [ACCENT[0], "#60a5fa", "#93c5fd"]

    fig, ((ax_v, ax_r), (ax_h, ax_w)) = plt.subplots(2, 2, figsize=(8.6, 5.8))
    t_rel = t - t[np.argmin(V[:, 0])]

    for j in range(V.shape[1]):
        ax_v.plot(t_rel, V[:, j], color=shades[j], lw=1.6 if j == 0 else 1, label=labels[j])
        ax_r.plot(t_rel, rhon[:, j], color=shades[j], lw=1.6 if j == 0 else 1)
    ax_v.plot(t_rel[alive], V_cl[alive], "--", color=ACCENT[1], lw=1.3, label="classical GR")
    ax_r.plot(t_rel[alive], rho_cl[alive], "--", color=ACCENT[1], lw=1.3)
    ax_v.set_yscale("log")
    for a in (ax_v, ax_r):
        a.set_xscale("asinh", linear_width=0.2)  # linear near the bounce, logarithmic far away
        a.set_xticks([-3, -1, -0.3, 0, 0.3, 1, 3], ["−3", "−1", "−0.3", "0", "0.3", "1", "3"])
    ax_v.set_xlabel("proper time $t - t_B$ ($t_P$)")
    ax_v.set_ylabel("volume $V$ ($\\ell_P^3$)")
    ax_v.legend(fontsize=7, loc="upper center")
    ax_v.set_title("Big Bang → Big Bounce", fontsize=10)
    for j, r in enumerate(rc):
        ax_r.axhline(r, color=shades[j], ls=":", lw=1)
    ax_r.set_yscale("log")
    ax_r.set_ylim(rhon.min() * 0.5, rc.max() * 30)
    ax_r.set_xlabel("$t - t_B$ ($t_P$)")
    ax_r.set_ylabel("density $\\rho$ ($\\rho_P$)")
    ax_r.set_title("ρ stops at $\\rho_c$ (dotted); GR diverges", fontsize=10)
    dot_v = ax_v.plot([], [], "o", color="#111827", ms=5)[0]
    dot_r = ax_r.plot([], [], "o", color="#111827", ms=5)[0]

    # effective Friedmann: H² against ρ (classical straight line vs LQC parabolas)
    r_line = np.geomspace(rhon.min(), rc.max(), 300)
    ax_h.plot(r_line, 8 * math.pi * G / 3 * r_line, "--", color=ACCENT[1], lw=1.3, label="GR: $H^2=8\\pi G\\rho/3$")
    for j, r in enumerate(rc):
        rr = np.geomspace(rhon.min(), r, 400)
        ax_h.plot(rr, 8 * math.pi * G / 3 * rr * (1 - rr / r), color=shades[j], lw=1.6 if j == 0 else 1)
    ax_h.set_xscale("log")
    ax_h.set_xlim(rhon.min(), rc.max() * 1.2)
    ax_h.set_ylim(0, 8 * math.pi * G / 3 * rc[0] / 4 * 2.2)  # the full-gap parabola peaks at 2πGρ_c/3
    ax_h.set_xlabel("$\\rho$ ($\\rho_P$)")
    ax_h.set_ylabel("$H^2$ ($t_P^{-2}$)")
    ax_h.legend(fontsize=7, loc="upper left")
    ax_h.set_title("$H^2 = \\frac{8\\pi G}{3}\\rho(1-\\rho/\\rho_c)$ (simulated •)", fontsize=10)
    dot_h = ax_h.plot([], [], "o", color="#111827", ms=5)[0]

    # Wheeler–DeWitt |Ψ|² in the (φ, ln V) plane vs the effective LQC trajectory
    xs, ps = x.cpu().numpy(), phi_grid.cpu().numpy()
    ax_w.imshow(prob.T, origin="lower", aspect="auto", extent=(ps[0], ps[-1], xs[0], xs[-1]),
                cmap=plt.get_cmap("Greys", 6), vmin=0, vmax=1.2, interpolation="nearest")  # fmt: skip
    ax_w.plot(phi[:, 0], np.log(V[:, 0]), color=ACCENT[0], lw=1.6, label="LQC effective (bounces)")
    cl = x_b - math.log(2) - math.sqrt(12 * math.pi * G) * (ps - phi_b)
    ax_w.plot(ps, cl, "--", color=ACCENT[1], lw=1.2, label="GR contracting branch")
    ax_w.set_ylim(xs.min(), xs.max())
    ax_w.set_xlabel("scalar field clock $\\varphi$")
    ax_w.set_ylabel("$\\ln V$")
    ax_w.legend(fontsize=7, loc="upper center")
    ax_w.set_title("grey: Wheeler–DeWitt $|\\Psi(\\varphi, \\ln V)|^2$", fontsize=10)
    ax_w.grid(False)
    dot_w = ax_w.plot([], [], "o", color="#111827", ms=5)[0]
    label = fig.text(0.5, 0.005, "", ha="center", fontsize=9)

    def update(k: int):
        dot_v.set_data([t_rel[k]], [V[k, 0]])
        dot_r.set_data([t_rel[k]], [rhon[k, 0]])
        dot_h.set_data([rhon[k, 0]], [Hn[k, 0] ** 2])
        dot_w.set_data([phi[k, 0]], [np.log(V[k, 0])])
        phase = "contracting" if Hn[k, 0] < 0 else "expanding"
        label.set_text(
            f"t − t_B = {t_rel[k]:+.3f} t_P   ρ/ρ_c = {rhon[k, 0] / rc[0]:.4f}   H = {Hn[k, 0]:+.3f}  ({phase})"
        )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save_gif(fig, update, _frames(t_rel), path)


def _frames(t_rel: np.ndarray) -> list[int]:
    """Frames evenly spaced in asinh(t/t_B-width): dense near the bounce where everything happens."""
    u = np.arcsinh(t_rel / 0.2)
    targets = np.linspace(u[0], u[-1], len(frame_indices(len(u), 50)))
    return [int(i) for i in np.searchsorted(u, targets).clip(0, len(u) - 1)]
