"""Animate the precessing orbit (vs Newton), the effective potential, and a radial plunge in τ and in t."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import effective_potential, isco_radius, photon_sphere_radius


def animate(orbit: dict, plunge: dict, M: float, path: Path) -> Path:
    xyz = orbit["gr"][:, 0, 0].cpu().numpy()  # (T, 4) = (t, r, θ, φ)
    r, phi = xyz[:, 1], xyz[:, 3]
    gr = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)  # (T, 2)
    nw = orbit["newton"][:, 0, 0].cpu().numpy()  # (Tn, 2)
    E, L = orbit["E"], orbit["L"]
    tau_p = plunge["tau"].cpu().numpy()
    t_p, r_p = plunge["traj"][:, 0, 0, 0].cpu().numpy(), plunge["traj"][:, 0, 0, 1].cpu().numpy()

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # --- orbit in the equatorial plane ---
    reach = 1.1 * r.max()
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=4))
    ax.add_patch(plt.Circle((0, 0), photon_sphere_radius(M), fill=False, ls=":", color="#f59e0b", lw=1.2))
    ax.add_patch(plt.Circle((0, 0), isco_radius(M), fill=False, ls="--", color="#9ca3af", lw=1))
    ax.text(0, -isco_radius(M) - 2.5, "ISCO 6M", ha="center", fontsize=7, color="#6b7280")
    ax.plot(nw[:, 0], nw[:, 1], color="#9ca3af", lw=1, label="Newton (closed ellipse)")
    trail = ax.plot([], [], color=ACCENT[0], lw=1.3, label="geodesic (GR)")[0]
    dot = ax.plot([], [], "o", color=ACCENT[0], ms=5)[0]
    peri = ax.plot([], [], "x", color=ACCENT[1], ms=8, mew=2, label="periapsis")[0]
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("x  (GM/c²)")
    ax.set_ylabel("y  (GM/c²)")
    dphi = orbit["precession"]
    ax.set_title(f"perihelion advance {np.degrees(dphi):.0f}°/orbit (weak field {orbit['weak_deg']:.0f}°)", fontsize=9)

    # --- effective potential ---
    rr = torch.linspace(2.2 * M, 1.3 * r.max(), 400, dtype=torch.float64)
    ax_v.plot(rr, effective_potential(rr, L, M), color=ACCENT[0], lw=1.5, label="GR")
    ax_v.plot(rr, effective_potential(rr, L, M, relativistic=False), color="#9ca3af", lw=1, label="Newton")
    L_isco = (12**0.5) * M
    ax_v.plot(rr, effective_potential(rr, L_isco, M), color=ACCENT[3], lw=1, ls="--", label="L = √12 M (ISCO)")
    level = 0.5 * (E**2 - 1)
    ax_v.axhline(level, color=ACCENT[1], lw=1)
    ax_v.axvline(isco_radius(M), color="#9ca3af", ls="--", lw=0.8)
    ax_v.axvline(photon_sphere_radius(M), color="#f59e0b", ls=":", lw=1)
    ax_v.set_ylim(level - 0.06, level + 0.08)
    ax_v.set_xlim(rr[0], rr[-1])
    ax_v.set_ylabel("V(r)")
    ax_v.legend(fontsize=6, loc="lower right")
    marker = ax_v.plot([], [], "o", color=ACCENT[0], ms=5)[0]

    # --- radial plunge: proper time vs coordinate time ---
    ax_p.axhline(2 * M, color="black", lw=1.2)
    ax_p.text(t_p[-1] * 0.98, 2.2 * M, "horizon r = 2M", ha="right", fontsize=7)
    ax_p.set_xlim(0, t_p[-1] * 1.05)
    ax_p.set_ylim(0, r_p[0] * 1.1)
    ax_p.set_xlabel("time  (GM/c³)")
    ax_p.set_ylabel("r")
    ax_p.axvline(plunge["tau_horizon"], color=ACCENT[2], ls=":", lw=1)
    run_tau = ax_p.plot([], [], color=ACCENT[2], lw=1.8, label="r(τ) proper time: finite")[0]
    run_t = ax_p.plot([], [], color=ACCENT[4], lw=1.8, label="r(t) far observer: freezes")[0]
    ax_p.legend(fontsize=6, loc="upper right")

    n_orbit, n_plunge = len(r), len(tau_p)
    frames = frame_indices(n_orbit, 100)
    peri_xy = gr[orbit["peri_steps"]]

    def update(step: int):
        trail.set_data(gr[: step + 1, 0], gr[: step + 1, 1])
        dot.set_data([gr[step, 0]], [gr[step, 1]])
        shown = orbit["peri_steps"] <= step
        peri.set_data(peri_xy[shown, 0], peri_xy[shown, 1])
        marker.set_data([r[step]], [level])
        k = min(n_plunge - 1, round(step / max(1, n_orbit - 1) * 1.6 * (n_plunge - 1)))  # plunge finishes early
        run_tau.set_data(tau_p[: k + 1], r_p[: k + 1])
        run_t.set_data(t_p[: k + 1], r_p[: k + 1])

    fig.tight_layout()
    return save_gif(fig, update, frames, path)
