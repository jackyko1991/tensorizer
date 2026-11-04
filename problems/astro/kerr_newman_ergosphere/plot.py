"""Animate the hole as spin and charge sweep past extremality: meridional cross-section, equatorial orbits,
the (a, Q) plane of the batched grid and the landmark radii."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

ERGO = "#fdba74"


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _meridional(r: np.ndarray, theta: np.ndarray, a: float):
    """Oblate-spheroidal drawing coordinates x = √(r² + a²) sin θ, z = r cos θ (right half-plane)."""
    return np.sqrt(r**2 + a**2) * np.sin(theta), r * np.cos(theta)


def _closed(x: np.ndarray, z: np.ndarray):
    """Mirror the right half (θ: 0 → π) to the left so the outline closes."""
    return np.concatenate([x, -x[::-1]]), np.concatenate([z, z[::-1]])


def animate(chi, a, Q, theta, sw: dict, grid: dict, angle_deg: float, path: Path) -> Path:
    chi, a, Q, th = _np(chi), _np(a), _np(Q), _np(theta)
    s = {k: _np(v) for k, v in sw.items()}
    n = len(chi)
    frames = frame_indices(n, 100)

    # particle phases on the circular orbits: φ advances by Ω Δt per frame (coordinate time)
    dt = 2.2
    phase = {}
    for key in ("isco_pro", "isco_retro", "photon_pro", "photon_retro"):
        om = np.nan_to_num(s[f"Omega_{key}"])[frames]
        phase[key] = dict(zip(frames, np.cumsum(om) * dt))

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1])
    ax_m = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[:, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_r = fig.add_subplot(gs[1, 2])

    # --- (a, Q) plane: r₊/M of the M = 1 slice of the batched grid ---
    top = float(grid["top"])
    img = np.where(grid["naked"], np.nan, grid["r_plus"])
    cmap = plt.get_cmap("viridis", 10).copy()
    cmap.set_bad("#e5e7eb")
    im = ax_q.imshow(img.T, origin="lower", extent=[0, top, 0, top], cmap=cmap, vmin=1, vmax=2, aspect="equal")
    cb = fig.colorbar(im, ax=ax_q, fraction=0.046, pad=0.03)
    cb.set_label("r₊ / M", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ring = np.linspace(0, np.pi / 2, 100)
    ax_q.plot(np.cos(ring), np.sin(ring), color="black", lw=1)
    ax_q.text(0.92 * top, 0.92 * top, "naked", ha="right", va="top", fontsize=7, color="#6b7280")
    ax_q.plot(a, Q, color=ACCENT[1], lw=1)
    here = ax_q.plot([], [], "o", color=ACCENT[1], ms=5)[0]
    ax_q.set_xlabel("a / M", fontsize=7)
    ax_q.set_ylabel("Q / M", fontsize=7)
    ax_q.tick_params(labelsize=6)
    ax_q.set_title("batched grid: horizon radius", fontsize=8)

    # --- landmark radii along the sweep ---
    curves = [
        ("r_plus", "black", "-", "r₊"),
        ("r_minus", "#6b7280", "--", "r₋"),
        ("ergo_equator", "#ea580c", "-", "ergosphere (equator)"),
        ("isco_pro", ACCENT[0], "-", "ISCO pro"),
        ("isco_retro", ACCENT[3], "-", "ISCO retro"),
        ("photon_pro", ACCENT[0], ":", "photon pro"),
        ("photon_retro", ACCENT[3], ":", "photon retro"),
    ]
    for key, color, ls, label in curves:
        ax_r.plot(chi, s[key], color=color, ls=ls, lw=1.2, label=label)
    ax_r.axvline(1.0, color="#9ca3af", lw=0.8)
    ax_r.text(1.01, 8.6, "extremal", fontsize=6, color="#6b7280")
    now = ax_r.axvline(0, color=ACCENT[1], lw=1)
    ax_r.set_xlim(0, chi[-1])
    ax_r.set_ylim(0, 9.5)
    ax_r.set_xlabel("χ = √(a² + Q²) / M", fontsize=7)
    ax_r.set_ylabel("r / M", fontsize=7)
    ax_r.tick_params(labelsize=6)
    ax_r.legend(fontsize=5, loc="upper left", ncol=2, frameon=False)

    # --- meridional cross-section: persistent artists, updated per frame ---
    ax_m.set_xlim(-2.6, 2.6)
    ax_m.set_ylim(-2.6, 2.6)
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("√(r² + a²) sin θ   (GM/c²)", fontsize=7)
    ax_m.set_ylabel("r cos θ", fontsize=7)
    ax_m.tick_params(labelsize=6)
    ax_m.text(0, -2.45, "orange: ergoregion,  red: ring singularity", ha="center", fontsize=6, color="#6b7280")
    empty = np.zeros((1, 2))
    ergo_m = [ax_m.add_patch(plt.Polygon(empty, color=ERGO)) for _ in range(2)]
    hole_m = ax_m.add_patch(plt.Polygon(empty, color="black"))
    inner_m = ax_m.plot([], [], color="white", ls="--", lw=0.8)[0]
    ring_m = ax_m.plot([], [], "o", color="#dc2626", ms=4, zorder=5)[0]

    # --- equatorial plane ---
    circle = np.linspace(0, 2 * np.pi, 200)
    ring_xy = lambda r: np.stack([r * np.cos(circle), r * np.sin(circle)], 1)  # noqa: E731
    ax_e.set_xlim(-10, 10)
    ax_e.set_ylim(-10, 10)
    ax_e.set_aspect("equal")
    ax_e.tick_params(labelsize=6)
    ax_e.set_xlabel("r cos φ  (Boyer–Lindquist r)", fontsize=7)
    ax_e.set_title("equatorial plane: ISCO (o) and photon orbits (*)\nblue prograde, purple retrograde", fontsize=8)
    ergo_e = ax_e.add_patch(plt.Polygon(empty, color=ERGO))
    hole_e = ax_e.add_patch(plt.Polygon(empty, color="black"))
    spiral = ax_e.plot([], [], color=ACCENT[2], lw=1.2, label="L = 0 infall (dragged)")[0]
    styles = {"isco_pro": (ACCENT[0], "--", "o"), "isco_retro": (ACCENT[3], "--", "o"),
              "photon_pro": (ACCENT[0], ":", "*"), "photon_retro": (ACCENT[3], ":", "*")}  # fmt: skip
    orbit_lines = {key: ax_e.plot([], [], color=c, ls=ls, lw=1)[0] for key, (c, ls, _) in styles.items()}
    particles = {key: ax_e.plot([], [], mk, color=c, ms=6)[0] for key, (c, _, mk) in styles.items()}
    naked_ring = ax_e.plot([], [], color="#dc2626", lw=1.5)[0]
    info = ax_e.text(0, -9.5, "", ha="center", fontsize=6)
    ax_e.legend(handles=[spiral], fontsize=6, loc="upper right", frameon=False)
    dk = 0.5 * (chi[1] - chi[0])

    def update(k: int):
        ak = float(a[k])
        naked = bool(s["naked"][k])
        outer, inner = s["ergo_outer"][k], s["ergo_inner"][k]
        ok = np.isfinite(outer)
        if naked:  # toroidal ergoregion between the inner and outer static limits, one ring per side
            if ok.any():
                xo, zo = _meridional(outer[ok], th[ok], ak)
                xi, zi = _meridional(inner[ok], th[ok], ak)
                for patch, sgn in zip(ergo_m, (1, -1)):
                    patch.set_xy(np.stack([sgn * np.concatenate([xo, xi[::-1]]), np.concatenate([zo, zi[::-1]])], 1))
            else:
                for patch in ergo_m:
                    patch.set_xy(empty)
            hole_m.set_xy(empty)
            inner_m.set_data([], [])
            ring_m.set_markersize(7)
            label = "naked singularity: no horizon"
        else:
            ergo_m[0].set_xy(np.stack(_closed(*_meridional(outer, th, ak)), 1))
            ergo_m[1].set_xy(empty)
            hole_m.set_xy(np.stack(_closed(*_meridional(np.full_like(th, s["r_plus"][k]), th, ak)), 1))
            inner_m.set_data(*_closed(*_meridional(np.full_like(th, s["r_minus"][k]), th, ak)))
            ring_m.set_markersize(3)
            label = "extremal black hole (r₊ = r₋)" if abs(chi[k] - 1) < dk else "black hole"
        ring_m.set_data([-ak, ak], [0, 0])
        ax_m.set_title(f"a = {ak:.2f} M,  Q = {Q[k]:.2f} M\n{label}", fontsize=8)

        rE = s["ergo_equator"][k]
        ergo_e.set_xy(ring_xy(rE) if np.isfinite(rE) else empty)
        if naked:
            hole_e.set_xy(empty)
            spiral.set_data([], [])
            for key in styles:
                orbit_lines[key].set_data([], [])
                particles[key].set_data([], [])
            naked_ring.set_data(*ring_xy(max(abs(ak), 0.05)).T)
            info.set_text("naked: weak cosmic censorship (a conjecture) forbids this")
        else:
            hole_e.set_xy(ring_xy(s["r_plus"][k]))
            r_s, phi_s = s["spiral_r"][k], s["spiral_phi"][k]
            spiral.set_data(r_s * np.cos(phi_s), r_s * np.sin(phi_s))
            for key in styles:
                r, p = s[key][k], phase[key][k]
                orbit_lines[key].set_data(*ring_xy(r).T)
                particles[key].set_data([r * np.cos(p)], [r * np.sin(p)])
            naked_ring.set_data([], [])
            info.set_text(
                f"horizon spin Ω_H = {s['omega_H'][k]:.3f},  surface gravity κ = {s['kappa'][k]:.3f}  (c³/GM)"
            )
        here.set_data([a[k]], [Q[k]])
        now.set_xdata([chi[k], chi[k]])

    fig.suptitle(f"Kerr–Newman sweep at mixing angle ψ = {angle_deg:g}° (0° Kerr, 90° Reissner–Nordström)", fontsize=9)
    fig.tight_layout()
    return save_gif(fig, update, frames + [frames[-1]] * 8, path)
