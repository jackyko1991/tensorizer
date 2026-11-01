"""Two Minkowski diagrams of the twin trip: the home frame S and a boosted frame S'."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import transform
from .solve import clock_ticks, simultaneity_line

HOME, TRAV, NOW = ACCENT[0], ACCENT[1], "#f59e0b"


def animate(sol: dict, v: float, beta_view: float, path: Path) -> Path:
    home_S, trav_S = sol["S"]["home"], sol["S"]["traveller"]
    tau_h, tau_t = sol["tau_home"], sol["tau_trav"]
    Lam = sol["Lam"]
    N = len(tau_h)
    T = float(home_S[-1, 0])

    ticks = {  # proper-time ticks every year, in S and in S'
        "home": clock_ticks(home_S, tau_h),
        "traveller": clock_ticks(trav_S, tau_t),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    panels = []
    for ax, frame in zip(axes, ["S", "S'"]):
        data = sol[frame]
        world = np.concatenate([data["home"].numpy(), data["traveller"].numpy()])
        x_lo, x_hi = world[:, 1].min() - 1.5, world[:, 1].max() + 1.5
        t_hi = world[:, 0].max() + 0.5
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(min(0.0, world[:, 0].min()) - 0.5, t_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        for line in data["grid"].numpy():  # the home frame's coordinate lines t = const, x = const
            ax.plot(line[:, 1], line[:, 0], color="#e5e7eb", lw=0.6, zorder=0)
        s = np.array([-40.0, 0.0, 40.0])  # light cone through the departure event: unchanged by any boost
        ax.plot(s, np.abs(s), color="#9ca3af", ls=(0, (4, 3)), lw=1)
        ax.set_xlabel(f"x{'′' if frame != 'S' else ''} (light-years)")
        ax.set_ylabel(f"t{'′' if frame != 'S' else ''} (years)")
        mats = Lam if frame != "S" else torch.eye(2, dtype=torch.float64)
        tick_h = transform(mats, ticks["home"]).numpy()
        tick_t = transform(mats, ticks["traveller"]).numpy()
        panels.append(
            {
                "ax": ax,
                "mats": mats,
                "home": data["home"].numpy(),
                "trav": data["traveller"].numpy(),
                "tick_h": tick_h,
                "tick_t": tick_t,
                "line_h": ax.plot([], [], color=HOME, lw=2, label="home twin")[0],
                "line_t": ax.plot([], [], color=TRAV, lw=2, label="traveller")[0],
                "dots_h": ax.plot([], [], "o", ms=3.5, color=HOME)[0],
                "dots_t": ax.plot([], [], "o", ms=3.5, color=TRAV)[0],
                "now": ax.plot([], [], color=NOW, lw=1.5, label="traveller's “now”")[0],
            }
        )
    axes[0].set_title("home frame S   (dots: one year of proper time)", fontsize=9)
    axes[1].set_title(f"frame S′ boosted by β = {beta_view:+.2f}:  x′ = Λx", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.95)
    ages = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

    def update(k: int):
        t_now = float(home_S[k, 0])
        v_now = v if t_now < T / 2 else -v
        now_S = simultaneity_line(trav_S[k], v_now)
        for p in panels:
            p["line_h"].set_data(p["home"][: k + 1, 1], p["home"][: k + 1, 0])
            p["line_t"].set_data(p["trav"][: k + 1, 1], p["trav"][: k + 1, 0])
            n_h, n_t = int(tau_h[k]), int(tau_t[k] + 1e-9)
            p["dots_h"].set_data(p["tick_h"][:n_h, 1], p["tick_h"][:n_h, 0])
            p["dots_t"].set_data(p["tick_t"][:n_t, 1], p["tick_t"][:n_t, 0])
            now = transform(p["mats"], now_S).numpy()
            p["now"].set_data(now[:, 1], now[:, 0])
        ages.set_text(
            f"t = {t_now:4.1f} yr    home twin aged {tau_h[k]:4.2f} yr    traveller aged {tau_t[k]:4.2f} yr"
            f"    (∫√(1−v²) dt)"
        )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    frames = frame_indices(N, 90) + [N - 1] * 15  # hold the reunion for a moment
    return save_gif(fig, update, frames, path)
