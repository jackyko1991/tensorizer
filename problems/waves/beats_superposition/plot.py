"""Top: the wave in space with its envelope and two riders (crest at v_p, envelope peak at v_g).
Bottom: the time trace at x = 0, where the beats are heard."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(x: torch.Tensor, t: torch.Tensor, result: dict, v_p: float, v_g: float, path: Path) -> Path:
    xs, ts = x.cpu().numpy(), t.cpu().numpy()
    y, env = result["y"].cpu().numpy(), result["env"].cpu().numpy()  # (N, J)
    trace, trace_env = result["trace"].cpu().numpy(), result["trace_env"].cpu().numpy()  # (N,)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.3))
    line = ax1.plot(xs, y[0], lw=1.5, color="#2563eb")[0]
    env_hi = ax1.plot(xs, env[0], "--", color="#9ca3af", lw=1)[0]
    env_lo = ax1.plot(xs, -env[0], "--", color="#9ca3af", lw=1)[0]
    crest = ax1.plot([], [], "o", color="#dc2626", ms=7, label=f"crest  v_p = {v_p:.2f} m/s")[0]
    group = ax1.plot([], [], "o", color="#16a34a", ms=7, label=f"envelope  v_g = {v_g:.2f} m/s")[0]
    ax1.set_xlim(xs[0], xs[-1])
    ax1.set_ylim(-1.15, 1.5)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    title = ax1.set_title("snapshot y(x, t)")

    trace_line = ax2.plot([], [], lw=1, color="#2563eb")[0]
    trace_env_line = ax2.plot([], [], "--", color="#9ca3af", lw=1)[0]
    ax2.set_xlim(ts[0], ts[-1])
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("y(0, t)")

    def update(step: int):
        line.set_ydata(y[step])
        env_hi.set_ydata(env[step])
        env_lo.set_ydata(-env[step])
        tn = ts[step]
        # riders start at x = 0 (a crest and the envelope peak coincide there at t = 0)
        x_c, x_g = v_p * tn, v_g * tn
        for dot, xr in ((crest, x_c), (group, x_g)):
            visible = xr <= xs[-1]
            dot.set_data([xr] if visible else [], [np.interp(xr, xs, env[step])] if visible else [])
        trace_line.set_data(ts[: step + 1], trace[: step + 1])
        trace_env_line.set_data(ts[: step + 1], trace_env[: step + 1])
        title.set_text(f"snapshot y(x, t)    t = {tn:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(ts), 80), path, fps=8)
