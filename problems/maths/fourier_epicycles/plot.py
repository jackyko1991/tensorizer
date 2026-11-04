"""Animate the chain of rotating circles drawing the curve, with more circles at every stage, beside the error curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, save_gif

from . import equation

MAX_DRAWN = 40  # circles drawn in the chain (the rest are too small to see)


def _circles(centres: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All circles as one polyline, separated by NaN, so a single Line2D draws them."""
    a = np.linspace(0, 2 * np.pi, 49)
    xs = centres.real[:, None] + radii[:, None] * np.cos(a)  # (C, 49)
    ys = centres.imag[:, None] + radii[:, None] * np.sin(a)
    pad = np.full((len(radii), 1), np.nan)
    return np.hstack([xs, pad]).ravel(), np.hstack([ys, pad]).ravel()


def animate(z: torch.Tensor, res: dict, m_stages: list[int], name: str, path: Path, frames_per_stage: int = 12):
    target = z.cpu().numpy()
    c, k, order = res["c"], res["k"], res["order"]
    radii = c[order].abs().cpu().numpy()
    curves = res["curves"].cpu().numpy()  # (stages, T)
    n_dense = curves.shape[1]
    err, parseval = res["err"].cpu().numpy(), res["parseval"].cpu().numpy()
    n = len(err)

    t_frames = torch.linspace(0, 1, frames_per_stage, dtype=torch.float64, device=c.device)
    jobs = [(s, f) for s in range(len(m_stages)) for f in range(frames_per_stage)]
    chains = {}  # stage → (frames, M + 1) circle centres at the frame times

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(8.6, 4.3), gridspec_kw={"width_ratios": [1.05, 1]})
    closed = np.append(target, target[0])
    ax.plot(closed.real, closed.imag, color="#d1d5db", lw=3, zorder=0)
    (circ_line,) = ax.plot([], [], color="#9ca3af", lw=0.7)
    (arm_line,) = ax.plot([], [], color=ACCENT[0], lw=1.2, marker="o", ms=1.5)
    (trace,) = ax.plot([], [], color=ACCENT[1], lw=1.8)
    (pen,) = ax.plot([], [], "o", color=ACCENT[1], ms=4)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    m_axis = np.arange(1, n + 1)
    ok = m_axis < n  # with all N circles the error is 0 (round-off): left off the log axis
    ax_err.loglog(m_axis[ok], err[ok], color=ACCENT[0], lw=2, label="error at the samples")
    ax_err.loglog(m_axis[ok][::4], parseval[ok][::4], "o", ms=3, mfc="none", color=ACCENT[2],
                  label=r"Parseval: $\Sigma\,|c_k|^2$ of dropped circles")  # fmt: skip
    ax_err.set_ylim(max(err[ok].min() * 0.3, 1e-12), err[0] * 3)
    ref = m_axis[3 : n // 2].astype(float)
    if name != "heart":
        anchor = err[7] * (ref / 8.0) ** -3
        ax_err.loglog(ref, anchor, "--", color="#9ca3af", lw=1, label=r"$\propto M^{-3}$ (corners: $|c_k|\sim k^{-2}$)")
    ax_err.set_xlabel("circles kept $M$ (largest first)")
    ax_err.set_ylabel(r"mean $|z_n - S_M(t_n)|^2$")
    ax_err.set_title("reconstruction error")
    ax_err.legend(fontsize=7, loc="lower left")
    marker = ax_err.axvline(1, color="#f59e0b", lw=1.5)

    def update(job: int):
        s, f = jobs[job]
        m = m_stages[s]
        if s not in chains:
            chains[s] = equation.chain(c, k, order[:m], t_frames).cpu().numpy()  # (frames, M + 1)
        centres = chains[s][f]
        shown = min(m, MAX_DRAWN)
        cx, cy = _circles(centres[:shown], radii[:shown])
        circ_line.set_data(cx, cy)
        arm_line.set_data(centres[: shown + 1].real, centres[: shown + 1].imag)
        upto = int(round(f / (frames_per_stage - 1) * (n_dense - 1))) + 1
        trace.set_data(curves[s, :upto].real, curves[s, :upto].imag)
        pen.set_data([centres[-1].real], [centres[-1].imag])
        title.set_text(f"{name}: {m} circle{'s' if m > 1 else ''}, error {err[m - 1]:.1e}")
        marker.set_xdata([m, m])

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(jobs))), path, fps=10)
