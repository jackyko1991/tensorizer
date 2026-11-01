"""Animate the patterns side by side, with the dispersion relation, the linear growth check and the (F, k) map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif


def animate(
    at: torch.Tensor,
    frames: torch.Tensor,
    amp: torch.Tensor,
    pairs: list[tuple[str, float, float]],
    curves: dict,
    q: torch.Tensor,
    grid_F: torch.Tensor,
    grid_k: torch.Tensor,
    classes: torch.Tensor,
    path: Path,
) -> Path:
    """frames (T, B, n, n) of v at steps at (T,); amp (n_steps + 1, B); pairs = [(name, F, k), ...];
    curves[name] = [((u*, v*), λ(q)), ...] for every non-trivial uniform state."""
    names = [name for name, _, _ in pairs]
    steps_at, v = at.cpu().numpy(), frames.cpu().numpy()
    a = amp.cpu().numpy()
    qs = q.numpy()
    B = len(names)
    colours = [f"C{b}" for b in range(B)]

    fig = plt.figure(figsize=(10.5, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=0.32, left=0.07, right=0.98, top=0.95, bottom=0.09)
    top = outer[0].subgridspec(1, B, wspace=0.08)
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)
    ax_v = [fig.add_subplot(top[b]) for b in range(B)]
    ax_d, ax_g, ax_m = (fig.add_subplot(bottom[c]) for c in range(3))

    cmap = plt.get_cmap("magma", 10)  # a discrete colormap keeps the GIF small
    images = []
    for b in range(B):
        images.append(ax_v[b].imshow(v[0, b], cmap=cmap, vmin=0, vmax=0.45, interpolation="nearest"))
        ax_v[b].set_xticks([])
        ax_v[b].set_yticks([])
        ax_v[b].grid(False)
        for side in ax_v[b].spines.values():
            side.set_color(colours[b])
            side.set_linewidth(2)
    titles = [ax_v[b].set_title("", fontsize=9) for b in range(B)]

    # dispersion relation of every non-trivial uniform state
    turing = None
    for b, name in enumerate(names):
        for s, (_, lam) in enumerate(curves[name]):
            lam = lam.numpy()
            unstable = lam[0] < 0 < lam.max()
            ax_d.plot(qs, lam, color=colours[b], lw=2 if unstable else 1, ls="-" if s == 0 else ":")
            if unstable:
                turing = (b, lam)
                ax_d.fill_between(qs, 0, lam, where=lam > 0, color=colours[b], alpha=0.25)
    ax_d.axhline(0, color="#374151", lw=0.8)
    ax_d.set_ylim(-0.03, 0.03)
    ax_d.set_xlim(0, qs[-1])
    ax_d.set_xlabel("wavenumber q (1/cell)", labelpad=1)
    ax_d.tick_params(labelsize=7)
    ax_g.tick_params(labelsize=7)
    ax_d.set_ylabel("growth rate Re λ (1/step)", labelpad=1)
    ax_d.set_title("dispersion relation of uniform states", fontsize=9)
    ax_d.text(0.97, 0.94, "(1, 0): λ < 0 for all q", transform=ax_d.transAxes, ha="right", va="top", fontsize=7)

    # growth of the pattern amplitude: linear theory predicts exp(λ_max t) once the slower modes have died out
    t_all = np.arange(len(a))
    for b in range(B):
        ax_g.plot(t_all, a[:, b], color=colours[b], lw=1, alpha=0.35 if turing and b != turing[0] else 1)
    if turing is not None:
        b, lam = turing
        lam_max = lam.max()
        early = a[: len(a) // 2, b]
        t0 = int(early.argmin())
        t_line = np.linspace(t0, min(len(a) - 1, t0 + 6 / lam_max), 50)
        ax_g.plot(t_line, early[t0] * np.exp(lam_max * (t_line - t0)), "--", color="#374151", lw=1.2)
        label = f"dashed: exp(λ_max t)\nλ_max = {lam_max:.4f} / step"
        ax_g.text(0.04, 0.92, label, transform=ax_g.transAxes, va="top", fontsize=7)
    ax_g.set_yscale("log")
    ax_g.set_yticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])  # plain labels: no mathtext to typeset per frame
    ax_g.minorticks_off()
    ax_g.set_xlim(0, len(a) - 1)
    ax_g.set_xlabel("step", labelpad=1)
    ax_g.set_ylabel("amplitude std(v)", labelpad=1)
    ax_g.set_title("growth from a uniform state", fontsize=9)
    (marker,) = ax_g.plot([], [], "o", color="#111827", ms=4)

    # where in the (k, F) plane Turing patterns are possible
    shades = ListedColormap(["#ffffff", "#e5e7eb", "#93c5fd"])
    kk, ff = grid_k.numpy(), grid_F.numpy()
    ax_m.imshow(
        classes.numpy(),
        cmap=shades,
        vmin=0,
        vmax=2,
        origin="lower",
        aspect="auto",
        extent=(kk[0], kk[-1], ff[0], ff[-1]),
        interpolation="nearest",
    )
    k_line = np.linspace(kk[0], kk[-1], 200)
    F_line = (1 - 8 * k_line - np.sqrt(np.clip(1 - 16 * k_line, 0, None))) / 8  # F = 4 (F + k)², lower root
    ax_m.plot(k_line, F_line, color="#9ca3af", lw=1)
    ax_m.plot(k_line, (1 - 8 * k_line + np.sqrt(np.clip(1 - 16 * k_line, 0, None))) / 8, color="#9ca3af", lw=1)
    for b, (_, f, k) in enumerate(pairs):
        ax_m.plot(k, f, "o", color=colours[b], ms=6, mec="#111827")
    ax_m.set_xlim(kk[0], kk[-1])
    ax_m.set_ylim(ff[0], ff[-1])
    ax_m.set_xlabel("kill rate k", labelpad=1)
    ax_m.set_ylabel("feed rate F", labelpad=1)
    ax_m.set_title("blue: Turing-unstable, grey: other states", fontsize=9)
    ax_m.tick_params(labelsize=7)

    def update(t: int):
        for b in range(B):
            images[b].set_data(v[t, b])
            titles[b].set_text(f"{names[b]}, step {steps_at[t]}")
        i = steps_at[t]
        marker.set_data([i], [a[i, turing[0] if turing else 0]])

    return save_gif(fig, update, frame_indices(len(steps_at), 60), path, fps=8)
