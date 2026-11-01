"""Animate the chain that ends lowest as it folds, the energy against Monte Carlo step, and its contact map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif

from .equation import contact_map, squared_distances

# contact map colours: nothing, chain bond, contact with a P residue, H–H contact
CMAP = ListedColormap(["#ffffff", "#e5e7eb", "#93c5fd", "#1e40af"])


def animate(out: dict, r: int, seq: str, e_ground: int, path: Path) -> Path:
    snaps = out["snapshots"][:, r]  # (K, N, 2) replica r at every recorded step
    maps = _contact_levels(snaps, seq)  # (K, N, N)
    chains = snaps.cpu().numpy()
    chains = chains - np.round(chains.mean(axis=1, keepdims=True))  # keep the chain in view
    energies = out["energies"].cpu().numpy()  # (K, R)
    step = out["record_steps"].numpy()
    is_h = np.array([c == "H" for c in seq])
    n = len(seq)

    fig, (ax_c, ax_e, ax_m) = plt.subplots(1, 3, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    # --- the chain on the lattice ---
    half = np.maximum(np.abs(chains).max(axis=(1, 2)) + 1, 4)  # (K,) zoom on the chain in every frame
    ax_c.set_aspect("equal")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    for g in np.arange(-n, n + 1):  # the square lattice, drawn once
        ax_c.axhline(g, color="#e5e7eb", lw=0.6, zorder=0)
        ax_c.axvline(g, color="#e5e7eb", lw=0.6, zorder=0)
    (contacts,) = ax_c.plot([], [], "--", color="#dc2626", lw=1.5)
    (backbone,) = ax_c.plot([], [], "-", color="#374151", lw=2)
    (h_dots,) = ax_c.plot([], [], "o", ms=7, color="#2563eb", mec="#1e40af", label="H")
    (p_dots,) = ax_c.plot([], [], "o", ms=7, color="white", mec="#374151", label="P")
    ax_c.legend(loc="upper right", fontsize=7, handletextpad=0.2, borderpad=0.3)
    title = ax_c.set_title("step 0, E = 0", fontsize=10)  # real text so tight_layout leaves room

    # --- energy against step ---
    ax_e.plot(step, energies.mean(axis=1), color="#9ca3af", lw=1.2, label="mean of replicas")
    ax_e.plot(step, energies.min(axis=1), color="#2563eb", lw=1.2, label="lowest replica")
    ax_e.plot(step, energies[:, r], color="#ea580c", lw=1.2, label="replica shown")
    ax_e.axhline(e_ground, color="#16a34a", ls="--", lw=1, label=f"ground state {e_ground}")
    (marker,) = ax_e.plot([], [], "o", color="#ea580c", ms=5)
    ax_e.set_ylim(e_ground - 1, 0.5)
    ax_e.set_xlabel("Monte Carlo step (T falls →)")
    ax_e.set_ylabel(r"energy $E$ (units of $\varepsilon$)")
    ax_e.legend(fontsize=7, loc="upper right")
    ax_e.set_title("simulated annealing", fontsize=10)

    # --- contact map C_ij ---
    image = ax_m.imshow(maps[0], cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    ticks = [i for i in range(0, n, 5)]
    ax_m.set_xticks(ticks)
    ax_m.set_yticks(ticks)
    ax_m.set_xlabel("residue $j$")
    ax_m.set_ylabel("residue $i$", labelpad=1)
    ax_m.grid(False)
    ax_m.set_title("contact map (dark: H–H)", fontsize=10)

    def update(k: int):
        c = chains[k]
        ax_c.set_xlim(-half[k] - 0.5, half[k] + 0.5)
        ax_c.set_ylim(-half[k] - 0.5, half[k] + 0.5)
        size = float(np.clip(50 / half[k], 3, 10))
        h_dots.set_markersize(size)
        p_dots.set_markersize(size)
        backbone.set_data(c[:, 0], c[:, 1])
        h_dots.set_data(c[is_h, 0], c[is_h, 1])
        p_dots.set_data(c[~is_h, 0], c[~is_h, 1])
        i, j = np.nonzero(np.triu(maps[k] == 3))
        xs = np.stack([c[i, 0], c[j, 0], np.full(len(i), np.nan)], axis=1).ravel()
        ys = np.stack([c[i, 1], c[j, 1], np.full(len(i), np.nan)], axis=1).ravel()
        contacts.set_data(xs, ys)
        title.set_text(f"step {step[k]}, E = {energies[k, r]:.0f}")
        marker.set_data([step[k]], [energies[k, r]])
        image.set_data(maps[k])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(step), 60), path, fps=8)


def _contact_levels(snaps, seq: str) -> np.ndarray:
    """Per snapshot an (N, N) map: 1 chain bond, 2 contact with a P residue, 3 H–H contact."""
    C = contact_map(squared_distances(snaps)).cpu().numpy()  # (K, N, N), the snapshots act as a batch
    is_h = np.array([c == "H" for c in seq])
    n = len(seq)
    idx = np.arange(n)
    levels = np.where(np.abs(idx[:, None] - idx[None, :]) == 1, 1, 0)[None].repeat(len(C), 0)
    levels = np.where(C == 1, 2, levels)
    return np.where((C == 1) & (is_h[:, None] & is_h[None, :]), 3, levels)
