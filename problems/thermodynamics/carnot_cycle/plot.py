"""Animate a point going round the p–V diagram (area filling in) and the T–S diagram side by side."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from problems._common.plotting import ACCENT, frame_indices, save_gif

LEG_NAMES = ["hot isotherm", "adiabatic expansion", "cold isotherm", "adiabatic compression"]


def animate(legs: torch.Tensor, T: torch.Tensor, S: torch.Tensor, title: str, path: Path) -> Path:
    n_legs, K = legs.shape[:2]
    V = 1e3 * legs[..., 0].numpy()  # (4, K) litres
    p = 1e-3 * legs[..., 1].numpy()  # (4, K) kPa
    T, S = T.numpy(), S.numpy()  # (4, K) each
    V_path, p_path = V.reshape(-1), p.reshape(-1)  # (4K,) the whole cycle in order
    T_path, S_path = T.reshape(-1), S.reshape(-1)

    fig, (ax_pv, ax_ts) = plt.subplots(1, 2, figsize=(7, 4))
    fig.suptitle(title, fontsize=10)
    area = Polygon(np.zeros((1, 2)), closed=True, color=ACCENT[0], alpha=0.2, lw=0)
    ax_pv.add_patch(area)
    for leg in range(n_legs):
        ax_pv.plot(V[leg], p[leg], color=ACCENT[leg], lw=2, label=LEG_NAMES[leg])
        ax_ts.plot(S[leg], T[leg], color=ACCENT[leg], lw=2)
    dot_pv = ax_pv.plot([], [], "o", color="black")[0]
    dot_ts = ax_ts.plot([], [], "o", color="black")[0]

    ax_pv.set_xlabel("V (L)")
    ax_pv.set_ylabel("p (kPa)")
    ax_pv.set_title("p–V: area = W = ∮ p dV")
    ax_pv.legend(fontsize=7, loc="upper right")
    ax_ts.set_xlabel("S (J/K)")
    ax_ts.set_ylabel("T (K)")
    ax_ts.set_title("T–S: area = Q_h − Q_c")
    ax_ts.margins(0.15)

    def update(step: int):
        area.set_xy(np.column_stack([V_path[: step + 1], p_path[: step + 1]]))  # traced so far, closed
        dot_pv.set_data([V_path[step]], [p_path[step]])
        dot_ts.set_data([S_path[step]], [T_path[step]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(V_path)), path)
