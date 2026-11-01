"""White-theme matplotlib defaults and GIF/PNG writers shared by every plot.py."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .progress import bar  # noqa: E402
from .runtime import fast_mode  # noqa: E402

MAX_FRAMES = 120
ACCENT = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#db2777", "#65a30d"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9ca3af",
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "axes.prop_cycle": matplotlib.cycler(color=ACCENT),
        "font.size": 10,
        "axes.titlesize": 11,
    }
)


def frame_indices(n_steps: int, n_frames: int = MAX_FRAMES) -> list[int]:
    """Pick evenly spaced time steps so a GIF never has more than n_frames frames."""
    n_frames = 5 if fast_mode() else min(n_frames, MAX_FRAMES)
    n_frames = max(1, min(n_frames, n_steps))
    return [round(k * (n_steps - 1) / max(1, n_frames - 1)) for k in range(n_frames)]


def save_gif(fig: plt.Figure, update: Callable[[int], object], frames: list[int], path: Path, fps: int = 20) -> Path:
    """update(step) redraws the figure for one time step; frames lists the steps to render."""
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    tmp = _tmp_path(path)
    with bar(len(frames), "GIF frames") as progress:
        anim.save(tmp, writer=PillowWriter(fps=fps), dpi=80, progress_callback=lambda i, n: progress.update(1))
    plt.close(fig)
    return _swap(tmp, path)  # swap in the finished file so a viewer never sees a half-written GIF


def save_png(fig: plt.Figure, path: Path) -> Path:
    tmp = _tmp_path(path)
    fig.savefig(tmp, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return _swap(tmp, path)


def _swap(tmp: Path, path: Path, attempts: int = 25) -> Path:
    """Replace path with tmp, retrying while Windows reports the old file as in use (a server reading it)."""
    for _ in range(attempts - 1):
        try:
            return tmp.replace(path)
        except PermissionError:
            time.sleep(0.2)
    return tmp.replace(path)


def _tmp_path(path: Path) -> Path:
    path = Path(path)
    # Unique per process (two renders of one problem can overlap); same suffix so matplotlib picks the format.
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
