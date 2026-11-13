"""The measured roofline: the bandwidth slope, each precision's measured matmul rate, and the ridge points."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import save_png

from . import equation

COLOURS = {"fp64": "#7c3aed", "fp32": "#2563eb", "tf32": "#0891b2", "bf16": "#16a34a", "fp16": "#f59e0b"}


def roofline(result: dict, path: Path) -> Path:
    bw = result["bandwidth"]["gbs"] * 1e9  # byte/s
    runs = [m for m in result["matmul"] if "tflops" in m]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.logspace(-2, 4, 200)
    top = max([m["tflops"] for m in runs] + [1e-3]) * 1e12
    ax.plot(xs, np.minimum(bw * xs, top) / 1e12, color="#9ca3af", lw=1.2, ls="--", label=f"memory: {bw / 1e9:.0f} GB/s")
    for m in runs:
        c = COLOURS.get(m["precision"], "#374151")
        peak = m["tflops"] * 1e12
        ax.plot(xs, [equation.roofline(peak, bw, x) / 1e12 for x in xs], color=c, lw=1.6)
        ax.scatter([m["intensity"]], [m["tflops"]], color=c, s=36, zorder=3)
        ax.annotate(
            f"{m['precision'].upper()} {m['tflops']:.2f} TFLOP/s",
            (m["intensity"], m["tflops"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=c,
        )
        r = equation.ridge(peak, bw)
        ax.axvline(r, color=c, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title(f"Measured roofline: {result['device']}", fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    return save_png(fig, path)
