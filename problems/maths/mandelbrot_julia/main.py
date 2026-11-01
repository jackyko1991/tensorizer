"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import cardioid_boundary, in_cardioid_or_bulb

M_AREA = 1.5066  # area of M, from published pixel-counting estimates (≈ 1.50659)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = 64 if fast_mode() else params["resolution"]
    base = params["max_iter"]

    # 1. the whole set: exact cardioid/bulb tests against the iteration
    centre, width = tensors.START
    c = tensors.pixel_grid(centre, width, 2 * n, dev)  # (2n, 2n)
    whole = solve.mandelbrot(c, base)
    bounded = whole["n"] == 0
    exact = in_cardioid_or_bulb(c)
    pixel_area = (width / (2 * n - 1)) ** 2
    print(f"overview: {c.numel()} pixels, {base} iterations, bailout |z| > 1000")
    print(f"pixels inside the cardioid or the period-2 bulb that escaped: {int((exact & ~bounded).sum())} (must be 0)")
    print(
        f"cardioid + bulb area 3π/8 + π/16 = {3 * math.pi / 8 + math.pi / 16:.4f}, "
        f"pixel count {exact.sum() * pixel_area:.4f}"
    )
    print(f"area of M from the iteration ≈ {bounded.sum() * pixel_area:.4f}   (literature ≈ {M_AREA})")
    d = whole["d"][~bounded]
    print(f"distance estimate outside: min {d.min():.2e}, all positive: {bool((d > 0).all())}")

    # 2. the zoom: one tensor per frame, more iterations as the frames get deeper
    n_frames = params["frames"]
    picked = frame_indices(n_frames, n_frames)  # fast mode keeps only 5 frames
    widths_all = tensors.zoom_widths(width, params["zoom_decades"], n_frames)
    widths = [widths_all[k] for k in picked]
    centres = tensors.zoom_centres(tensors.TARGETS[params["target"]], widths)
    zoom = []
    for w, ctr in zip(widths, centres):
        zoom.append(solve.mandelbrot(tensors.pixel_grid(ctr, w, n, dev), tensors.iterations_for(w, base)))
    last = zoom[-1]
    print(
        f"zoom into {params['target']}: width {widths[0]:.1f} → {widths[-1]:.2e} (×{widths[0] / widths[-1]:.1e}), "
        f"{tensors.iterations_for(widths[-1], base)} iterations in the deepest frame, "
        f"{(last['n'] == 0).float().mean() * 100:.1f}% of its pixels still bounded"
    )

    # 3. Julia sets with c on the main cardioid's edge, all frames as one (F, n, n) batch
    theta_all = tensors.julia_angles(n_frames, dev)
    c_julia = cardioid_boundary(theta_all[picked])  # (F,)
    z0 = tensors.pixel_grid(0j, 3.2, n, dev)  # (n, n)
    jul = solve.julia(z0[None], c_julia[:, None, None], 300)  # (F, n, n)
    julia = [{k: v[f] for k, v in jul.items()} for f in range(len(picked))]
    filled = (jul["n"] == 0).float().mean(dim=(1, 2))
    for f in (0, len(picked) // 2, len(picked) - 1):
        cf = c_julia[f]
        print(
            f"Julia c = {cf.real:+.4f} {cf.imag:+.4f}i: |2z*| = {abs(1 - torch.sqrt(1 - 4 * cf)):.4f}, "
            f"filled {filled[f]:.1%}"
        )

    box = (centre.real - width / 2, centre.real + width / 2, -width / 2, width / 2)
    return plot.animate(zoom, widths, centres, julia, c_julia, whole, box, out_dir / "mandelbrot_julia.gif")


if __name__ == "__main__":
    cli(run, __file__)
