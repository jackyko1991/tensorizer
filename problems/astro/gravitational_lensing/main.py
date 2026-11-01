"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    M = 1.0
    b, y0 = tensors.ray_fan(params["b_min"], params["b_max"], params["n_rays"], torch.device("cpu"))
    rays = solve.trace_rays(y0, M)
    alpha = solve.deflection_angles(rays)
    rays["paths"] = solve.ray_paths(rays)

    print(f"shadow: rays with b < √27 M = {equation.shadow_radius(M):.3f} M fall in")
    for b_k, a_k in zip(b.tolist(), alpha.tolist()):
        weak = equation.thin_lens_deflection(torch.tensor(b_k), M)
        fate = (
            "captured" if math.isnan(a_k) else f"α = {math.degrees(a_k):7.2f}°   4GM/(c²b) = {math.degrees(weak):6.2f}°"
        )
        print(f"b = {b_k:5.2f} M   {fate}")

    # a 10-solar-mass black hole halfway to a star 8 kpc away
    kpc = 3.0857e19  # m
    theta_E = equation.einstein_radius(10 * 1477.0, 4 * kpc, 8 * kpc)
    print(f"Einstein radius for 10 M_sun at 4 kpc, source at 8 kpc: {theta_E * 206264.806e3:.2f} milliarcseconds")

    half = 2.5
    theta = tensors.sky_grid(params["n_pixels"], half, dev)
    centres = tensors.source_path(120, params["impact"], 2.0, dev)
    lensed, plain = solve.lensed_images(theta, centres, params["source_radius"])
    mu = solve.magnification(lensed, plain)
    print(f"peak magnification {mu.max():.2f} (point source at u = {params['impact']:g}: "
          f"{equation.point_magnification(torch.tensor(max(params['impact'], 1e-3))):.2f})")  # fmt: skip

    sky = {"lensed": lensed, "centres": centres, "mu": mu, "half_width": half}
    return plot.animate(rays, b, alpha, M, sky, out_dir / "gravitational_lensing.gif")


if __name__ == "__main__":
    cli(run, __file__)
