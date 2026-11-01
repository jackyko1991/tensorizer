"""Main loop: parameters → Cornell box → progressive render + high-spp reference → noise analysis → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    res = 32 if fast else params["resolution"]
    spp, ref_spp = (16, 64) if fast else (params["spp"], params["reference_spp"])
    roulette = params["termination"] == "roulette"
    scene = tensors.cornell_box(dev, params["light_power"])
    cam = tensors.camera(res)
    n_prims = len(scene["albedo"])
    print(f"Cornell box: {len(scene['n'])} planes + {len(scene['radius'])} spheres = {n_prims} primitives")
    print(f"{res}×{res} pixels, {res * res * 16:,} rays per batch of 16 samples per pixel")
    rule = f"Russian roulette after {params['bounces']} bounces" if roulette else f"{params['bounces']} bounces"
    print(f"paths end by {rule}")

    milestones = sorted({round(v) for v in torch.logspace(0, torch.log10(torch.tensor(spp)), 30).tolist()})
    start = time.perf_counter()
    gi, di = solve.progressive(scene, cam, milestones, torch.Generator(dev).manual_seed(1), params["bounces"], roulette)
    t_prog = time.perf_counter() - start
    start = time.perf_counter()
    ref_gi, ref_di = solve.render(scene, cam, ref_spp, torch.Generator(dev).manual_seed(2), params["bounces"], roulette)
    t_ref = time.perf_counter() - start
    print(f"progressive render to {spp} spp: {t_prog:.1f} s; reference with {ref_spp} spp: {t_ref:.1f} s")

    ns = torch.tensor(milestones, dtype=torch.float32)
    err_gi, err_di = solve.rmse(gi, ref_gi).cpu(), solve.rmse(di, ref_di).cpu()
    usable = ns <= spp / 4  # the reference has noise of its own; fit where it is still small by comparison
    slope = solve.noise_slope(ns[usable], err_gi[usable])
    for i in sorted({0, len(milestones) // 3, 2 * len(milestones) // 3, len(milestones) - 1}):
        print(f"  spp {milestones[i]:4d}: RMSE vs reference {err_gi[i]:.4f} (GI), {err_di[i]:.4f} (direct only)")
    print(f"log-log slope of RMSE against spp: {slope:.2f} (Monte Carlo: −0.5, i.e. noise ∝ 1/√spp)")

    left, right = tensors.pixel_of(cam, [-0.8, 0.3, -1.0]), tensors.pixel_of(cam, [0.8, 0.3, -1.0])
    for name, (r, c) in (("left (next to red)", left), ("right (next to green)", right)):
        g, d = ref_gi[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1)), ref_di[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1))
        print(f"white back wall, {name}: red/green = {g[0] / g[1]:.2f} with GI, {d[0] / d[1]:.2f} direct only")
    share = 1 - ref_di.mean() / ref_gi.mean()
    print(f"indirect (bounced) light is {100 * share:.0f}% of the image's total brightness")
    return plot.animate(gi, di, ref_gi, ref_di, milestones, err_gi, err_di, ref_spp, out_dir / "path_tracing.gif")


if __name__ == "__main__":
    cli(run, __file__)
