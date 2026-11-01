"""Main loop: parameters → tensors → solve → plot."""

import time
from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import sphere_hits


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    sc = tensors.scene(dev)
    target = torch.tensor([0.0, 0.7, 0.0], device=dev)
    n = params["frames"]
    if params["orbit"] == "light":  # the light circles the scene, the camera stands still
        lights = tensors.orbit(n, 4.0, 5.0, 0.8, dev)
        eyes = torch.tensor([0.0, 2.2, 7.5], device=dev).expand(n, 3)
    else:  # the camera circles the scene, the light stands still
        lights = torch.tensor([3.0, 5.0, 2.5], device=dev).expand(n, 3)
        eyes = tensors.orbit(n, 7.5, 2.2, 0.0, dev)
    keep = frame_indices(n, n)  # every frame (only a handful in fast mode)
    eyes, lights = eyes[keep], lights[keep]

    w, h = params["width"], params["height"]
    o, d = tensors.camera_rays(eyes[0], target, params["fov"], w, h)
    t = sphere_hits(o, d, sc["centres"], sc["radii"])
    print(f"{o.shape[0]} rays (one per pixel of {w}×{h}) against {len(sc['radii'])} spheres: t table {tuple(t.shape)}")

    start = time.perf_counter()
    images, stats = solve.render(sc, eyes, lights, target, params["fov"], w, h, params["bounces"], params["shininess"])
    elapsed = time.perf_counter() - start
    s = stats[0]
    print(f"first rays: {s['spheres']:.1%} hit a sphere, {s['ground']:.1%} the ground, {s['sky']:.1%} the sky")
    print(f"ground pixels in shadow (frame 0): {s['shadow']:.1%}")
    print(f"{params['bounces']} mirror bounce(s) + shadow rays: {len(images)} frames in {elapsed:.1f} s")
    return plot.animate(images, eyes, lights, sc, params["orbit"], out_dir / "ray_tracing_spheres.gif")


if __name__ == "__main__":
    cli(run, __file__)
