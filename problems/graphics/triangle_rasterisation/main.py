"""Main loop: parameters → mesh, camera and pixels → rasterise every frame → plot."""

import time
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    res = 40 if fast_mode() else params["resolution"]
    n_frames = 5 if fast_mode() else params["frames"]
    scene = tensors.scene(params["mesh"], dev)
    cam = tensors.camera(res, dev)
    q = tensors.pixels(res, dev)  # (P, 2)
    print(f"{params['mesh']}: {len(scene['verts'])} vertices, {len(scene['faces'])} triangles (2 of them the floor)")
    print(
        f"{res}×{res} = {len(q)} pixels → barycentrics for up to {len(q) * len(scene['faces']):,} pixel-triangle pairs"
    )

    start = time.perf_counter()
    frames, stats = solve.animate_mesh(scene, cam, q, tensors.light(dev), n_frames, params["tilt"])
    elapsed = time.perf_counter() - start
    culled = sum(s["culled"] for s in stats) / len(stats)
    covered = sum(s["covered"] for s in stats) / len(stats)
    print(f"rendered {n_frames} frames in {elapsed:.1f} s ({1000 * elapsed / n_frames:.0f} ms per frame)")
    print(f"back-face culling removes {culled:.0f} of {stats[0]['triangles']} triangles per frame on average")
    print(f"surface covers {100 * covered / len(q):.0f}% of the pixels")
    print(f"largest perspective-vs-affine texture-coordinate gap on the floor: {stats[0]['uv_error']:.2f}")
    print("(uv runs 0 → 1 across the floor, so the affine checker is off by up to that many floor widths)")
    return plot.animate(frames, params["mesh"], out_dir / "triangle_rasterisation.gif")


if __name__ == "__main__":
    cli(run, __file__)
