"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import extrinsics, intrinsics, look_at, model_matrix, perspective, pinhole, to_pixels


def matrix(a: torch.Tensor) -> str:
    """Print a small matrix one row per line."""
    return "\n".join("  [" + "  ".join(f"{x:8.3f}" for x in row) + " ]" for row in a.tolist())


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    w, h, fov = params["width"], params["height"], params["fov"]
    obj, edges = tensors.shape(params["shape"], dev)  # (N, 4), (E, 2)
    grid, grid_edges = tensors.floor_grid(3.0, 7, dev)
    model = model_matrix(params["yaw"], 1.8, 0.9, dev)  # (4, 4): scale 1.8, standing on the floor
    target = torch.tensor([0.0, 0.9, 0.0], device=dev)
    eyes = tensors.orbit(params["frames"], params["distance"], params["height_cam"], dev)
    eyes = eyes[frame_indices(len(eyes), len(eyes))]
    print(f"{params['shape']}: {len(obj)} vertices, {len(edges)} edges; floor grid {len(grid)} vertices")

    view = look_at(eyes[0], target)
    k, rt = intrinsics(fov, w, h, dev), extrinsics(view)
    print(f"intrinsics K (f = {k[0, 0]:.1f} px, principal point ({k[0, 2]:.0f}, {k[1, 2]:.0f})):\n{matrix(k)}")
    print(f"extrinsics [R|t] of the first camera:\n{matrix(rt)}")
    centre = torch.cat([target, torch.ones(1, device=dev)])[None]
    u, v = pinhole(k, rt, centre)[0].tolist()
    print(f"the look-at point lands on pixel ({u:.2f}, {v:.2f}) = the principal point")
    clip = torch.einsum("ij,jk,kl,vl->vi", perspective(fov, w / h, solve.NEAR, solve.FAR, dev), view, model, obj)
    gap = (to_pixels(clip, w, h) - pinhole(k, rt, torch.einsum("kl,vl->vk", model, obj))).abs().max()
    print(f"4×4 pipeline P·V·M vs K[R|t]·M: largest difference {gap:.1e} px")

    res = solve.render(obj, grid, model, eyes, target, fov, w, h)
    vp = res["vp"][0]
    x_vp, z_vp = (f"({a:.0f}, {b:.0f})" for a, b in vp.tolist())
    print(f"vanishing points of the floor lines (frame 0): x-lines at {x_vp}, z-lines at {z_vp} px")
    print(f"horizon at image row v = {res['horizon'][0]:.1f} (above the centre {h / 2:.0f}: the camera looks down)")
    print(
        f"dolly zoom: fov {res['dolly_fov'].min():.0f}°–{res['dolly_fov'].max():.0f}°, "
        f"distance {res['dolly_dist'].min():.2f}–{res['dolly_dist'].max():.2f}"
    )
    return plot.animate(res, edges, grid_edges, w, h, fov, out_dir / "camera_projection.gif")


if __name__ == "__main__":
    cli(run, __file__)
