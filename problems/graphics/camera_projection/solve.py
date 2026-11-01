"""Project the object and the floor grid for every camera position, in three cameras side by side."""

import math

import torch

from problems._common.progress import steps

from .equation import (
    clip_coords,
    extrinsics,
    intrinsics,
    look_at,
    orthographic,
    perspective,
    to_pixels,
    vanishing_point,
)

NEAR, FAR = 0.1, 100.0


def project(p: torch.Tensor, v: torch.Tensor, m: torch.Tensor, x: torch.Tensor, w: int, h: int):
    """Pixels (N, 2) of homogeneous points x (N, 4), and a mask (N,) of points in front of the camera."""
    clip = clip_coords(p, v, m, x)
    return to_pixels(clip, w, h), clip[:, 3] > NEAR


def dolly(fov0: float, distance: float, n_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dolly zoom: the field of view swings between 20° and 80° while the camera moves so that
    distance · tan(fov/2) stays fixed, keeping the object the same size.   → fov (F,), distance (F,)"""
    phase = torch.linspace(0, 2 * math.pi, n_frames + 1)[:-1]
    fov = 50 - 30 * torch.cos(phase)
    dist = distance * math.tan(math.radians(fov0) / 2) / torch.tan(torch.deg2rad(fov) / 2)
    return fov, dist


def render(
    obj: torch.Tensor,
    grid: torch.Tensor,
    model: torch.Tensor,
    eyes: torch.Tensor,
    target: torch.Tensor,
    fov: float,
    w: int,
    h: int,
) -> dict[str, torch.Tensor]:
    """For every frame: object and grid pixels in the perspective, orthographic and dolly-zoom cameras,
    and the vanishing points of the two grid directions (x and z) plus the horizon row of the perspective view."""
    dev = obj.device
    eye_grid = torch.eye(4, device=dev)  # the grid is already in world coordinates: M = I
    aspect = w / h
    persp = perspective(fov, aspect, NEAR, FAR, dev)
    dolly_fov, dolly_dist = dolly(fov, torch.linalg.vector_norm(eyes[0] - target).item(), len(eyes))
    k = intrinsics(fov, w, h, dev)
    out = {name: [] for name in ("obj", "obj_ok", "grid", "grid_ok", "vp", "horizon")}
    for f in steps(len(eyes), "frames"):
        view = look_at(eyes[f], target)
        dist = torch.linalg.vector_norm(eyes[f] - target).item()
        ortho = orthographic(dist * math.tan(math.radians(fov) / 2), aspect, NEAR, FAR, dev)  # same size at target
        d_eye = target + (eyes[f] - target) * dolly_dist[f].item() / dist
        cams = [
            (persp, view),
            (ortho, view),
            (perspective(dolly_fov[f].item(), aspect, NEAR, FAR, dev), look_at(d_eye, target)),
        ]
        for name, x, m in (("obj", obj, model), ("grid", grid, eye_grid)):
            px, ok = zip(*(project(p, v, m, x, w, h) for p, v in cams))
            out[name].append(torch.stack(px))  # (3, N, 2)
            out[name + "_ok"].append(torch.stack(ok))  # (3, N)

        rt = extrinsics(view)
        axes = torch.eye(3, device=dev)
        out["vp"].append(torch.stack([vanishing_point(k, rt, axes[0]), vanishing_point(k, rt, axes[2])]))  # (2, 2)
        ahead = target - eyes[f]
        ahead[1] = 0  # the horizontal direction straight ahead: its vanishing point lies on the horizon
        out["horizon"].append(vanishing_point(k, rt, ahead)[1])
    result = {name: torch.stack(v) for name, v in out.items()}
    result["dolly_fov"], result["dolly_dist"] = dolly_fov, dolly_dist
    return result
