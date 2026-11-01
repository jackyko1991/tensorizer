"""For every loop radius: B on the x–z plane and on the axis, plus the analytic axis field for comparison."""

import torch

from . import tensors
from .equation import biot_savart, loop_axis_field


def solve(layout: str, radii: list[float], current: float, n_seg: int, n_grid: int, eps, device) -> dict:
    """Return a dict: B_plane (T, G, 3), B_axis (T, A, 3), B_axis_exact (T, A), plus the points used.

    Each radius is one call to biot_savart: every grid point and every wire segment in one einsum.
    """
    xs, plane = tensors.plane_xz(n_grid, device)  # (G, 3)
    axis = tensors.z_axis(200, device)  # (A, 3)
    h = xs[1] - xs[0]

    B_plane, B_axis, B_exact = [], [], []
    for R in radii:
        mid, dl = tensors.coils(layout, R, n_seg, device)  # (S, 3), (S, 3)
        B_plane.append(biot_savart(plane, mid, dl, eps, current, softening=0.5 * h))
        B_axis.append(biot_savart(axis, mid, dl, eps, current))
        heights = [z * R for z, _ in tensors.LAYOUTS[layout]]
        currents = [sign * current for _, sign in tensors.LAYOUTS[layout]]
        B_exact.append(loop_axis_field(axis[:, 2], R, heights, currents))

    return {
        "xs": xs,
        "plane": plane,
        "axis": axis,
        "radii": torch.tensor(radii),
        "B_plane": torch.stack(B_plane),
        "B_axis": torch.stack(B_axis),
        "B_axis_exact": torch.stack(B_exact),
    }
