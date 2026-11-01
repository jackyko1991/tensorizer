"""Iterate the quadratic map on every pixel at once and record how (and how fast) each one escapes."""

import torch

from problems._common.progress import steps

from .equation import distance_estimate, smooth_count, step


def escape(z0: torch.Tensor, c: torch.Tensor, dz0: float, dc: float, max_iter: int, radius: float = 1e3) -> dict:
    """Escape-time iteration on a batch of pixels of any shape.

    Mandelbrot: z0 = 0, c = pixels, dz0 = 0, dc = 1.   Julia: z0 = pixels, c = constant, dz0 = 1, dc = 0.
    Only pixels that are still bounded are updated: escaped ones are written out and dropped from the tensors.
    Returns n (escape step, 0 = never escaped), smooth count ν and distance estimate d, all with the input's shape.
    """
    shape = torch.broadcast_shapes(z0.shape, c.shape)
    c = c.expand(shape).flatten()  # (P,)
    z = z0.expand(shape).flatten().clone()
    dz = torch.full_like(z, dz0)
    alive = torch.arange(z.numel(), device=z.device)  # indices of the pixels still iterating
    n_esc = torch.zeros(z.numel(), dtype=torch.float64, device=z.device)
    z_abs = torch.ones_like(n_esc)
    dz_abs = torch.ones_like(n_esc)
    for n in steps(max_iter, "iterations"):
        z, dz = step(z, dz, c, dc)
        out = z.abs() > radius
        if out.any():
            done = alive[out]
            n_esc[done] = n + 1
            z_abs[done] = z[out].abs()
            dz_abs[done] = dz[out].abs()
            keep = ~out
            alive, z, dz, c = alive[keep], z[keep], dz[keep], c[keep]
            if alive.numel() == 0:
                break
    escaped = n_esc > 0
    nu = torch.where(escaped, smooth_count(n_esc, z_abs), torch.nan)
    dist = torch.where(escaped, distance_estimate(z_abs, dz_abs), 0.0)
    return {"n": n_esc.reshape(shape), "nu": nu.reshape(shape), "d": dist.reshape(shape)}


def mandelbrot(c: torch.Tensor, max_iter: int, radius: float = 1e3) -> dict:
    """M: start every orbit at z = 0 with the pixel as c; derivative with respect to c."""
    return escape(torch.zeros_like(c), c, 0.0, 1.0, max_iter, radius)


def julia(z0: torch.Tensor, c: torch.Tensor, max_iter: int, radius: float = 1e3) -> dict:
    """Filled Julia set K_c: start at the pixel with c fixed (c may be (F, 1, 1) for one c per frame)."""
    return escape(z0, c, 1.0, 0.0, max_iter, radius)
