"""Apply the thin-lens rules to every frame: image position, magnification and the principal rays."""

import torch

from .equation import height_at_lens, image_distance, lens_kink, magnification


def solve(s: torch.Tensor, h: float, f: float, slope_in: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a dict of tensors, one row per frame k.

    s_img (K,), m (K,), image height h_img (K,),
    y (K, 3) where each principal ray crosses the lens, slope_out (K, 3) its slope afterwards.
    """
    s_img = image_distance(s, f)
    m = magnification(s, s_img)
    y = height_at_lens(h, s, slope_in)
    slope_out = lens_kink(y, slope_in, f)
    return {"s_img": s_img, "m": m, "h_img": m * h, "y": y, "slope_out": slope_out}


def height_after_lens(y: torch.Tensor, slope_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Height of each ray at a distance x behind the lens: y + θ' x.   (K, R), (K,) → (K, R)"""
    return y + slope_out * x[:, None]
