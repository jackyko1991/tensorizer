"""Fill k-space line by line, reconstruct after every line, and compare undersampling schemes."""

import torch

from . import tensors
from .equation import regularised, zero_filled


def filling(s: torch.Tensor, n_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre-out Cartesian acquisition: masks (F, N, N) after each frame and the zero-filled images (F, N, N)."""
    n = s.shape[0]
    order = tensors.centric_order(n)  # (N,) lines in acquisition order
    counts = torch.linspace(1, n, n_frames).round().long()  # lines acquired by each frame
    masks = torch.stack([tensors.line_mask(order[:c], n, s.device) for c in counts])  # (F, N, N)
    return masks, zero_filled(s, masks).abs()  # one batched inverse FFT for all frames


def support_weights(s: torch.Tensor) -> torch.Tensor:
    """outside_{pr}: 1 where a blurry image from the central 1/8 of k-space shows no tissue, else 0. (N, N)"""
    n = s.shape[0]
    low = zero_filled(s, tensors.line_mask(tensors.scheme_lines("lowpass", n, 1 / 8), n, s.device)).abs()
    inside = (low > 0.15 * low.max()).double()
    inside = torch.nn.functional.max_pool2d(inside[None, None], 5, 1, 2)[0, 0]  # grow the head mask by 2 px
    return 1.0 - inside


def schemes(
    s: torch.Tensor, E: torch.Tensor, R: int, lowpass: float, random_frac: float, lam: float
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(name, mask, zero-filled image, regularised image) for skip-R, low-pass and random undersampling."""
    n = s.shape[0]
    outside = support_weights(s)
    ordinal = {2: "2nd", 3: "3rd"}.get(R, f"{R}th")
    out = []
    for name, scheme, value in [
        (f"every {ordinal} line", "skip", R),
        (f"central {lowpass:.0%} only", "lowpass", lowpass),
        (f"random {random_frac:.0%}", "random", random_frac),
    ]:
        lines = tensors.scheme_lines(scheme, n, value)
        mask = tensors.line_mask(lines, n, s.device)
        out.append((name, mask, zero_filled(s, mask).abs(), regularised(s, lines, E, outside, lam).abs()))
    return out


def rmse(image: torch.Tensor, truth: torch.Tensor) -> float:
    return float((image - truth).pow(2).mean().sqrt())
