"""Build the concrete tensors: procedurally drawn cartoon faces, random patterns and corrupted queries.

The faces are drawn here from a handful of random numbers each (head shape, eyes, brows, mouth, hair, glasses,
ears) with distance fields on a pixel grid: no images or characters are copied from anywhere.
"""

import torch


def _grid(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel centres y (down) and x (right) in [−1, 1]: two (size, size) tensors."""
    c = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size * 2 - 1
    return torch.meshgrid(c, c, indexing="ij")


def faces(n: int, size: int, seed: int, device: torch.device) -> torch.Tensor:
    """n cartoon faces as ±1 images (ink +1, paper −1): shape (n, size, size). Every feature is broadcast over the
    batch, so all faces are drawn at once from their (n, 1, 1) parameters."""
    gen = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> torch.Tensor:  # one uniform number per face, shaped (n, 1, 1) to broadcast
        return (lo + (hi - lo) * torch.rand(n, 1, 1, generator=gen)).to(device)

    y, x = _grid(size, device)  # (S, S)
    # head outline: an ellipse ring
    a, b = u(0.62, 0.86), u(0.72, 0.92)
    head = ((x / a) ** 2 + ((y - 0.05) / b) ** 2).sqrt()  # (n, S, S) 1 on the outline
    ink = (head - 1).abs() < 0.07
    # ears: small rings on both sides (half of the faces)
    ears = (u(0, 1) < 0.5) & ((((x.abs() - a) / 0.11) ** 2 + ((y - 0.05) / 0.17) ** 2).sqrt() - 1).abs().lt(0.35)
    ink |= ears & (head > 1)
    # hair: a filled cap above a random hairline, or nothing
    hairline = u(-0.75, -0.35)
    ink |= (u(0, 1) < 0.65) & (head < 1.06) & (y < hairline + 0.08 * torch.cos(6 * x * u(0.5, 1.5)))
    # eyes: two discs, with optional glasses around them
    ex, ey, er = u(0.22, 0.38), u(-0.18, 0.02), u(0.07, 0.13)
    eye = (((x.abs() - ex) ** 2 + (y - ey) ** 2).sqrt() < er) & (head < 1)
    glasses = (u(0, 1) < 0.35) & ((((x.abs() - ex) ** 2 + (y - ey) ** 2).sqrt() - 2 * er - 0.04).abs() < 0.04)
    ink |= eye | (glasses & (head < 1))
    # eyebrows: tilted bars above the eyes
    tilt = u(-0.5, 0.5)
    brow_y = ey - er - u(0.08, 0.16) + tilt * (x.abs() - ex)
    ink |= ((x.abs() - ex).abs() < 0.14) & ((y - brow_y).abs() < 0.035) & (head < 1)
    # nose: a short vertical stroke
    ink |= (x.abs() < 0.035) & (y > ey + 0.05) & (y < ey + u(0.18, 0.3))
    # mouth: a parabola y = m0 + c x² (c > 0 smiles, c < 0 frowns) of random width
    m0, c, mw = u(0.38, 0.55), u(-0.9, 1.6), u(0.18, 0.4)
    ink |= ((y - (m0 + c * (x**2 - mw**2))).abs() < 0.045) & (x.abs() < mw)
    return ink.float() * 2 - 1


def random_patterns(n: int, dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """n random ±1 patterns of dimension dim, each bit a fair coin: (n, dim). The textbook capacity case."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n, dim), generator=gen) * 2 - 1).float().to(device)


def corrupt(patterns: torch.Tensor, mask_rows: float, flip: float, seed: int) -> torch.Tensor:
    """Queries: blank (−1) the bottom fraction mask_rows of every image and flip a fraction `flip` of the other
    pixels.   (P, S, S) → (P, S, S)"""
    gen = torch.Generator().manual_seed(seed)
    q = patterns.clone()
    size = q.shape[-1]
    q[..., size - int(mask_rows * size) :, :] = -1
    flips = (torch.rand(q.shape, generator=gen) < flip).to(q.device)
    return torch.where(flips, -q, q)


def flip_bits(patterns: torch.Tensor, flip: float, seed: int) -> torch.Tensor:
    """Flip a random fraction `flip` of the bits of every pattern.   (P, N) → (P, N)"""
    gen = torch.Generator().manual_seed(seed)
    flips = (torch.rand(patterns.shape, generator=gen) < flip).to(patterns.device)
    return torch.where(flips, -patterns, patterns)
