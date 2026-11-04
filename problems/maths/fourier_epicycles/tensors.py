"""Build the concrete tensors: a closed curve sampled at N points (equal time steps) as complex z_n = x_n + i y_n.

Three original shapes:
  logo   the outline of the Tensorizer logo (frontend/public/logo-light.svg): the hexagon silhouette of a 3×3×3 cube
         plus the three inner edges (the "Y"), walked as one closed pen stroke that goes out and back along each spoke
  star   a five-pointed star polygon (sharp corners: ringing when few circles are kept)
  heart  the smooth parametric heart x = 16 sin³s, y = 13 cos s − 5 cos 2s − 2 cos 3s − cos 4s, sampled evenly in s:
         a trigonometric polynomial (16 sin³s = 12 sin s − 4 sin 3s), so exactly 8 circles k = ±1 … ±4 draw it
"""

import math

import torch

SHAPES = ["logo", "star", "heart"]

# corners of the logo's cube in its 64 × 64 SVG frame (y pointing down), taken from logo-light.svg
_TOP, _UR, _LR = (32.00, 2.45), (57.59, 17.68), (57.59, 46.77)
_BOT, _LL, _UL, _MID = (32.00, 61.33), (6.41, 46.77), (6.41, 17.68), (32.00, 32.00)
LOGO_PATH = [_UL, _TOP, _UR, _LR, _BOT, _LL, _UL, _MID, _UR, _MID, _BOT, _MID]  # closes back to _UL


def times(n: int, device: torch.device) -> torch.Tensor:
    """t_n = n / N, n = 0 … N−1: one trip round the curve takes t from 0 to 1.  (N,) float64."""
    return torch.arange(n, dtype=torch.float64, device=device) / n


def resample(corners: torch.Tensor, n: int) -> torch.Tensor:
    """N points equally spaced by arc length along the closed polygon through `corners` (P,) complex → (N,) complex."""
    seg = torch.roll(corners, -1) - corners  # (P,) segment vectors, the last one closes the loop
    length = seg.abs()
    start = torch.cat([length.new_zeros(1), torch.cumsum(length, 0)[:-1]])  # (P,) arc length at each corner
    s = torch.arange(n, dtype=torch.float64, device=corners.device) * length.sum() / n  # (N,) target arc lengths
    which = torch.searchsorted(start, s, right=True) - 1  # (N,) segment that holds each point
    frac = (s - start[which]) / length[which]
    return corners[which] + frac * seg[which]


def normalise(z: torch.Tensor) -> torch.Tensor:
    """Centre on the mean and scale so the farthest point sits on the unit circle."""
    z = z - z.mean()
    return z / z.abs().max()


def shape(name: str, n: int, device: torch.device) -> torch.Tensor:
    """The curve `name` sampled at N points, (N,) complex128, centred and inside the unit disc."""
    if name == "logo":
        xy = torch.tensor(LOGO_PATH, dtype=torch.float64, device=device)
        z = resample(torch.complex(xy[:, 0], -xy[:, 1]), n)  # flip y: SVG y points down
    elif name == "star":
        a = math.pi / 2 + torch.arange(10, dtype=torch.float64, device=device) * math.pi / 5  # outer, inner, …
        r = torch.where(torch.arange(10, device=device) % 2 == 0, 1.0, 0.38).to(torch.float64)
        z = resample(r * torch.exp(1j * a), n)
    elif name == "heart":
        s = 2 * math.pi * times(n, device)  # sampled evenly in the parameter s, not in arc length
        x = 16 * torch.sin(s) ** 3
        y = 13 * torch.cos(s) - 5 * torch.cos(2 * s) - 2 * torch.cos(3 * s) - torch.cos(4 * s)
        z = torch.complex(x, y)
    else:
        raise ValueError(f"unknown shape {name!r}; choose from {SHAPES}")
    return normalise(z)
