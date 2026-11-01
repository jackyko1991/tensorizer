"""Build the concrete tensors: pixel/detector grid, the Shepp–Logan phantom and the projection angles."""

import math

import torch

# Modified Shepp–Logan head phantom (Toft): amplitude A, semi-axes a, b, centre (x0, y0), tilt φ in degrees.
SHEPP_LOGAN = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),  # skull
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),  # brain
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),  # ventricles
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),  # three small tumours
    (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
]


def grid(n: int, device: torch.device) -> torch.Tensor:
    """Pixel centres in [-1, 1]: (N,). The detector bins s use the same grid."""
    return (torch.arange(n, device=device) + 0.5) * (2.0 / n) - 1.0


def ellipse_params(device: torch.device) -> torch.Tensor:
    """The phantom's ellipses as a (K, 6) tensor, angles converted to radians."""
    e = torch.tensor(SHEPP_LOGAN, device=device)
    e[:, 5] = torch.deg2rad(e[:, 5])
    return e


def shepp_logan(x: torch.Tensor, ellipses: torch.Tensor) -> torch.Tensor:
    """f_{yx} = A_k · inside_{kyx}  (k summed): the sum of uniform ellipses.  → (Y, X), row index = y."""
    yy, xx = torch.meshgrid(x, x, indexing="ij")  # (Y, X)
    amp, a, b, x0, y0, phi = ellipses.T  # each (K,)
    dx = xx[None] - x0[:, None, None]
    dy = yy[None] - y0[:, None, None]
    c, s = torch.cos(phi)[:, None, None], torch.sin(phi)[:, None, None]
    u, v = dx * c + dy * s, -dx * s + dy * c  # coordinates in each ellipse's own frame
    inside = ((u / a[:, None, None]) ** 2 + (v / b[:, None, None]) ** 2 <= 1.0).float()  # (K, Y, X)
    return torch.einsum("k,kyx->yx", amp, inside)


def disk(x: torch.Tensor, radius: float) -> torch.Tensor:
    """A centred uniform disk of value 1: (Y, X)."""
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    return (xx**2 + yy**2 <= radius**2).float()


def angles(n: int, device: torch.device) -> torch.Tensor:
    """θ_a = π a / A for a = 0 … A−1: a half turn is enough for parallel beams: (A,)."""
    return torch.arange(n, device=device) * (math.pi / n)
