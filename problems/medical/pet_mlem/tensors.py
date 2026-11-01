"""Build the concrete tensors: image grid, activity/attenuation phantom, sinogram bins and the system matrix A."""

import math

import torch

WIDTH = 30.0  # cm, side of the square image (field of view)
RING_RADIUS = 40.0  # cm, radius of the detector ring
MU_WATER = 0.096  # 1/cm, attenuation of 511 keV photons in water / soft tissue


def pixel_centres(n: int, device: torch.device) -> torch.Tensor:
    """Centres of the n×n pixels, shape (n, n, 2) with [iy, ix] = (x, y) in cm."""
    pixel = WIDTH / n
    c = -WIDTH / 2 + pixel * (torch.arange(n, device=device) + 0.5)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def phantom(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Activity (relative uptake) and attenuation μ (1/cm), both (n, n): a body with two lungs and hot lesions."""
    p = pixel_centres(n, device)
    x, y = p[..., 0], p[..., 1]

    def ellipse(cx, cy, ax, ay):
        return ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 <= 1

    body = ellipse(0, 0, 13, 9.5)
    lungs = ellipse(-5.5, 1, 3.5, 5) | ellipse(5.5, 1, 3.5, 5)
    activity = 1.0 * body.float()
    activity[lungs] = 0.25
    activity[ellipse(0, -5.5, 1.8, 1.8)] = 0.0  # cold spine
    for cx, cy, rad, hot in [(-5.0, 2.5, 1.2, 8.0), (5.5, -0.5, 1.6, 5.0), (2.0, 5.0, 0.8, 10.0), (-1, -1.5, 2.0, 3.0)]:
        activity[ellipse(cx, cy, rad, rad)] = hot  # "hot" lesions (tumours take up more tracer)
    mu = MU_WATER * body.float()
    mu[lungs] = 0.03
    return activity, mu


def sinogram_bins(n: int) -> tuple[int, int, float]:
    """(n_theta, n_s, s_max): angles in [0, π) and distances in [-s_max, s_max] covering the whole image."""
    s_max = WIDTH / math.sqrt(2)
    return int(round(1.25 * n)), int(math.ceil(2 * s_max / (WIDTH / n))), s_max


def system_matrix(n: int, device: torch.device) -> torch.Tensor:
    """A_{bj}: probability that a decay in pixel j lands in sinogram bin b = (θ_k, s_l).  Shape (K·L, n·n).

    Each bin's line is sampled at many points; every sample adds its step length to the pixel it falls in,
    which approximates the length of the line inside each pixel.  Dividing by K·pixel²/Δs turns lengths into
    probabilities, so each pixel's column sums to ≈ 1 (every decay gives one LOR at a random angle).
    """
    K, L, s_max = sinogram_bins(n)
    pixel = WIDTH / n
    ds = 2 * s_max / L
    theta = (torch.arange(K, device=device) + 0.5) * math.pi / K  # (K,)
    s = -s_max + ds * (torch.arange(L, device=device) + 0.5)  # (L,)
    step = pixel / 4
    t = torch.arange(-s_max, s_max, step, device=device)  # (M,) position along the line
    normal = torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # (K, 2)
    along = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)  # (K, 2)
    # points p_{klm i} = s_l n_{ki} + t_m u_{ki}
    pts = s[None, :, None, None] * normal[:, None, None, :] + t[None, None, :, None] * along[:, None, None, :]
    ix = torch.floor((pts[..., 0] + WIDTH / 2) / pixel).long()
    iy = torch.floor((pts[..., 1] + WIDTH / 2) / pixel).long()
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    rows = torch.arange(K * L, device=device).view(K, L, 1).expand_as(ix)
    A = torch.zeros(K * L, n * n, device=device)
    lengths = torch.full((int(inside.sum()),), step, device=device)
    A.index_put_((rows[inside], (iy * n + ix)[inside]), lengths, accumulate=True)  # repeated pixels add up
    return A * ds / (K * pixel**2)


def path_lengths(A: torch.Tensor, n: int) -> torch.Tensor:
    """Undo the probability scaling of A: geometric length (cm) of each LOR inside each pixel, (B, J)."""
    K, L, s_max = sinogram_bins(n)
    return A * K * (WIDTH / n) ** 2 / (2 * s_max / L)


def background_shape(n: int, device: torch.device) -> torch.Tensor:
    """Shape of randoms + scatter in the sinogram (sums to 1): half flat (randoms), half broad in s (scatter)."""
    K, L, s_max = sinogram_bins(n)
    s = -s_max + 2 * s_max / L * (torch.arange(L, device=device) + 0.5)
    profile = 0.5 / L + 0.5 * torch.softmax(-((s / 10.0) ** 2), dim=0)  # (L,)
    return (profile / K).repeat(K)  # (K·L,) bins ordered as b = k·L + l
