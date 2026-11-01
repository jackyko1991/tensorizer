r"""Physics → tensor translation.

A CT scanner measures line integrals of the attenuation map f(x, y) — the Radon transform.
For a ray at angle θ and signed distance s from the centre, parametrised by t along the ray,

    p(\theta, s) = \int f(s\cos\theta - t\sin\theta,\; s\sin\theta + t\cos\theta)\, dt.

On the grid: sample the image at rotated points (θ, t, s), then sum over t:

    p_{\theta s} = f_{\theta t s}\, \Delta t                    (t summed)

Filtered back-projection inverts it. Filter every projection with the ramp |ν| along s
(a product in Fourier space), then smear each filtered projection back along its rays and
add up all angles:

    q_{\theta s} = \mathcal F^{-1}_s\big[\,|\nu|\, \mathcal F_s[p_{\theta s}]\,\big]
    f_{yx} = Q_{\theta y x}\, \Delta\theta,   Q_{\theta yx} = q_\theta(x\cos\theta + y\sin\theta)   (θ summed)

Coordinates live in [-1, 1]²; pixel and detector centres coincide at θ = 0.
"""

import torch
import torch.nn.functional as F


def rotated_points(theta: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Sample points of every ray: x = s cosθ − t sinθ, y = s sinθ + t cosθ.  → (A, T, S, 2) as (x, y)."""
    c, sn = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]  # (A, 1, 1)
    x = s[None, None, :] * c - t[None, :, None] * sn
    y = s[None, None, :] * sn + t[None, :, None] * c
    return torch.stack([x, y], dim=-1)


def radon(f: torch.Tensor, theta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    r"""p_{\theta s} = f_{\theta t s} \Delta t.   f: (N, N) indexed [y, x] → sinogram (A, S).

    grid_sample interpolates f bilinearly at the rotated points; outside [-1, 1]² it returns 0.
    """
    points = rotated_points(theta, s, s)  # use the detector grid for t too: (A, T, S, 2)
    image = f[None, None].expand(len(theta), 1, *f.shape)  # (A, 1, N, N)
    samples = F.grid_sample(image, points, mode="bilinear", align_corners=False)[:, 0]  # (A, T, S)
    dt = s[1] - s[0]
    return torch.einsum("ats->as", samples) * dt  # sum over t


def ramp_filter(n_det: int, ds: float, window: str = "ram-lak", device=None) -> tuple[torch.Tensor, int]:
    r"""Frequency response H(ν) ≈ |ν| on a zero-padded detector of P ≥ 2S bins.

    Built as the FFT of the band-limited ramp kernel h(0) = 1/(4Δs²), h(n odd) = −1/(πnΔs)²
    (Kak & Slaney), which gets the zero-frequency term right. "shepp-logan" multiplies by sinc(ν Δs).
    Returns (H (P,), P).
    """
    pad = 1 << (2 * n_det - 1).bit_length()  # next power of two ≥ 2S: no wrap-around
    idx = torch.arange(pad, device=device)
    n = torch.where(idx < pad // 2, idx, idx - pad).float()
    h = torch.where(n % 2 == 1, -1.0 / (torch.pi * n * ds) ** 2, torch.zeros_like(n))
    h[0] = 1.0 / (4 * ds**2)
    H = ds * torch.fft.fft(h).real  # ≈ |ν| in cycles per unit length
    if window == "shepp-logan":
        nu = torch.fft.fftfreq(pad, d=ds, device=device)
        H = H * torch.sinc(nu * ds)
    return H, pad


def filter_projections(p: torch.Tensor, H: torch.Tensor, pad: int) -> torch.Tensor:
    r"""q_{\theta s} = F^{-1}[H(ν) F[p_θ](ν)] along s, for every angle at once: (A, S) → (A, S)."""
    spectrum = torch.fft.fft(p, n=pad, dim=-1)  # zero-padded FFT along s
    return torch.fft.ifft(spectrum * H, dim=-1).real[:, : p.shape[-1]]


def smear(q: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Q_{\theta yx} = q_\theta(x\cos\theta + y\sin\theta): each projection spread back along its rays.

    q: (A, S) on the detector grid (= pixel grid, so s is already a normalised coordinate) → (A, Y, X).
    """
    s = x[None, None, :] * torch.cos(theta)[:, None, None] + y[None, :, None] * torch.sin(theta)[:, None, None]
    points = torch.stack([s, torch.zeros_like(s)], dim=-1)  # (A, Y, X, 2): sample a 1-pixel-high "image"
    return F.grid_sample(q[:, None, None, :], points, mode="bilinear", align_corners=False)[:, 0]


def backproject(Q: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""f_{yx} = Q_{\theta yx} \Delta\theta  (θ summed over [0, π)).   (A, Y, X) → (Y, X)"""
    dtheta = torch.full_like(theta, torch.pi / len(theta))
    return torch.einsum("ayx,a->yx", Q, dtheta)


def ellipse_projection(theta, s, amp, a, b, x0, y0, phi):
    r"""Analytic Radon transform of a uniform ellipse: 2 A a b \sqrt{r^2 - s'^2} / r^2 (0 outside)."""
    s_shift = s[None, :] - x0 * torch.cos(theta)[:, None] - y0 * torch.sin(theta)[:, None]
    r2 = (a * torch.cos(theta - phi)) ** 2 + (b * torch.sin(theta - phi)) ** 2  # (A,)
    inside = (r2[:, None] - s_shift**2).clamp_min(0.0)
    return 2 * amp * a * b * torch.sqrt(inside) / r2[:, None]
