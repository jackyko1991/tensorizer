"""Build the concrete tensors: a procedurally drawn test image, its noisy copies and the filter kernels."""

import math

import torch


def test_image(size: int, device: torch.device) -> torch.Tensor:
    """A clean grey-level image I_{ij} in [0, 1] of shape (size, size): a gradient, three shapes and a chirp."""
    s = torch.linspace(0, 1, size, device=device)
    y, x = torch.meshgrid(s, s, indexing="ij")  # (H, W), y grows downwards
    img = 0.25 + 0.3 * x  # smooth left-to-right gradient
    img = torch.where((x - 0.28) ** 2 + (y - 0.3) ** 2 < 0.16**2, 0.95, img)  # bright disk
    img = torch.where((x > 0.55) & (x < 0.9) & (y > 0.12) & (y < 0.42), 0.08, img)  # dark square
    img = torch.where((y > 0.5) & (y < 0.72) & (abs(x - 0.5) < (y - 0.5) * 1.6), 0.8, img)  # triangle
    chirp = 0.5 + 0.4 * torch.sin(2 * math.pi * (4 * x + 26 * x**2))  # stripes that get finer to the right
    return torch.where(y > 0.8, chirp, img).clamp(0, 1)


def gaussian_noise(img: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """I + σ ξ with ξ standard normal (fixed seed): (H, W)."""
    gen = torch.Generator().manual_seed(seed)
    return img + sigma * torch.randn(img.shape, generator=gen).to(img.device)


def salt_and_pepper(img: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    """Set a random `fraction` of the pixels to 0 or 1 (half each): (H, W)."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(img.shape, generator=gen).to(img.device)
    out = torch.where(u < fraction / 2, 0.0, img)
    return torch.where((u >= fraction / 2) & (u < fraction), 1.0, out)


def box(size: int, device: torch.device) -> torch.Tensor:
    """Box (mean) kernel: (k, k) with all entries 1/k²."""
    return torch.full((size, size), 1.0 / size**2, device=device)


def gaussian_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """g_i ∝ exp(−i²/2σ²) for |i| ≤ 3σ, normalised to sum 1: (2r+1,)."""
    r = max(1, math.ceil(3 * sigma))
    i = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(i**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_2d(sigma: float, device: torch.device) -> torch.Tensor:
    """G_{ij} = g_i g_j: the 2D Gaussian is an outer product of two 1D ones (that is what separable means)."""
    g = gaussian_1d(sigma, device)
    return torch.einsum("i,j->ij", g, g)


def sobel(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel kernels: smoothing [1, 2, 1] across × central difference [−1, 0, 1] along.   (3, 3) each."""
    smooth = torch.tensor([1.0, 2.0, 1.0], device=device)
    diff = torch.tensor([-1.0, 0.0, 1.0], device=device)
    return torch.einsum("i,j->ij", smooth, diff), torch.einsum("i,j->ij", diff, smooth)  # K_x, K_y (y down)


def laplacian_of_gaussian(sigma: float, device: torch.device) -> torch.Tensor:
    """σ² ∇²G = (x² + y² − 2σ²)/σ² · G, shifted to zero sum so flat regions give 0: (2r+1, 2r+1)."""
    g = gaussian_2d(sigma, device)
    r = g.shape[0] // 2
    i = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    rr = i[:, None] ** 2 + i[None, :] ** 2
    k = (rr - 2 * sigma**2) / sigma**2 * g
    return k - k.mean()
