"""Build the concrete tensors: pixel coordinates and the aperture masks (1 = open, 0 = opaque)."""

import torch

N = 256  # grid size: N × N pixels
SHAPES = ["square", "circle", "double slit", "grating"]


def coordinates(n: int = N) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel coordinates centred on the grid: y[a, b] = a − N/2, x[a, b] = b − N/2, each (N, N)."""
    c = torch.arange(n, dtype=torch.float32) - n // 2
    y, x = torch.meshgrid(c, c, indexing="ij")
    return y, x


def aperture(shape: str, size: float, n: int = N) -> torch.Tensor:
    """Mask A_{ab} of the given shape, overall width `size` pixels: (N, N) float."""
    y, x = coordinates(n)
    half = size / 2
    if shape == "square":  # exactly `size` pixels wide in x and y
        open_ = (x >= -half) & (x < half) & (y >= -half) & (y < half)
    elif shape == "circle":  # diameter `size`
        open_ = x**2 + y**2 <= half**2
    elif shape == "double slit":  # two slits, centres size/2 apart, each size/8 wide and `size` tall
        width = max(1.0, size / 8)
        open_ = ((x - size / 4).abs() < width / 2) | ((x + size / 4).abs() < width / 2)
        open_ &= y.abs() < half
    elif shape == "grating":  # 5 slits with period size/5, half open
        period = max(2.0, size / 5)
        open_ = (torch.remainder(x + period / 4, period) < period / 2) & (x.abs() < 2.5 * period)
        open_ &= y.abs() < half
    else:
        raise ValueError(f"unknown aperture shape {shape!r}")
    return open_.float()


def schedule(size_min: float, size_max: float, frames_per_shape: int) -> list[tuple[str, float]]:
    """(shape, size) for every animation frame: each shape in turn, its size growing from min to max."""
    sizes = torch.linspace(size_min, size_max, frames_per_shape).round().tolist()
    return [(shape, size) for shape in SHAPES for size in sizes]
