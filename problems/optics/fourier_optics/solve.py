"""Transform every aperture of the animation to its far field; measure the Airy ring of the circular hole."""

import torch

from .equation import far_field, intensity, log_intensity
from .tensors import aperture


def solve(schedule: list[tuple[str, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (apertures (F, N, N), far-field log₁₀ intensity (F, N, N)) for every (shape, size) frame."""
    masks, patterns = [], []
    for shape, size in schedule:
        a = aperture(shape, size)
        masks.append(a)
        patterns.append(log_intensity(intensity(far_field(a))))
    return torch.stack(masks), torch.stack(patterns)


def radial_profile(i: torch.Tensor) -> torch.Tensor:
    """Average of a centred (N, N) image over rings of width 1 pixel around the centre: (N/2,)."""
    n = i.shape[0]
    c = torch.arange(n) - n // 2
    r = torch.sqrt(c[:, None] ** 2 + c[None, :] ** 2).round().long()  # (N, N) ring number of each pixel
    keep = r < n // 2
    total = torch.zeros(n // 2).index_add_(0, r[keep], i[keep])
    count = torch.zeros(n // 2).index_add_(0, r[keep], torch.ones_like(i[keep]))
    return total / count


def first_dark_ring(diameter: float) -> float:
    """Radius (frequency pixels) of the first minimum of the circular hole's far-field intensity."""
    profile = radial_profile(intensity(far_field(aperture("circle", diameter))))
    for r in range(1, len(profile) - 1):
        if profile[r] < profile[r - 1] and profile[r] <= profile[r + 1]:
            return float(r)
    return float("nan")
