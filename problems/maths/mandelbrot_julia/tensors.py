"""Build the concrete tensors: complex pixel grids, the zoom path and the Julia parameters on the cardioid."""

import math

import torch

# Zoom targets on the boundary of M (complex128 needs a width above ~1e-13 before pixels merge).
TARGETS = {
    "seahorse": complex(-0.743643887037151, 0.131825904205330),  # deep in the seahorse valley
    "dendrite": complex(0.0, 1.0),  # c = i: a Misiurewicz point, 0 → i → −1+i → −i → −1+i → … (preperiodic)
    "minibrot": complex(-1.754877666246693, 0.0),  # centre of the period-3 minibrot, root of c³ + 2c² + c + 1 = 0
}
START = (complex(-0.6, 0.0), 3.2)  # centre and width of the first frame: the whole set


def pixel_grid(centre: complex, width: float, n: int, device: torch.device) -> torch.Tensor:
    """Square grid of complex numbers c_{ij} = centre + (x_j + i y_i), shape (n, n), complex128; row 0 is the top."""
    s = torch.linspace(-width / 2, width / 2, n, dtype=torch.float64, device=device)
    return centre + s[None, :] + 1j * s.flip(0)[:, None]


def zoom_widths(width0: float, decades: float, n_frames: int) -> list[float]:
    """Frame widths shrinking geometrically from width0 to width0·10^(−decades)."""
    return [width0 * 10 ** (-decades * k / max(1, n_frames - 1)) for k in range(n_frames)]


def zoom_centres(target: complex, widths: list[float]) -> list[complex]:
    """Glide from the overview centre to the target while zooming (the target stays in every frame)."""
    c0, w0 = START
    out = []
    for w in widths:
        t = min(1.0, math.log(w0 / w) / math.log(10))  # the first decade pans, then the centre is the target
        out.append(c0 + (target - c0) * (3 * t * t - 2 * t * t * t))
    return out


def iterations_for(width: float, base: int) -> int:
    """Deeper frames need more iterations: base + 150 per decade of zoom."""
    return int(base + 150 * max(0.0, math.log10(START[1] / width)))


def julia_angles(n_frames: int, device: torch.device) -> torch.Tensor:
    """Angles θ_f around the main cardioid for the Julia panel: (F,). θ = π is c = −¾, θ = 2π/3 the period-3 root."""
    return torch.linspace(0.15, 2 * math.pi - 0.15, n_frames, dtype=torch.float64, device=device)
