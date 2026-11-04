"""Build the concrete tensors: the finite-volume grid on the 2-sphere, lumpy conformal factors, and profile metrics
on the 3-sphere (round and dumbbell)."""

import math

import torch


def sphere_grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Polar angle x_j = (j + ½)h, h = π/N, at cell centres; exact cell weights w_j = cos x_{j−½} − cos x_{j+½}
    (Σ w = 2, so 2π Σ w = 4π is the round area); and the axisymmetric Laplacian Δ_{S²} in flux form,
    (Δu)_j = [sin x_{j+½}(u_{j+1} − u_j) − sin x_{j−½}(u_j − u_{j−1})] / (h w_j), as an (N, N) matrix.
    sin vanishes at the poles, so no flux leaves the sphere and Σ_j w_j (Δu)_j = 0 exactly."""
    h = math.pi / n
    edges = torch.arange(n + 1, dtype=torch.float64, device=device) * h  # x_{j−½}, j = 0 … N
    x = edges[:-1] + h / 2  # (N,)
    w = torch.cos(edges[:-1]) - torch.cos(edges[1:])  # (N,)
    s = torch.sin(edges)  # sin at the cell faces; s[0] = s[N] = 0 (up to round-off)
    s[0] = s[-1] = 0.0
    lap = torch.zeros(n, n, dtype=torch.float64, device=device)
    j = torch.arange(n, device=device)
    lap[j, j] = -(s[1:] + s[:-1]) / (h * w)
    lap[j[:-1], j[:-1] + 1] = s[1:-1] / (h * w[:-1])
    lap[j[1:], j[1:] - 1] = s[1:-1] / (h * w[1:])
    return x, w, lap


def lumpy_factors(x: torch.Tensor) -> torch.Tensor:
    """Three conformal factors u(x), metric e^{2u}(dx² + sin²x dθ²): a peanut, a pear and a lumpy sphere, built from
    Legendre polynomials P_l(cos x) (axisymmetric spherical harmonics).   (N,) → (3, N)"""
    c = torch.cos(x)
    p2 = (3 * c**2 - 1) / 2
    p3 = (5 * c**3 - 3 * c) / 2
    p4 = (35 * c**4 - 30 * c**2 + 3) / 8
    p6 = (231 * c**6 - 315 * c**4 + 105 * c**2 - 5) / 16
    return torch.stack([0.35 * p2 + 0.25 * p4, 0.12 * p3 + 0.2 * p2, 0.12 * p4 + 0.1 * p6 + 0.15 * p3])


def s3_grid(n: int, device: torch.device, a: float = 0.85) -> tuple:
    """Cells uniform in η ∈ (0, π), x = X(η) = η − (a/2) sin 2η: X′ = 1 − a cos 2η is 1 − a near the poles and
    1 + a at the equator, so the poles (where the bulbs of a dumbbell end up in conformal coordinates) get finer
    cells. X is odd about both poles, so mirror ghost cells still work. Returns (x, x_η, x_ηη, h), each (N,)."""
    h = math.pi / n
    eta = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * h
    return eta - a / 2 * torch.sin(2 * eta), 1 - a * torch.cos(2 * eta), 2 * a * torch.sin(2 * eta), h


def round_s3(r: float, x: torch.Tensor) -> torch.Tensor:
    """Round 3-sphere of radius r in the conformal form g = e^{2u}(dx² + sin²x g_{S²}): u = ln r.   → (1, N)"""
    return torch.full_like(x, math.log(r))[None]


def conformal_from_profile(s: torch.Tensor, psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Turn a profile ψ(s), s ∈ [0, ℓ] with ψ = 0 only at the ends, into the conformal factor u on the grid x.
    Solving dx/ds = sin x / ψ gives ln tan(x/2) = ∫ ds/ψ + const; then u = ln(ψ / sin x). The constant (a conformal
    motion of the round S³) is fixed so the two ends sit symmetrically.   s, psi (M,) fine samples → (1, N)"""
    inner = slice(1, -1)  # the integral of 1/ψ diverges logarithmically at the poles
    ds = s[inner][1:] - s[inner][:-1]
    y = torch.cat(
        [torch.zeros(1, dtype=s.dtype), torch.cumsum(ds * 0.5 * (1 / psi[inner][1:] + 1 / psi[inner][:-1]), 0)]
    )
    y = y - 0.5 * (y[0] + y[-1] + torch.log((s[-1] - s[-2]) / s[1]))  # symmetric ends (ψ ≈ distance to the pole)
    xs = 2 * torch.atan(torch.exp(y))  # (M−2,) increasing from ≈ 0 to ≈ π
    us = torch.log(psi[inner] / torch.sin(xs))
    k = torch.searchsorted(xs, x).clamp(1, len(xs) - 1)  # linear interpolation onto the grid, all cells at once
    frac = ((x - xs[k - 1]) / (xs[k] - xs[k - 1])).clamp(0, 1)
    return (us[k - 1] + frac * (us[k] - us[k - 1]))[None]


def dumbbell(neck: float, x: torch.Tensor) -> torch.Tensor:
    """A symmetric dumbbell 3-sphere: profile ψ(s) = sin s · (1 − (1 − ε) e^{−((s − π/2)/0.35)²}), s ∈ [0, π]: two
    bulbs of radius ≈ 0.9 joined by a neck of radius ε, converted to the conformal factor u.   → (1, N)"""
    s = torch.linspace(0, math.pi, 20001, dtype=torch.float64, device=x.device)
    psi = torch.sin(s) * (1 - (1 - neck) * torch.exp(-(((s - math.pi / 2) / 0.35) ** 2)))
    return conformal_from_profile(s, psi, x)
