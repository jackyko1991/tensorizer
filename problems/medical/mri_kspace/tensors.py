"""Build the concrete tensors: Shepp–Logan phantom, encoding matrix, gradient-made k-space positions, sampling masks."""

import math

import torch

# Modified Shepp–Logan head phantom: (intensity, semi-axis a, semi-axis b, centre x, centre y, angle in degrees).
_ELLIPSES = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0),
]


def shepp_logan(n: int, device: torch.device) -> torch.Tensor:
    """Proton-density image ρ_{pr} (row p, column r) of the Shepp–Logan phantom. Shape (N, N) float64."""
    c = torch.linspace(-1, 1, n, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(-c, c, indexing="ij")  # row 0 at the top (y = +1)
    rho = torch.zeros(n, n, dtype=torch.float64, device=device)
    for value, a, b, x0, y0, deg in _ELLIPSES:
        th = math.radians(deg)
        xr = (x - x0) * math.cos(th) + (y - y0) * math.sin(th)
        yr = -(x - x0) * math.sin(th) + (y - y0) * math.cos(th)
        rho += value * ((xr / a) ** 2 + (yr / b) ** 2 <= 1).double()
    return rho


def encoding_matrix(n: int, device: torch.device) -> torch.Tensor:
    """E_{up} = exp(−2πi u p / N): the phase a spin at pixel p picks up at k-space line u. Shape (N, N) complex.

    In 2D the encoding is separable, E_{(u,v),(p,r)} = E_{up} E_{vr}, so the full (N², N²) matrix is never built.
    """
    idx = torch.arange(n, dtype=torch.float64, device=device)
    return torch.exp(-2j * math.pi * torch.outer(idx, idx) / n)


def readout_k(n: int, fov: float, gradient: float, gamma_bar: float = 42.58e6) -> tuple[torch.Tensor, float]:
    """k(t) = γ/(2π) ∫ G dt along one readout, sampled every dwell time Δt.  Returns (k (N,) in 1/m, Δt in s).

    The k-space step must be Δk = 1/FOV to cover the field of view, so Δt = Δk / (γ̄ G).
    A pre-phasing gradient lobe first moves k to −N/2·Δk, then the readout gradient G sweeps it at constant speed.
    """
    dk = 1.0 / fov
    dwell = dk / (gamma_bar * gradient)
    G = torch.full((n,), gradient, dtype=torch.float64)  # (N,) constant readout gradient
    area = torch.cumsum(G * dwell, dim=0) - G * dwell  # (N,) ∫_0^{t_n} G dt at the start of each sample
    k = -n / 2 * dk + gamma_bar * area  # (N,) runs from −N/2·Δk to (N/2 − 1)·Δk
    return k, dwell


def centric_order(n: int) -> torch.Tensor:
    """Phase-encode lines u in the order 0, 1, −1, 2, −2, … (centre of k-space first), as FFT indices. (N,)"""
    signed = torch.arange(n) - n // 2  # −N/2 … N/2 − 1
    order = signed[torch.argsort(signed.abs() * 2 - (signed > 0).long(), stable=True)]
    return order % n


def line_mask(lines: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Sampling mask m_{uv}: 1 on every acquired line u (all readout samples v), 0 elsewhere. (N, N)"""
    mask = torch.zeros(n, n, dtype=torch.float64, device=device)
    mask[lines] = 1.0
    return mask


def scheme_lines(scheme: str, n: int, value: float, seed: int = 0) -> torch.Tensor:
    """Acquired line indices (FFT order) for an undersampling scheme.

    'skip': every R-th line (R = value), 'lowpass': only the central fraction value of lines,
    'random': a fully sampled centre plus random outer lines, value = fraction kept (fixed seed).
    """
    signed = torch.arange(n) - n // 2
    if scheme == "skip":
        lines = torch.arange(0, n, int(value))
    elif scheme == "lowpass":
        keep = max(1, round(value * n))
        lines = signed[torch.argsort(signed.abs(), stable=True)[:keep]] % n
    else:
        centre = signed.abs() <= n // 16
        outer = torch.nonzero(~centre).flatten()
        g = torch.Generator().manual_seed(seed)
        extra = max(0, round(value * n) - int(centre.sum()))
        chosen = outer[torch.randperm(len(outer), generator=g)[:extra]]
        lines = torch.cat([signed[centre] % n, signed[chosen] % n])
    return torch.sort(lines).values
