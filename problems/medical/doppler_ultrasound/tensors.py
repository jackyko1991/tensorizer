"""Build the concrete tensors: sample times, the pulsatile centreline velocity and the random scatterers."""

import math

import torch


def sample_times(prf: float, heart_rate: float, n_beats: int, device: torch.device) -> torch.Tensor:
    """Slow-time samples t, one per transmitted pulse (Δt = 1/PRF) over n_beats cardiac cycles: shape (T,)."""
    duration = n_beats * 60.0 / heart_rate
    n = int(duration * prf)
    return torch.arange(n, device=device) / prf


def cardiac_waveform(t: torch.Tensor, v_peak: float, heart_rate: float) -> torch.Tensor:
    """Centreline velocity v_max(t) of an artery, shape (T,): a sharp systolic peak, a short reverse-flow dip
    and slow forward flow in diastole (a stylised triphasic waveform, roughly like a leg artery)."""
    period = 60.0 / heart_rate
    tau = torch.remainder(t, period)  # time since the start of the beat
    systole = torch.exp(-(((tau - 0.12) / 0.045) ** 2))  # peak 0.12 s after the beat starts
    reverse = torch.exp(-(((tau - 0.27) / 0.04) ** 2))  # brief backflow as the valve closes
    return v_peak * (0.2 + 0.8 * systole - 0.35 * reverse)


def scatterers(n: int, R: float, length: float, device: torch.device, seed: int = 0) -> dict[str, torch.Tensor]:
    """n red-blood-cell scatterers spread uniformly over the vessel cross-section, all shape (N,).

    r       radial distance from the axis (r = R √u gives a uniform density over the disc)
    phase   random starting phase φ_n of each echo
    amp     echo amplitude a_n (cells scatter with random strength)
    x, y    position in the longitudinal view used by the animation (y = r cos α is the projection)
    """
    g = torch.Generator().manual_seed(seed)
    r = R * torch.sqrt(torch.rand(n, generator=g))
    alpha = 2 * math.pi * torch.rand(n, generator=g)
    out = {
        "r": r,
        "phase": 2 * math.pi * torch.rand(n, generator=g),
        "amp": 0.5 + torch.rand(n, generator=g),
        "x": length * torch.rand(n, generator=g),
        "y": r * torch.cos(alpha),
    }
    return {k: v.to(device) for k, v in out.items()}


def noise(n: int, sigma: float, device: torch.device, seed: int = 1) -> torch.Tensor:
    """Complex white receiver noise of standard deviation sigma per sample: shape (T,)."""
    g = torch.Generator().manual_seed(seed)
    return (sigma / math.sqrt(2) * torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g))).to(device)
