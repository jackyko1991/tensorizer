"""Build the concrete tensors: pulsar sky positions and distances, the random plane waves, the time grid.

Units: time in years, frequency in 1/yr, distances in light-years (so L/c is in years).
"""

import math

import torch


def random_directions(n: int, gen: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    """n unit vectors uniform on the sphere (normalised Gaussians): (n, 3)."""
    v = torch.randn(n, 3, generator=gen, dtype=dtype)
    return v / v.norm(dim=-1, keepdim=True)


def pulsar_distances(n: int, gen: torch.Generator) -> torch.Tensor:
    """L_a between about 0.5 and 3 kpc, in light-years (1 kpc ≈ 3262 ly): (n,)."""
    return 3262.0 * (0.5 + 2.5 * torch.rand(n, generator=gen, dtype=torch.float64))


def time_grid(n_times: int, span_years: float) -> torch.Tensor:
    """Observation epochs t_k, evenly spaced over the span: (T,)."""
    return torch.linspace(0.0, span_years, n_times, dtype=torch.float64)


def plane_waves(n_waves: int, n_freq: int, span_years: float, gen: torch.Generator) -> dict:
    """One realisation of an isotropic, Gaussian, red background built from M plane waves.

    Ω̂_m (M, 3): propagation directions, uniform on the sphere
    ψ_m (M,): polarisation angles in [0, π)
    ω_m (M,): angular frequency 2π j/T_span, j = 1 … n_freq (the lowest PTA frequency bins)
    h_{mA} (M, 2), φ_{mA} (M, 2): Gaussian amplitudes with σ ∝ f^{−7/6} (strain spectrum S_h ∝ f^{−7/3},
        the f^{−2/3} characteristic strain of inspiralling supermassive black-hole binaries), random phases
    """
    omega = random_directions(n_waves, gen)
    psi = math.pi * torch.rand(n_waves, generator=gen, dtype=torch.float64)
    j = torch.randint(1, n_freq + 1, (n_waves,), generator=gen).double()
    f = j / span_years
    amp = torch.randn(n_waves, 2, generator=gen, dtype=torch.float64) * f[:, None] ** (-7 / 6)
    phase = 2 * math.pi * torch.rand(n_waves, 2, generator=gen, dtype=torch.float64)
    return {"omega": omega, "psi": psi, "omega_t": 2 * math.pi * f, "amp": amp, "phase": phase}


def bin_edges(n_bins: int) -> torch.Tensor:
    """Angular-separation bin edges in degrees, 0 … 180: (n_bins + 1,)."""
    return torch.linspace(0.0, 180.0, n_bins + 1, dtype=torch.float64)
