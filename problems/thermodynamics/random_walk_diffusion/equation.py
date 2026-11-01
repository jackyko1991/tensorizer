r"""Physics → tensor translation.

Brownian motion: N particles n = 1..N in d = 2 dimensions (i ∈ {x, y}) each receive a random kick every
time step Δt,

    x_{t+1,ni} = x_{t,ni} + \sqrt{2 D \Delta t} \; \xi_{t,ni},       \xi \sim \mathcal N(0, 1)

so the position is the running sum of the kicks, x_{tni} = \sum_{s \le t} \sqrt{2D\Delta t}\,\xi_{sni}.

The mean-squared displacement averages the squared distance over all walkers. With the repeated indices
n and i summed:

    \mathrm{MSD}_t = \frac1N \, x_{tni} \, x_{tni} = 2 d D \, t_t        (einsum('tni,tni->t') / N)

Fitting a straight line through the origin, D_fit = t_t \mathrm{MSD}_t / (2d \, t_t t_t), sums over t.
The density of walkers solves the diffusion equation ∂_t ρ = D ∂_i ∂_i ρ:

    \rho(\mathbf x, t) = \frac{1}{4 \pi D t} \, e^{-x_i x_i / 4 D t}
"""

import math

import torch


def kick_size(D: float, dt: float) -> float:
    """Standard deviation of one step along each axis: √(2 D Δt)."""
    return math.sqrt(2 * D * dt)


def mean_squared_displacement(x: torch.Tensor) -> torch.Tensor:
    r"""MSD_t = (1/N) x_{tni} x_{tni}, summed over walkers n and components i.   (T, N, d) → (T,)"""
    return torch.einsum("tni,tni->t", x, x) / x.shape[1]


def msd_theory(t: torch.Tensor, D: float, d: int = 2) -> torch.Tensor:
    """Einstein's result ⟨x_i x_i⟩ = 2 d D t.   (T,) → (T,)"""
    return 2 * d * D * t


def fit_diffusion(t: torch.Tensor, msd: torch.Tensor, d: int = 2) -> float:
    """Least-squares slope through the origin, divided by 2d:  D = Σ_t t·MSD / (2d Σ_t t²)."""
    return (torch.einsum("t,t->", t, msd) / (2 * d * torch.einsum("t,t->", t, t))).item()


def gaussian_density(X: torch.Tensor, Y: torch.Tensor, t: float, D: float) -> torch.Tensor:
    """ρ(x, y, t) = exp(−(x² + y²) / 4Dt) / (4πDt): walkers per unit area, all started at the origin."""
    return torch.exp(-(X**2 + Y**2) / (4 * D * t)) / (4 * math.pi * D * t)
