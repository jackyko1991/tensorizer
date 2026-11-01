"""March the start-up flow with RK4; also the exact Bessel-series solution and the dye-line positions."""

import math

import torch

from problems._common.integrators import integrate

from .equation import rhs


def simulate(
    u0: torch.Tensor, L: torch.Tensor, G: float, rho: float, nu: float, dr: float, t_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), u (T, N+1)) with the wall value u_N = 0 appended.

    Explicit RK4 is stable for diffusion while λ_max Δt < 2.79; here λ_max ≲ 8ν/Δr², so Δt = Δr²/(4ν) is safe.
    """
    dt = dr**2 / (4 * nu)
    n_steps = int(math.ceil(t_max / dt))
    u = integrate(lambda u, t: rhs(u, L, G, rho, nu), u0, dt, n_steps)  # (T, N)
    u = torch.nn.functional.pad(u, (0, 1))  # (T, N+1): add u = 0 at the wall
    return torch.arange(n_steps + 1, device=u0.device, dtype=u0.dtype) * dt, u


def bessel_zeros(n: int) -> torch.Tensor:
    """First n positive zeros λ_k of J_0, by Newton's method from the estimate (k - ¼)π (J_0' = -J_1)."""
    lam = (torch.arange(1, n + 1, dtype=torch.float64) - 0.25) * math.pi
    for _ in range(20):
        lam = lam + torch.special.bessel_j0(lam) / torch.special.bessel_j1(lam)
    return lam


def series_solution(r: torch.Tensor, t: float, G: float, mu: float, rho: float, R: float, terms: int = 40):
    r"""Exact start-up flow in a pipe:

    u = G(R² - r²)/(4μ) - (2 G R²/μ) Σ_k J_0(λ_k r/R) / (λ_k³ J_1(λ_k)) exp(-λ_k² ν t / R²)
    """
    lam = bessel_zeros(terms).to(r.device)  # (K,)
    decay = torch.exp(-(lam**2) * (mu / rho) * t / R**2) / (lam**3 * torch.special.bessel_j1(lam))  # (K,)
    modes = torch.special.bessel_j0(r[:, None] * lam / R)  # (N, K)
    return G * (R**2 - r**2) / (4 * mu) - 2 * G * R**2 / mu * torch.einsum("nk,k->n", modes, decay)


def dye_lines(times: torch.Tensor, u: torch.Tensor, release_every: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Dye lines released from a wire across the pipe every `release_every` seconds (the hydrogen-bubble trick).

    C(t, r) = ∫_0^t u dt' is how far fluid at radius r has moved. A line released at t_k sits at
    z_k(r, t) = C(t, r) - C(t_k, r). Returns release times (K,) and the cumulative distance C (T, N+1).
    """
    dt = (times[1] - times[0]).item()
    C = torch.cumsum(u, dim=0) * dt - u * dt  # (T, N+1) left-Riemann sum, C(0) = 0
    releases = torch.arange(0.0, times[-1].item(), release_every, dtype=times.dtype)
    return releases, C
