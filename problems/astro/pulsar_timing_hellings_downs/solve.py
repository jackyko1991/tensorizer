"""Simulate many sky realisations, cross-correlate every pulsar pair, bin by angle and compare models."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import (
    antenna_patterns,
    correlation,
    dipole,
    hellings_downs,
    monopole,
    polarisation_tensors,
    residuals,
)


def simulate(
    pulsars: torch.Tensor,
    distances: torch.Tensor,
    t: torch.Tensor,
    n_real: int,
    n_waves: int,
    n_freq: int,
    noise: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Timing residuals r_{kat} for K independent skies: (K, N, T).

    The GW part is scaled so its rms over all pulsars is 1; white noise of rms `noise` is added on top.
    """
    span = float(t[-1] - t[0])
    out = []
    for _ in steps(n_real, "sky realisations"):
        w = tensors.plane_waves(n_waves, n_freq, span, gen)
        e = polarisation_tensors(w["omega"], w["psi"])  # (M, 2, 3, 3)
        f = antenna_patterns(pulsars, w["omega"], e)  # (M, 2, N)
        tau = distances * (1 + torch.einsum("mi,ni->mn", w["omega"], pulsars))  # (M, N) pulsar-term delay
        r = residuals(f, w["amp"], w["phase"], w["omega_t"], w["omega_t"][:, None] * tau, t)  # (N, T)
        r = r - r.mean(dim=-1, keepdim=True)
        r = r / r.pow(2).mean().sqrt()
        out.append(r + noise * torch.randn(r.shape, generator=gen, dtype=r.dtype))
    return torch.stack(out)


def pair_correlations(r: torch.Tensor, cos_zeta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Correlations of every distinct pair a < b: (ζ in degrees (P,), ρ (K, P))."""
    rho = correlation(r)  # (K, N, N)
    a, b = torch.triu_indices(r.shape[1], r.shape[1], offset=1)
    return torch.rad2deg(torch.acos(cos_zeta[a, b].clamp(-1, 1))), rho[:, a, b]


def bin_by_angle(zeta: torch.Tensor, rho: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Mean correlation of the pairs in each angular bin, per realisation: (K, B)."""
    which = torch.bucketize(zeta, edges[1:-1])  # (P,) bin index of each pair
    onehot = torch.nn.functional.one_hot(which, len(edges) - 1).double()  # (P, B)
    return torch.einsum("kp,pb->kb", rho, onehot) / onehot.sum(0).clamp(min=1)


def fit_models(centres_deg: torch.Tensor, y: torch.Tensor, err: torch.Tensor) -> dict:
    """Weighted least-squares amplitude A for ρ = A·Γ(ζ) with Γ = HD, monopole, dipole; returns (A, χ²) each."""
    c = torch.cos(torch.deg2rad(centres_deg))
    w = 1.0 / err**2
    fits = {}
    for name, model in (("Hellings–Downs", hellings_downs), ("monopole", monopole), ("dipole", dipole)):
        g = model(c)
        amp = (w * y * g).sum() / (w * g * g).sum()
        fits[name] = (amp.item(), (w * (y - amp * g) ** 2).sum().item())
    return fits


def bin_errors(binned: torch.Tensor) -> torch.Tensor:
    """Standard error of the mean over realisations, per bin: (B,). Needs K ≥ 2."""
    return binned.std(dim=0) / binned.shape[0] ** 0.5


def sky_average_power(pulsar: torch.Tensor, n_dirs: int, gen: torch.Generator) -> float:
    """⟨Σ_A (F^A)²⟩ over isotropic wave directions for one pulsar (analytically 1/3)."""
    omega = tensors.random_directions(n_dirs, gen)
    psi = torch.pi * torch.rand(n_dirs, generator=gen, dtype=torch.float64)
    f = antenna_patterns(pulsar[None], omega, polarisation_tensors(omega, psi))  # (M, 2, 1)
    return f.pow(2).sum(1).mean().item()


def hd_from_sky_integral(zeta_deg: torch.Tensor, n_dirs: int, gen: torch.Generator) -> torch.Tensor:
    """Γ(ζ) = (3/8π) ∮ dΩ Σ_A F^A_a F^A_b, by Monte Carlo over directions (Earth terms only): (Z,)."""
    z = torch.deg2rad(zeta_deg.double())
    a = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(len(z), 3)
    b = torch.stack([torch.sin(z), torch.zeros_like(z), torch.cos(z)], -1)  # (Z, 3)
    omega = tensors.random_directions(n_dirs, gen)
    psi = torch.pi * torch.rand(n_dirs, generator=gen, dtype=torch.float64)
    e = polarisation_tensors(omega, psi)
    fa, fb = antenna_patterns(a, omega, e), antenna_patterns(b, omega, e)  # (M, 2, Z)
    return 1.5 * torch.einsum("mAz,mAz->z", fa, fb) / n_dirs  # (3/8π)·4π·mean = (3/2)·mean
