r"""Physics → tensor translation: horizons with a time-dependent speed of light.

Light travels a comoving distance dχ = c(t) dt / a. With x = ln a and dt = dx / H, the particle horizon (the farthest
comoving distance any signal can have covered since the beginning) and the comoving Hubble radius are

    χ_p(a) = ∫ c(a′) / (a′ H(a′)) d ln a′,          r_H(a) = c(a) / (a H(a)),
    H(a) = H₀ √(Ω_r a⁻⁴ + Ω_m a⁻³ + Ω_Λ)   (standard ΛCDM expansion).

The last-scattering surface (a_rec ≈ 1/1090) lies at comoving distance D = χ_p(1) − χ_p(a_rec) from us (light after
recombination travels at today's c in every model here). Two points on opposite sides of the sky are 2D apart; their
past light cones overlap only if χ_p(a_rec) ≥ D. Otherwise one horizon spans an angle θ = χ_p(a_rec)/D and the sky
holds ≈ 2/(1 − cos θ) causally disconnected caps. Three ways to fill the gap, batched over models/frames m:

  ΛCDM:      c = c₀, standard H: θ ≈ 1°, the horizon problem.
  inflation: H = H(a_*) (de Sitter) for N e-folds before a_*: r_H shrinks ∝ 1/a, χ_p grows by ≈ c/(a_* H_*) e^N.
  VSL:       c(a) = c₀ [1 + (r − 1) s(a)], s a smooth step of width w e-folds ending at a_* (Albrecht–Magueijo;
             Moffat): a radiation era with c larger by r adds χ ≈ r c₀ a_*/(H₀√Ω_r) · πw/sin(πw).
"""

import math

import torch

from .tensors import A_REC, C0, H0, OMEGA_L, OMEGA_M, OMEGA_R, STEP_WIDTH

# Observational bounds on varying "constants" (dimensionless α = e²/(4πε₀ħc) and the speed of gravity).
BOUNDS = {
    "atomic clocks, α̇/α today (1σ)": {
        "value": (-0.1e-18, 2.1e-18), "unit": "per year", "ref": "Lange et al. 2021, PRL 126, 011102"},
    "Oklo natural reactor (≈2 Gyr ago)": {
        "value": (-0.9e-7, 1.2e-7), "unit": "Δα/α", "ref": "Damour & Dyson 1996"},
    "CMB at z ≈ 1100 (about 0.4 %)": {
        "value": (-4e-3, 4e-3), "unit": "Δα/α", "ref": "Planck intermediate results XXIV (2015)"},
    "GW170817 speed of gravity": {
        "value": (-3e-15, 7e-16), "unit": "(v_gw − c)/c", "ref": "Abbott et al. 2017, ApJL 848, L13"},
}  # fmt: skip


def hubble(a: torch.Tensor) -> torch.Tensor:
    """Standard ΛCDM expansion rate H(a) in km/s/Mpc, elementwise."""
    return H0 * torch.sqrt(OMEGA_R * a**-4 + OMEGA_M * a**-3 + OMEGA_L)


def hubble_with_inflation(a: torch.Tensor, efolds: torch.Tensor, a_star: float) -> torch.Tensor:
    """H frozen at H(a_*) for a_* e^{−N} < a < a_*, ΛCDM elsewhere.   a (A,), efolds (M,) → (M, A)"""
    h_star = hubble(torch.tensor(a_star, dtype=a.dtype, device=a.device))
    inside = (a[None, :] < a_star) & (a[None, :] > a_star * torch.exp(-efolds[:, None]))
    return torch.where(inside, h_star, hubble(a)[None, :].expand(len(efolds), -1))


def speed_of_light(a: torch.Tensor, c_ratio: torch.Tensor, a_star: float) -> torch.Tensor:
    """c(a) = c₀ [1 + (r − 1) σ((ln a_* − ln a)/w)]: r times today's c long before a_*.   (A,), (M,) → (M, A)"""
    step = torch.sigmoid((math.log(a_star) - torch.log(a)) / STEP_WIDTH)  # 1 early, 0 late
    return C0 * (1 + (c_ratio[:, None] - 1) * step[None, :])


def comoving_hubble_radius(a: torch.Tensor, c: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """r_H = c / (a H) in Mpc: how far light gets in one expansion time.   broadcast → (M, A)"""
    return c / (a * H)


def particle_horizon(x: torch.Tensor, r_H: torch.Tensor) -> torch.Tensor:
    r"""χ_p(x) = ∫_{x_min}^{x} r_H dx′ (cumulative trapezoid along the last axis).   (M, A) → (M, A)"""
    pieces = 0.5 * (r_H[:, 1:] + r_H[:, :-1]) * x.diff()
    return torch.cat([torch.zeros_like(r_H[:, :1]), pieces.cumsum(-1)], dim=-1)


def horizon_angle(chi_rec: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """θ = χ_p(a_rec)/D, the angle one causal patch subtends on today's sky (capped at π: whole sky)."""
    return (chi_rec / distance).clamp(max=math.pi)


def patch_count(theta: torch.Tensor) -> torch.Tensor:
    """Number of caps of angular radius θ needed to tile the sphere: 4π / (2π(1 − cos θ)), at least 1."""
    return (2 / (1 - torch.cos(theta))).clamp(min=1.0)


def vsl_extra_horizon(c_ratio: torch.Tensor, a_star: float, w: float = STEP_WIDTH) -> torch.Tensor:
    r"""Closed form for the extra horizon from the fast-light era, radiation domination 1/(aH) = a/(H₀√Ω_r):

    (r − 1) c₀/(H₀√Ω_r) ∫ e^x σ((x_* − x)/w) dx = (r − 1) c₀ a_*/(H₀√Ω_r) · πw / sin(πw)   (valid for w < 1).
    """
    return (c_ratio - 1) * C0 * a_star / (H0 * math.sqrt(OMEGA_R)) * (math.pi * w / math.sin(math.pi * w))


def recombination_index(x: torch.Tensor) -> int:
    return int(torch.searchsorted(x, torch.tensor([math.log(A_REC)], dtype=x.dtype, device=x.device)))
