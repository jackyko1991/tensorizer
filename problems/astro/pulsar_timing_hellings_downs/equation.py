r"""Physics → tensor translation for a pulsar timing array.

A plane gravitational wave travelling along the unit vector Ω̂ has metric perturbation
h_{ij}(t) = Σ_A h_A(t) e^A_{ij}(Ω̂) with polarisation tensors (m, n ⟂ Ω̂, rotated by ψ)

    e^+_{ij} = m_i m_j − n_i n_j,        e^×_{ij} = m_i n_j + n_i m_j.

The pulse frequency of pulsar a (unit vector p̂_a) is shifted by the antenna pattern

    F^A_a = ½ p̂^i_a p̂^j_a e^A_{ij} / (1 + Ω̂·p̂_a)                 i, j summed

    z_a(t) = Σ_A F^A_a [h_A(t) − h_A(t − L_a(1 + Ω̂·p̂_a)/c)]           Earth term − pulsar term

and the timing residual is r_a(t) = ∫ z_a dt. For an isotropic background, the expected correlation of two
pulsars separated by ζ (distinct pulsars, pulsar terms uncorrelated) is the Hellings–Downs curve

    Γ(ζ) = ½ − x/4 + (3/2) x ln x,       x = (1 − cos ζ)/2,

normalised so Γ(0⁺) = ½, Γ(π) = ¼ and Γ_aa = 1 (Earth + pulsar term). Equivalently
Γ(ζ) = (3/8π) ∮ dΩ̂ Σ_A F^A_a F^A_b.
"""

import math

import torch


def polarisation_tensors(omega: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""e^A_{ij} for every wave.   omega (…, 3) unit vectors, psi (…,) → (…, 2, 3, 3)."""
    z = torch.tensor([0.0, 0.0, 1.0], dtype=omega.dtype, device=omega.device).expand_as(omega)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=omega.dtype, device=omega.device).expand_as(omega)
    helper = torch.where((omega[..., 2:].abs() > 0.9), x, z)  # any vector not parallel to Ω̂
    u = torch.linalg.cross(omega, helper)
    u = u / u.norm(dim=-1, keepdim=True)
    v = torch.linalg.cross(omega, u)
    c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
    m, n = c * u + s * v, -s * u + c * v  # rotate the transverse frame by the polarisation angle ψ
    plus = torch.einsum("...i,...j->...ij", m, m) - torch.einsum("...i,...j->...ij", n, n)
    cross = torch.einsum("...i,...j->...ij", m, n) + torch.einsum("...i,...j->...ij", n, m)
    return torch.stack([plus, cross], dim=-3)


def antenna_patterns(pulsars: torch.Tensor, omega: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""F^A_{ma} = ½ p̂^i_a p̂^j_a e^A_{mij} / (1 + Ω̂_m·p̂_a) — i and j summed.

    pulsars (N, 3), omega (M, 3), e (M, 2, 3, 3) → (M, 2, N)
    """
    numerator = 0.5 * torch.einsum("mAij,ni,nj->mAn", e, pulsars, pulsars)
    denominator = 1.0 + torch.einsum("mi,ni->mn", omega, pulsars)  # (M, N)
    return numerator / denominator[:, None, :].clamp(min=1e-9)


def separation_cos(pulsars: torch.Tensor) -> torch.Tensor:
    r"""cos ζ_{ab} = p̂^i_a p̂^i_b — i summed.   (N, 3) → (N, N)"""
    return torch.einsum("ai,bi->ab", pulsars, pulsars)


def hellings_downs(cos_zeta: torch.Tensor) -> torch.Tensor:
    r"""Γ(ζ) = ½ − x/4 + (3/2) x ln x with x = (1 − cos ζ)/2   (x ln x → 0 at ζ = 0)."""
    x = (1 - cos_zeta) / 2
    return 0.5 - x / 4 + 1.5 * torch.xlogy(x, x)


def monopole(cos_zeta: torch.Tensor) -> torch.Tensor:
    """Γ = 1 for every pair: a common clock error (a correlated error of the time standard)."""
    return torch.ones_like(cos_zeta)


def dipole(cos_zeta: torch.Tensor) -> torch.Tensor:
    """Γ = cos ζ: an error in the Solar-System ephemeris (Earth's position) looks like this."""
    return cos_zeta


def hd_minimum() -> tuple[float, float]:
    """dΓ/dx = 0 ⇒ ln x = −5/6: returns (ζ_min in degrees, Γ_min = ½ − (3/2) e^{−5/6})."""
    x = math.exp(-5 / 6)
    return math.degrees(math.acos(1 - 2 * x)), 0.5 - 1.5 * x


def residuals(
    f_pattern: torch.Tensor,
    amp: torch.Tensor,
    phase: torch.Tensor,
    omega_t: torch.Tensor,
    delay_phase: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    r"""r_{at} = Σ_{m,A} F^A_{ma} (h_A/ω_m) [sin(ω_m t + φ_{mA}) − sin(ω_m t + φ_{mA} − Δ_{ma})]   (∫ of z dt).

    f_pattern (M, 2, N), amp and phase (M, 2), omega_t (M,) angular frequencies, delay_phase Δ (M, N), t (T,)
    → (N, T). Uses sin(θ − Δ) = sin θ cos Δ − cos θ sin Δ so every sum is a matrix product over (m, A).
    """
    theta = omega_t[:, None, None] * t + phase[..., None]  # (M, 2, T)
    weight = f_pattern * (amp / omega_t[:, None])[..., None]  # (M, 2, N)
    earth_minus_pulsar_sin = weight * (1 - torch.cos(delay_phase))[:, None, :]
    pulsar_cos = weight * torch.sin(delay_phase)[:, None, :]
    return torch.einsum("mAn,mAt->nt", earth_minus_pulsar_sin, torch.sin(theta)) + torch.einsum(
        "mAn,mAt->nt", pulsar_cos, torch.cos(theta)
    )


def correlation(r: torch.Tensor) -> torch.Tensor:
    r"""ρ_{ab} = ⟨r̃_a r̃_b⟩_t / ⟨r̃²⟩, r̃ = r − mean, normalised by the array-average auto-power.   (…, N, T) → (…, N, N)

    Every pulsar sees the same expected background power, so one common normaliser keeps ρ an unbiased
    (noise-diluted) estimate of Γ: for pure GW residuals ⟨ρ_{ab}⟩ = Γ(ζ_{ab}) and ρ_{aa} ≈ 1.
    """
    r = r - r.mean(dim=-1, keepdim=True)
    c = torch.einsum("...at,...bt->...ab", r, r) / r.shape[-1]  # t summed
    auto = torch.diagonal(c, dim1=-2, dim2=-1).mean(dim=-1)  # (…,) mean over pulsars of ⟨r̃²⟩
    return c / auto[..., None, None]
