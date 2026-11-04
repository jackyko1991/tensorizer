r"""Physics → tensor translation: distances in a Friedmann–Lemaître universe.

The expansion rate of a universe with matter Ω_m, a cosmological constant Ω_Λ and curvature Ω_k = 1 − Ω_m − Ω_Λ is

    H(z) = H_0\,E(z),        E(z)^2 = Ω_m (1+z)^3 + Ω_k (1+z)^2 + Ω_Λ.

Light from a supernova at redshift z has crossed the (dimensionless) comoving distance

    χ(z) = \int_0^z dz' / E(z'),

and its flux falls off with the luminosity distance

    d_L = (1+z)\,\frac{c}{H_0}\,S_k(χ),   S_k(χ) = \sinh(\sqrt{Ω_k} χ)/\sqrt{Ω_k}   (open, Ω_k > 0)
                                           = χ                                      (flat)
                                           = \sin(\sqrt{-Ω_k} χ)/\sqrt{-Ω_k}       (closed, Ω_k < 0).

Astronomers quote the distance modulus μ = m − M = 5 \log_{10}(d_L / \mathrm{Mpc}) + 25.

Every function broadcasts: the cosmological parameters can carry leading batch axes (a whole grid of models, shape
(A, B, 1)) while the redshifts sit on the last axis, so one call evaluates every model at every supernova.
"""

import torch

C_KMS = 299792.458  # speed of light in km/s, so c/H0 is in Mpc when H0 is in km/s/Mpc


def e_squared(z: torch.Tensor, Om: torch.Tensor, OL: torch.Tensor) -> torch.Tensor:
    r"""E(z)^2 = Ω_m (1+z)^3 + Ω_k (1+z)^2 + Ω_Λ.   z: (Z,), Om, OL: (..., 1) → (..., Z)"""
    Ok = 1 - Om - OL
    x = 1 + z
    return Om * x**3 + Ok * x**2 + OL


def comoving_integral(nodes: torch.Tensor, Om: torch.Tensor, OL: torch.Tensor):
    r"""χ at every node: cumulative trapezoid of 1/E on the uniform grid z_0 = 0 < z_1 < … (Z,) → (..., Z).

    Also returns the integrand f = dχ/dz = 1/E at the nodes, used by `interpolate`. E² is clamped at a tiny
    positive value so the models with no Big Bang inside the data range (E² < 0 somewhere, see `valid`) stay
    finite; they are excluded from the fit by `valid`, not here.
    """
    f = e_squared(nodes, Om, OL).clamp_min(1e-12).rsqrt()  # 1/E, (..., Z)
    h = nodes[1] - nodes[0]
    steps = 0.5 * h * (f[..., 1:] + f[..., :-1])
    return torch.cat([torch.zeros_like(f[..., :1]), steps.cumsum(-1)], dim=-1), f


def interpolate(nodes: torch.Tensor, chi_nodes: torch.Tensor, f_nodes: torch.Tensor, z: torch.Tensor):
    """Cubic Hermite interpolation of χ from the grid to the supernova redshifts.   (..., Z) → (..., N)

    Hermite uses the value χ and the slope f = 1/E at both neighbouring nodes. The indices and weights depend
    only on the redshifts, so the result stays differentiable in Ω.
    """
    h = nodes[1] - nodes[0]
    k = (z / h).floor().long().clamp(0, len(nodes) - 2)  # (N,) left node of each supernova
    s = z / h - k  # (N,) position between the nodes, in [0, 1)
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, 3 * s**2 - 2 * s**3, s**3 - s**2
    return (
        h00 * chi_nodes[..., k]
        + h10 * h * f_nodes[..., k]
        + h01 * chi_nodes[..., k + 1]
        + h11 * h * f_nodes[..., k + 1]
    )


def transverse(chi: torch.Tensor, Ok: torch.Tensor) -> torch.Tensor:
    r"""S_k(χ): sinh for open, sin for closed and χ (1 + Ω_k χ²/6) near flat, so Ω_k = 0 is smooth."""
    root = Ok.abs().clamp_min(1e-12).sqrt()
    x = root * chi
    open_ = torch.sinh(x) / root
    closed = torch.sin(x) / root
    flat = chi * (1 + Ok * chi**2 / 6)
    return torch.where(Ok > 1e-6, open_, torch.where(Ok < -1e-6, closed, flat))


def luminosity_distance(z, nodes, H0, Om, OL) -> torch.Tensor:
    r"""d_L = (1+z) (c/H_0) S_k(χ(z)) in Mpc.   z: (N,), H0, Om, OL: (...) or (..., 1) → (..., N)"""
    H0, Om, OL = (torch.as_tensor(p, dtype=z.dtype, device=z.device) for p in (H0, Om, OL))
    if Om.dim() == 0:
        H0, Om, OL = H0[None], Om[None], OL[None]
    chi = interpolate(nodes, *comoving_integral(nodes, Om, OL), z)  # (..., N)
    return (1 + z) * C_KMS / H0 * transverse(chi, 1 - Om - OL)


def distance_modulus(d_L: torch.Tensor) -> torch.Tensor:
    """μ = 5 log10(d_L / Mpc) + 25 (25 = 5 log10 of 1 Mpc / 10 pc)."""
    return 5 * torch.log10(d_L) + 25


def chi_squared(mu_model: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""χ² = Σ_i ((μ_i − μ^{model}_i) / σ_i)²: einsum over the supernova index i.   (..., N) → (...)"""
    r = (mu - mu_model) / sigma
    return torch.einsum("...i,...i->...", r, r)


def deceleration(Om: torch.Tensor, OL: torch.Tensor) -> torch.Tensor:
    r"""q_0 = −\ddot a a / \dot a^2 today = Ω_m/2 − Ω_Λ: negative means the expansion is speeding up."""
    return Om / 2 - OL


def valid(nodes: torch.Tensor, Om: torch.Tensor, OL: torch.Tensor) -> torch.Tensor:
    """True where E² > 0 at every redshift up to the last supernova (a model the light could have crossed)."""
    return (e_squared(nodes, Om, OL) > 0).all(dim=-1)
