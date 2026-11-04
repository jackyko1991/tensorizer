r"""Physics → tensor translation: a scalar field rolling in an expanding universe, with e-folds N = ln a as clock.

Units: reduced Planck mass M_pl = 1. The Friedmann and Klein–Gordon equations,

    H² = (φ̇²/2 + V) / 3,        φ̈ + 3Hφ̇ + V'(φ) = 0,

become, with ψ = dφ/dN = φ̇/H and the Hubble slow-roll parameter ε_H = −Ḣ/H² = ψ²/2,

    dφ/dN = ψ,        dψ/dN = −(3 − ε_H) (ψ + V'/V),        H² = V / (3 − ε_H).

Only V'/V enters: the overall height of the potential drops out of the background (and of n_s and r), so every
potential is written with height 1. Inflation (ä > 0, i.e. ä/a = H²(1 − ε_H) > 0) lasts while ε_H < 1.

Predictions at the pivot scale, which left the horizon N_* e-folds before the end:

    potential slow roll (first order):  ε_V = (V'/V)²/2,  η_V = V''/V,  n_s = 1 − 6ε_V + 2η_V,  r = 16 ε_V
    Hubble flow on the exact background (second order, Stewart & Lyth 1993; Schwarz et al. 2001):
        ε₁ = ε_H,  ε_{i+1} = d ln ε_i / dN,  C = γ_E + ln 2 − 2 ≈ −0.7296
        n_s = 1 − 2ε₁ − ε₂ − 2ε₁² − (2C + 3) ε₁ε₂ − C ε₂ε₃,      r = 16 ε₁ (1 + C ε₂)

The batch: B fields, one per potential, each picked from the five formulas by a one-hot row w_{bp},

    V_b = w_{bp} U_p(φ_b)        (repeated p summed)
"""

import math

import torch

C_FLOW = 0.5772156649015329 + math.log(2) - 2  # γ_E + ln 2 − 2
STAROBINSKY = math.sqrt(2 / 3)


def shapes(phi: torch.Tensor, f: float, mu: float) -> torch.Tensor:
    r"""U_p(φ) for the five potentials, each with height 1 where that makes sense: (B,) → (B, 5).

    m²φ²: φ²/2      λφ⁴: φ⁴/4      Starobinsky R²: (1 − e^{−√(2/3) φ})²
    natural: 1 + cos(φ/f)          quartic hilltop: 1 − (φ/μ)⁴
    """
    return torch.stack(
        [
            phi**2 / 2,
            phi**4 / 4,
            (1 - torch.exp(-STAROBINSKY * phi)) ** 2,
            1 + torch.cos(phi / f),
            1 - (phi / mu) ** 4,
        ],
        dim=-1,
    )


def potential(phi: torch.Tensor, model: dict) -> torch.Tensor:
    r"""V_b = w_{bp} U_p(φ_b): repeated p summed.   (..., B) → (..., B)"""
    return torch.einsum("bp,...bp->...b", model["w"], shapes(phi, model["f"], model["mu"]))


def slopes(phi: torch.Tensor, f: float, mu: float) -> torch.Tensor:
    """U'_p(φ), the derivatives of shapes() written out by hand (fast inside the time loop): (B,) → (B, 5)."""
    e = torch.exp(-STAROBINSKY * phi)
    return torch.stack(
        [phi, phi**3, 2 * STAROBINSKY * e * (1 - e), -torch.sin(phi / f) / f, -4 * phi**3 / mu**4],
        dim=-1,
    )


def log_slope(phi: torch.Tensor, model: dict) -> torch.Tensor:
    r"""V'/V = w_{bp} U'_p / w_{bp} U_p (repeated p summed).   (..., B) → (..., B)"""
    dV = torch.einsum("bp,...bp->...b", model["w"], slopes(phi, model["f"], model["mu"]))
    return dV / potential(phi, model)


def autograd_slope(phi: torch.Tensor, model: dict) -> torch.Tensor:
    """V'/V by autograd (each V_b depends only on φ_b, so the gradient of the sum is the batch of slopes)."""
    with torch.enable_grad():
        x = phi.detach().requires_grad_(True)
        V = potential(x, model)
        (dV,) = torch.autograd.grad(V.sum(), x)
    return dV / V.detach()


def slow_roll_potential(phi: torch.Tensor, model: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """ε_V = (V'/V)²/2 and η_V = V''/V, with V' and V'' from autograd.   (B,) → (B,), (B,)"""
    with torch.enable_grad():
        x = phi.detach().requires_grad_(True)
        V = potential(x, model)
        (dV,) = torch.autograd.grad(V.sum(), x, create_graph=True)
        (d2V,) = torch.autograd.grad(dV.sum(), x)
    V, dV = V.detach(), dV.detach()
    return 0.5 * (dV / V) ** 2, d2V / V


def slow_roll_velocity(phi: torch.Tensor, model: dict) -> torch.Tensor:
    """The slow-roll attractor dφ/dN = −V'/V (from 3Hφ̇ = −V' and H² = V/3)."""
    return -log_slope(phi, model)


def rhs(state: torch.Tensor, model: dict) -> torch.Tensor:
    """d/dN of state = (φ, ψ) stacked as (2, B): the Klein–Gordon equation in e-folds."""
    phi, psi = state
    eps = 0.5 * psi**2
    return torch.stack([psi, -(3 - eps) * (psi + log_slope(phi, model))])


def epsilon_2(psi: torch.Tensor, phi: torch.Tensor, model: dict) -> torch.Tensor:
    """ε₂ = d ln ε_H / dN = 2 ψ'/ψ, with ψ' taken from the equation of motion (no finite differences)."""
    dpsi = -(3 - 0.5 * psi**2) * (psi + log_slope(phi, model))
    return 2 * dpsi / psi


def ns_r_potential(eps_v: torch.Tensor, eta_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return 1 - 6 * eps_v + 2 * eta_v, 16 * eps_v


def ns_r_hubble_flow(e1, e2, e3) -> tuple[torch.Tensor, torch.Tensor]:
    C = C_FLOW
    ns = 1 - 2 * e1 - e2 - 2 * e1**2 - (2 * C + 3) * e1 * e2 - C * e2 * e3
    return ns, 16 * e1 * (1 + C * e2)
