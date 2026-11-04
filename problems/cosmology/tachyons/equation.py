r"""Physics → tensor translation (units with c = 1).

A tachyon is a hypothetical particle with imaginary mass m = iμ, so m² = −μ² < 0. The mass shell keeps its form

    E² = p² + m² = p² − μ²,      v = p/E = \sqrt{1 + μ²/E²} > 1,      E = mγ with γ = 1/\sqrt{1 − v²} imaginary,

so a tachyon speeds up as it loses energy (v → ∞ as E → 0). Lorentz boosts act on event and momentum
two-vectors alike,

    x'^a = Λ^a{}_b(β)\, x^b,      Λ(β) = γ_β \begin{pmatrix}1 & −β\\ −β & 1\end{pmatrix},

and keep the invariant x^a η_{ab} x^b. A signal of speed u > 1 sent from the origin arrives at t' = γ_β(1 − βu)/u:
earlier than it was sent whenever β > 1/u (= c²/u). In field theory a tachyonic mass m² < 0 is not a fast particle
but an unstable vacuum: φ = 0 is a hill of V(φ) = ½m²φ² + ¼λφ⁴ and the field rolls to φ = ±\sqrt{−m²/λ}.
"""

import torch


def energy(p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""E = \sqrt{p² + m²} with a complex mass (m = iμ for a tachyon).   → complex, same shape as p"""
    return torch.sqrt(p.to(torch.complex128) ** 2 + m**2)


def velocity(p: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""v = p/E (group velocity dE/dp, the same for any mass)."""
    return p / e


def lorentz_factor(v: torch.Tensor) -> torch.Tensor:
    r"""γ = 1/\sqrt{1 − v²}: real below c, imaginary above (complex input)."""
    return 1 / torch.sqrt(1 - v**2)


def boost_matrix(beta: torch.Tensor) -> torch.Tensor:
    r"""Λ^a{}_b(β) for a batch of frame velocities.   beta: (B,) → (B, 2, 2)"""
    g = 1 / torch.sqrt(1 - beta**2)
    row0 = torch.stack([g, -g * beta], dim=-1)
    row1 = torch.stack([-g * beta, g], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def boost(four: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""x'^a = Λ^a{}_b x^b  (b summed) for every frame.

    four: (..., 2) as (t, x) or (E, p), beta: (B,) → (B, ..., 2)
    """
    return torch.einsum("Bab,...b->B...a", boost_matrix(beta), four)


def minkowski_square(four: torch.Tensor) -> torch.Tensor:
    r"""x^a η_{ab} x^b = t² − x² (or E² − p² = m²), η = diag(1, −1)."""
    eta = torch.tensor([1.0, -1.0], dtype=four.dtype, device=four.device)
    return torch.einsum("...a,a,...a->...", four, eta, four)


def velocity_add(u: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""Speed u seen from a frame moving at β:  u' = (u − β)/(1 − uβ)  (rapidities subtract)."""
    return (u - beta) / (1 - u * beta)


def antitelephone(u: torch.Tensor, beta: torch.Tensor, t_send: float = 1.0) -> dict:
    r"""Tolman's antitelephone. Bob moves away at β (worldline x = βt). Alice sends a tachyon of speed u at t_send;
    Bob answers at once with a tachyon of speed u *in his own frame*. Returns the reception event and the time the
    answer gets back to Alice; t_back < t_send closes a causal loop.   u, beta broadcast together.
    """
    t_rec = u * t_send / (u - beta)  # u (t − t_send) = β t
    x_rec = beta * t_rec
    w = velocity_add(-u, -beta)  # Bob's −u in Alice's frame: (β − u)/(1 − βu)
    t_back = t_rec - x_rec / w  # where the reply worldline meets x = 0
    return {"t_receive": t_rec, "x_receive": x_rec, "w": w, "t_back": t_back}


def potential(phi: torch.Tensor, m2: float, lam: float) -> torch.Tensor:
    r"""V(φ) = ½ m² φ² + ¼ λ φ⁴; with m² < 0 the top φ = 0 is a false vacuum."""
    return 0.5 * m2 * phi**2 + 0.25 * lam * phi**4


def vacuum(m2: float, lam: float) -> float:
    r"""True vacuum |φ| = \sqrt{−m²/λ} after tachyon condensation."""
    return (-m2 / lam) ** 0.5


def force(phi: torch.Tensor, m2: float, lam: float, dx: float) -> torch.Tensor:
    r"""−δH/δφ = ∂²φ/∂x² − m²φ − λφ³ on a periodic lattice (torch.roll for the neighbours).   (..., N)"""
    lap = (torch.roll(phi, 1, -1) - 2 * phi + torch.roll(phi, -1, -1)) / dx**2
    return lap - m2 * phi - lam * phi**3


def mass_from_speed(energy_gev: float, dv_over_c: float) -> float:
    r"""|m| = E\sqrt{v² − 1} of the tachyon that would have speed v = 1 + δv at energy E."""
    v = 1 + dv_over_c
    return energy_gev * (v**2 - 1) ** 0.5


def speed_excess(energy_gev, mu: float):
    r"""(v − c)/c = \sqrt{1 + μ²/E²} − 1 for a tachyon of mass |m| = μ at energy E (float or tensor)."""
    return (1 + (mu / energy_gev) ** 2) ** 0.5 - 1
