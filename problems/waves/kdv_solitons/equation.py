r"""Physics → tensor translation.

The Korteweg–de Vries equation u_t + 6 u u_x + u_xxx = 0, with 6 u u_x = 3 (u²)_x, in Fourier space
(∂_x → ik, so ∂_x³ → −ik³):

    \partial_t \hat u_k = i k^3 \hat u_k - 3 i k \, F_{kj} (u_j u_j)

(F_{kj} is the discrete Fourier transform, j summed.) The first term is linear and stiff; it is solved exactly
with the integrating factor \hat v_k = e^{-ik^3 t} \hat u_k, and RK4 handles the nonlinear term.

Single soliton of speed c (height c/2):   u = (c/2) sech²(√c/2 (x − c t)).
Conserved: mass M = ∫ u dx and momentum P = ∫ u² dx.
"""

import math

import torch


def soliton(x: torch.Tensor, c: float, x0: float) -> torch.Tensor:
    """u = (c/2) sech²(√c/2 (x − x0)).   (J,) → (J,)"""
    return 0.5 * c / torch.cosh(0.5 * math.sqrt(c) * (x - x0)) ** 2


def nonlinear(u_hat: torch.Tensor, k: torch.Tensor, dealias: torch.Tensor) -> torch.Tensor:
    r"""−3ik F[u²] with the 2/3 rule.   u_hat, k, dealias: (K,) → (K,)"""
    u = torch.fft.ifft(u_hat).real
    return -3j * k * torch.fft.fft(u * u) * dealias


def if_rk4_step(u_hat: torch.Tensor, k: torch.Tensor, dealias: torch.Tensor, dt: float) -> torch.Tensor:
    r"""One integrating-factor RK4 step: the dispersion e^{ik³Δt} is applied exactly, RK4 does the rest."""
    E = torch.exp(0.5j * k**3 * dt)  # half-step propagator of the linear part
    E2 = E * E
    a = dt * nonlinear(u_hat, k, dealias)
    b = dt * nonlinear(E * (u_hat + a / 2), k, dealias)
    c = dt * nonlinear(E * u_hat + b / 2, k, dealias)
    d = dt * nonlinear(E2 * u_hat + E * c, k, dealias)
    return E2 * u_hat + (E2 * a + 2 * E * (b + c) + d) / 6


def mass(u: torch.Tensor, dx: float) -> torch.Tensor:
    """M = Σ_j u_j Δx.   (…, J) → (…)"""
    return u.sum(dim=-1) * dx


def momentum(u: torch.Tensor, dx: float) -> torch.Tensor:
    """P = Σ_j u_j u_j Δx (j summed).   (…, J) → (…)"""
    return torch.einsum("...j,...j->...", u, u) * dx


def phase_shifts(c1: float, c2: float) -> tuple[float, float]:
    """Exact collision shifts (forward for the faster, backward for the slower), κ = √c / 2."""
    k1, k2 = math.sqrt(c1) / 2, math.sqrt(c2) / 2
    log = math.log((k1 + k2) / abs(k1 - k2))
    return log / k1, log / k2
