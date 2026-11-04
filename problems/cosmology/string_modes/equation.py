r"""Physics → tensor translation: the bosonic string in light-cone gauge.

In light-cone gauge the Nambu–Goto string reduces to D − 2 = 24 free transverse fields Xⁱ(τ, σ) obeying the wave
equation (∂τ² − ∂σ²) Xⁱ = 0. In the centre-of-mass frame (p = 0, x = 0), with α′ the Regge slope:

  closed (σ ∈ [0, 2π)):  Xⁱ = √(2α′) Σ_{n>0} (1/n) Re[ i α_nⁱ e^{−in(τ−σ)} + i α̃_nⁱ e^{−in(τ+σ)} ]
  open   (σ ∈ [0, π]):   Xⁱ = 2√(2α′) Σ_{n>0} (1/n) Re[ i α_nⁱ e^{−inτ} ] cos nσ

Both are batched mode sums: a phase tensor E_{tsn} (time t, point s, mode n) contracted with the amplitudes α_{nd}
over the repeated mode index n (einsum 'tsn,nd->tsd'). The oscillator level is the energy in the modes,

  N = Σ_{n>0} α_{−n}·α_n = Σ_n |α_n|²   (quantum: Σ n k_n with k_n quanta in mode n)
    = (1/4πα′) ∫ (Ẋ² + X′²) dσ          (Parseval; closed strings give N + Ñ)

and the quantum mass formulae (D = 26, normal-ordering intercept a = 1) are

  open:  α′M² = N − 1            closed:  α′M² = 2(N + Ñ − 2) = 4(N − 1),  with level matching N = Ñ.

The number of states at level N is the coefficient of qᴺ in ∏_{n≥1} (1 − qⁿ)^{−24} (24 colours of oscillator).
"""

import math

import torch

from .tensors import ALPHA_PRIME


def _phases(tau: torch.Tensor, sigma: torch.Tensor, n: torch.Tensor, sign: int) -> torch.Tensor:
    r"""E_{tsn} = e^{−in(τ_t − sign·σ_s)}: right movers (sign +1) or left movers (sign −1).   → (T, S, N) complex"""
    arg = n[None, None, :] * (tau[:, None, None] - sign * sigma[None, :, None])
    return torch.polar(torch.ones_like(arg), -arg)


def _modes(alpha: torch.Tensor) -> torch.Tensor:
    return torch.arange(1, alpha.shape[0] + 1, dtype=alpha.real.dtype, device=alpha.device)  # n = 1 … N_modes


def embedding(tau, sigma, alpha, alpha_t, kind):
    r"""Xⁱ(τ, σ) for every time and point at once.   tau (T,), sigma (S,), alpha (N, 3) complex → X (T, S, 3)

    The mode index n is summed in the einsum; division by n weights high modes down (their energy n|·|² is equal).
    """
    n = _modes(alpha)
    if kind == "closed":
        X = torch.einsum("tsn,nd->tsd", 1j * _phases(tau, sigma, n, +1) / n, alpha)
        X = X + torch.einsum("tsn,nd->tsd", 1j * _phases(tau, sigma, n, -1) / n, alpha_t)
        return math.sqrt(2 * ALPHA_PRIME) * X.real
    e = torch.polar(torch.ones_like(tau[:, None] * n), -tau[:, None] * n)  # (T, N) e^{−inτ}
    c = torch.cos(sigma[:, None] * n).to(e.dtype)  # (S, N) cos nσ
    return 2 * math.sqrt(2 * ALPHA_PRIME) * torch.einsum("tn,sn,nd->tsd", 1j * e / n, c, alpha).real


def derivatives(tau, sigma, alpha, alpha_t, kind):
    """(Ẋ, X′), both (T, S, 3): the same mode sums differentiated term by term."""
    n = _modes(alpha)
    k = math.sqrt(2 * ALPHA_PRIME)
    if kind == "closed":
        right = torch.einsum("tsn,nd->tsd", _phases(tau, sigma, n, +1), alpha).real  # ∂X_R/∂τ
        left = torch.einsum("tsn,nd->tsd", _phases(tau, sigma, n, -1), alpha_t).real
        return k * (right + left), k * (left - right)  # X′ = ∂σ: right movers flip sign
    e = torch.polar(torch.ones_like(tau[:, None] * n), -tau[:, None] * n)  # (T, N)
    c = torch.cos(sigma[:, None] * n).to(e.dtype)
    s = torch.sin(sigma[:, None] * n).to(e.dtype)
    X_dot = 2 * k * torch.einsum("tn,sn,nd->tsd", e, c, alpha).real
    X_prime = -2 * k * torch.einsum("tn,sn,nd->tsd", 1j * e, s, alpha).real
    return X_dot, X_prime


def _integrate(f: torch.Tensor, sigma: torch.Tensor, kind: str) -> torch.Tensor:
    """∫ f dσ over the string, f (T, S) → (T,): rectangle rule on the closed loop, trapezoid on [0, π]."""
    if kind == "closed":
        return f.sum(-1) * (2 * math.pi / len(sigma))
    return torch.trapezoid(f, sigma, dim=-1)


def level_from_field(X_dot, X_prime, sigma, kind):
    r"""N (+ Ñ) = (1/4πα′) ∫ (Ẋⁱ Ẋⁱ + X′ⁱ X′ⁱ) dσ, i summed.   (T, S, 3) → (T,)"""
    density = torch.einsum("tsi,tsi->ts", X_dot, X_dot) + torch.einsum("tsi,tsi->ts", X_prime, X_prime)
    return _integrate(density, sigma, kind) / (4 * math.pi * ALPHA_PRIME)


def sigma_momentum(X_dot, X_prime, sigma):
    r"""P_σ = ∫ Ẋⁱ X′ⁱ dσ = 2πα′ (Ñ − N): generates σ translations; level matching says it vanishes.   → (T,)"""
    return _integrate(torch.einsum("tsi,tsi->ts", X_dot, X_prime), sigma, "closed")


def oscillator_level(alpha: torch.Tensor) -> float:
    r"""N = Σ_{n,i} |α_nⁱ|² (the classical value of Σ α_{−n}·α_n)."""
    return float((alpha.abs() ** 2).sum())


def level_degeneracies(n_max: int, colours: int = 24) -> torch.Tensor:
    r"""Coefficients d_0 … d_{n_max} of ∏_{n≥1} (1 − qⁿ)^{−colours}, as a tensor power series.

    log ∏ (1 − qⁿ)^{−c} = c Σ_m σ₁(m)/m qᵐ with σ₁(m) the sum of the divisors of m (a broadcast divisibility mask).
    Truncated power series multiply like lower-triangular Toeplitz matrices, so exp of the series is matrix_exp.
    """
    m = torch.arange(1, n_max + 1, dtype=torch.float64)
    divides = (m[:, None] % m[None, :] == 0).to(torch.float64)  # (M, M): d | m
    sigma1 = divides @ m  # (M,) divisor sums
    log_series = torch.cat([torch.zeros(1, dtype=torch.float64), colours * sigma1 / m])  # (n_max + 1,)
    i = torch.arange(n_max + 1)
    lag = i[:, None] - i[None, :]  # Toeplitz index i − j
    toeplitz = torch.where(lag >= 0, log_series[lag.clamp(min=0)], torch.zeros(()))
    return torch.linalg.matrix_exp(toeplitz)[:, 0]


def open_mass_squared(N: torch.Tensor) -> torch.Tensor:
    """α′M² = N − 1 in units of 1/α′ (intercept a = 1 in D = 26)."""
    return (N - 1) / ALPHA_PRIME


def closed_mass_squared(N: torch.Tensor) -> torch.Tensor:
    """α′M² = 2(N + Ñ − 2) = 4(N − 1) for level-matched states N = Ñ."""
    return 4 * (N - 1) / ALPHA_PRIME


def circle_mass_squared(R, n, w, N, N_t):
    r"""Closed string with one dimension on a circle of radius R: momentum n/R, winding wR/α′.

    M² = (n/R)² + (wR/α′)² + (2/α′)(N + Ñ − 2), with level matching N − Ñ = n w. All arguments broadcast.
    Swapping R ↔ α′/R and n ↔ w leaves M² unchanged: T-duality.
    """
    return (n / R) ** 2 + (w * R / ALPHA_PRIME) ** 2 + 2 * (N + N_t - 2) / ALPHA_PRIME
