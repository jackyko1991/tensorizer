r"""Physics → tensor translation.

A reaction A → products with rate law of order n and a temperature-dependent rate constant (Arrhenius):

    r_{ot} = k_t [A]_{ot}^{n_o},        k_t = A e^{-E_a / (R T_t)},        d[A]_{ot}/dt = -r_{ot}

o runs over the three orders n = 0, 1, 2 and t over the temperatures; no index is summed in the rate law,
every (o, t) pair is its own experiment. Each run has its own clock, t = τ t_end,ot with τ ∈ [0, 1], so the
integrator steps in τ:

    d[A]_{ot}/dτ = -t_end,ot k_t [A]_{ot}^{n_o}

Fitting a straight line y_s ≈ β_0 + β_1 x_s by least squares is also tensor algebra (normal equations):

    G_{ij} = w_s X_{si} X_{sj},    b_i = w_s X_{si} y_s,    G_{ij} β_j = b_i        (s and j summed)

with design matrix X_{si} = (1, x_s) and weights w_s (1 for a point that counts, 0 for one left out).
"""

import math

import torch

R_GAS = 8.314462618  # J/(mol K)


def rate_constant(T: torch.Tensor, A: float, Ea: float) -> torch.Tensor:
    """Arrhenius law k_t = A exp(-E_a / (R T_t)).   T: (T,) → (T,).

    Units of k: M^(1-n) s⁻¹; the same number is used for every order n.
    """
    return A * torch.exp(-Ea / (R_GAS * T))


def rate(conc: torch.Tensor, k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """r_{ot} = k_t [A]_{ot}^{n_o}.   conc: (O, T), k: (T,), n: (O,) → (O, T).  Nothing is summed."""
    return torch.einsum("t,ot->ot", k, conc ** n[:, None])


def rhs(conc: torch.Tensor, k: torch.Tensor, n: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
    """d[A]/dτ = -t_end r for the state (O, T) in reduced time τ = t / t_end."""
    return -t_end * rate(conc, k, n)


def integrated_law(t: torch.Tensor, A0: float, k: torch.Tensor, n: int) -> torch.Tensor:
    """The textbook integrated rate laws (exact solutions), for comparison and for the half-life method.

    n = 0:  [A] = [A]_0 - k t               (straight line in [A])
    n = 1:  [A] = [A]_0 e^{-k t}             (straight line in ln[A])
    n = 2:  [A] = [A]_0 / (1 + [A]_0 k t)    (straight line in 1/[A])
    """
    if n == 0:
        return A0 - k * t
    if n == 1:
        return A0 * torch.exp(-k * t)
    return A0 / (1.0 + A0 * k * t)


def time_to_fraction(k: torch.Tensor, n: torch.Tensor, A0: float, f: float) -> torch.Tensor:
    """Time for [A] to fall from [A]_0 to f [A]_0, from the integrated laws (n may be a tensor of orders).

    n = 0: (1 - f) [A]_0 / k     n = 1: ln(1/f) / k     n = 2: (1/f - 1) / ([A]_0 k)
    """
    zero = (1.0 - f) * A0 / k
    first = math.log(1.0 / f) / k
    second = (1.0 / f - 1.0) / (A0 * k)
    return torch.where(n == 0, zero, torch.where(n == 1, first, second))


def linearize(conc: torch.Tensor) -> torch.Tensor:
    """Stack the three candidate straight-line plots: g = ([A], ln[A], 1/[A]).   (...) → (3, ...)."""
    safe = conc.clamp_min(1e-6)  # a noisy point can dip below zero near the end
    return torch.stack([conc, torch.log(safe), 1.0 / safe])


def least_squares(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    r"""Fit y ≈ β_0 + β_1 x along the last axis, batched over any leading axes.   Returns β (..., 2).

    X_{si} = (1, x_s);  G_{ij} = w_s X_{si} X_{sj};  b_i = w_s X_{si} y_s;  solve G β = b   (s summed).
    """
    if w is None:
        w = torch.ones_like(y)
    X = torch.stack([torch.ones_like(x), x], dim=-1)  # (..., S, 2)
    G = torch.einsum("...s,...si,...sj->...ij", w, X, X)  # (..., 2, 2)
    b = torch.einsum("...s,...si,...s->...i", w, X, y)  # (..., 2)
    return torch.linalg.solve(G, b)


def r_squared(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Coefficient of determination R² = 1 - SS_res / SS_tot of the fitted line (1 = perfectly straight)."""
    if w is None:
        w = torch.ones_like(y)
    y_fit = beta[..., :1] + beta[..., 1:] * x
    mean = (w * y).sum(-1, keepdim=True) / w.sum(-1, keepdim=True)
    ss_res = (w * (y - y_fit) ** 2).sum(-1)
    ss_tot = (w * (y - mean) ** 2).sum(-1)
    return 1.0 - ss_res / ss_tot


def arrhenius_fit(T: torch.Tensor, k: torch.Tensor) -> tuple[float, float]:
    """Straight line through the Arrhenius plot: ln k_t = ln A - (E_a/R)(1/T_t).   Returns (E_a in J/mol, A)."""
    beta = least_squares(1.0 / T, torch.log(k))  # β_0 = ln A, β_1 = -E_a / R
    return -R_GAS * beta[1].item(), math.exp(beta[0].item())


def order_from_half_lives(t_half_1: torch.Tensor, t_half_2: torch.Tensor) -> torch.Tensor:
    """n = 1 + log₂(t½⁽²⁾ / t½⁽¹⁾): since t½ ∝ [A]_0^{1-n}, the second half-life (starting from [A]_0/2)
    is 2^{n-1} times the first: shorter for n = 0, equal for n = 1, twice as long for n = 2."""
    return 1.0 + torch.log2(t_half_2 / t_half_1)
