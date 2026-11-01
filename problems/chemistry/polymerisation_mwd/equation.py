r"""Physics → tensor translation.

STEP GROWTH (A–B monomers, e.g. a hydroxy acid making a polyester): any chain end A can react with any end B.
With Flory's equal-reactivity assumption every chain has one A and one B end and the same rate constant k, so
the population balance for P_x (chains of length x) is the Smoluchowski equation with a constant kernel:

    dP_x/dτ = Σ_y P_y P_{x-y} - 2 P_x N,       N = Σ_y P_y,       τ = k N_0 t

(y summed; the first term forms x from y + (x - y), the second loses x through either of its two ends).
Its exact solution is Flory's most probable distribution at conversion p = 1 - N/N_0 = τ/(1 + τ):

    n(x) = (1 - p) p^{x-1}        (number fraction),       w(x) = x (1 - p)² p^{x-1}    (weight fraction)

    X_n = 1/(1 - p)   (Carothers),     X_w = (1 + p)/(1 - p),     Đ = X_w / X_n = 1 + p  →  2

LIVING CHAIN GROWTH (anionic, ATRP, …): every initiator starts one chain at once, chains only add monomer,
nothing terminates. Measured in the kinetic chain length ν = ∫ k_p [M] dt (monomers added per chain):

    dP_x/dν = P_{x-1} - P_x       ⇒      P_x = e^{-ν} ν^{x-1} / (x-1)!    (Poisson)

    X_n = 1 + ν,       Đ = 1 + ν / (1 + ν)²  ≈  1 + 1/X_n

Both mechanisms are integrated together in one (S, L) tensor (S = 2 mechanisms, L chain lengths), with the
clock s = X_n - 1 (s = τ for step growth, s = ν for living growth), so every frame compares equal X_n.
"""

import torch

# --- step growth ---


def convolution(P: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""C_x = Σ_y P_y P_{x-y}   (y summed):  T_{xy} = P_{x-y} by gathering, then einsum('xy,y->x').   (L,) → (L,)."""
    T = P[idx] * mask  # (L, L) Toeplitz matrix of partner populations
    return torch.einsum("xy,y->x", T, P)


def step_growth_rhs(P: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""dP_x/dτ = Σ_y P_y P_{x-y} - 2 P_x N.   (L,) → (L,)."""
    return convolution(P, idx, mask) - 2.0 * P * P.sum()


def flory_number(x: torch.Tensor, p) -> torch.Tensor:
    """n(x) = (1 - p) p^{x-1}: the fraction of chains that have length x.   x (L,), p scalar or (F, 1)."""
    return (1 - p) * p ** (x - 1)


def flory_weight(x: torch.Tensor, p) -> torch.Tensor:
    """w(x) = x (1 - p)² p^{x-1}: the fraction of monomer units that sit in chains of length x."""
    return x * (1 - p) ** 2 * p ** (x - 1)


def carothers(p):
    """X_n = 1 / (1 - p): number-average degree of polymerisation at conversion p."""
    return 1.0 / (1.0 - p)


# --- living chain growth ---


def living_rhs(P: torch.Tensor) -> torch.Tensor:
    r"""dP_x/dν = P_{x-1} - P_x  (a chain of length x-1 gains a monomer, one of length x leaves).   (L,) → (L,)."""
    shifted = torch.cat([torch.zeros_like(P[:1]), P[:-1]])  # P_{x-1}, with P_0 = 0
    return shifted - P


def poisson(x: torch.Tensor, nu) -> torch.Tensor:
    r"""P_x = e^{-ν} ν^{x-1} / (x-1)!, written with lgamma so it stays finite for long chains."""
    nu = torch.as_tensor(nu, dtype=x.dtype, device=x.device)
    return torch.exp(-nu + torch.xlogy(x - 1, nu) - torch.lgamma(x))  # xlogy: 0 · log 0 = 0 at ν = 0


# --- both: the (S, L) state ---


def rhs(P: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """d/ds of the (2, L) state: row 0 step growth (s = τ), row 1 living growth (s = ν)."""
    return torch.stack([step_growth_rhs(P[0], idx, mask), living_rhs(P[1])])


def moments(P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""μ_k = Σ_x x^k P_x for k = 0, 1, 2 (x summed):  einsum('...x,kx->...k').   (..., L) → (..., 3)."""
    powers = torch.stack([torch.ones_like(x), x, x**2])  # (3, L)
    return torch.einsum("...x,kx->...k", P, powers)


def averages(P: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X_n = μ1/μ0, X_w = μ2/μ1 and the dispersity Đ = X_w / X_n."""
    m = moments(P, x)
    xn, xw = m[..., 1] / m[..., 0], m[..., 2] / m[..., 1]
    return xn, xw, xw / xn


def weight_fraction(P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """w_x = x P_x / Σ_y y P_y."""
    xP = x * P
    return xP / xP.sum(-1, keepdim=True)
