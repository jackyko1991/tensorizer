"""Frobenius traces → partial products and a rank estimate; Dirichlet coefficients → L(E, s) and its order at s = 1."""

import math

import numpy as np
import torch

from .equation import frobenius_traces
from .tensors import primes_up_to


def log_partial_products(coeffs: torch.Tensor, primes: torch.Tensor) -> torch.Tensor:
    """log ∏_{p ≤ X} N_p/p for X running over the primes, good primes only (N_p = p + 1 − a_p).   → (C, P)"""
    from .equation import discriminant

    a_p = frobenius_traces(coeffs, primes)  # (C, P)
    good = discriminant(coeffs)[:, None] % primes[None, :] != 0  # (C, P)
    p = primes.double()[None, :]
    terms = torch.where(good, torch.log((p + 1 - a_p) / p), 0.0)
    return torch.cumsum(terms, dim=1)


def rank_estimate(log_prod: torch.Tensor, primes: torch.Tensor, start: float = 0.5) -> torch.Tensor:
    """BSD heuristic ∏_{p ≤ X} N_p/p ≈ C (log X)^r: least-squares slope of log ∏ against log log X over
    X ∈ [X_max^start, X_max].   (C, P) → (C,)"""
    keep = primes >= primes[-1].double() ** start
    u = torch.log(torch.log(primes[keep].double()))  # (K,)
    v = log_prod[:, keep]  # (C, K)
    uc = u - u.mean()
    return torch.einsum("ck,k->c", v - v.mean(dim=1, keepdim=True), uc) / (uc @ uc)


def dirichlet_coefficients(coeffs: torch.Tensor, n_max: int) -> torch.Tensor:
    """a_n for n = 0 … n_max (a_0 unused) of every curve, from a_p by multiplicativity: start from a = δ_{n,1} and
    multiply in the local factor of each prime, Σ_k a_{p^k} p^{−ks}, with strided slices.   → (C, n_max + 1)"""
    from .equation import discriminant

    primes = primes_up_to(n_max, coeffs.device)
    a_p = frobenius_traces(coeffs, primes).double()  # (C, P)
    bad = discriminant(coeffs)[:, None] % primes[None, :] == 0
    a = torch.zeros(len(coeffs), n_max + 1, dtype=torch.float64, device=coeffs.device)
    a[:, 1] = 1.0
    for i, p in enumerate(primes.tolist()):
        local = [torch.ones_like(a_p[:, i]), a_p[:, i]]  # a_{p^0}, a_{p^1}
        while p ** len(local) <= n_max:  # a_{p^k} = a_p a_{p^{k−1}} − p a_{p^{k−2}}  (bad p: a_p^k)
            nxt = a_p[:, i] * local[-1] - torch.where(bad[:, i], 0.0, p * local[-2])
            local.append(nxt)
        new = a.clone()
        for k in range(1, len(local)):
            q = p**k
            new[:, q::q] += local[k][:, None] * a[:, 1 : n_max // q + 1]  # a_{q m} gets a_q a_m (p ∤ m so far)
        a = new
    return a


def l_function(coeffs: torch.Tensor, conductors: list[int], n_max: int, max_order: int = 4) -> dict:
    """Root number ε, the derivatives L^{(r)}(E, 1) for r ≡ parity of ε, the analytic rank and L(E, s) on a real grid.

    Λ(s) = ∫_1^∞ F(t)(t^{s−1} + ε t^{1−s}) dt with F(t) = Σ_n a_n e^{−2πnt/√N}; t = e^u turns every integral into
    a weighted sum over Gauss–Legendre nodes u_m: G_{nm} = e^{−2πn e^{u_m}/√N}, and Σ_n a_n G_{nm} is an einsum.
    """
    a = dirichlet_coefficients(coeffs, n_max)  # (C, n_max + 1)
    n = torch.arange(1, n_max + 1, dtype=torch.float64, device=coeffs.device)
    s_grid = torch.linspace(0.5, 1.5, 101, dtype=torch.float64, device=coeffs.device)
    nodes, weights = np.polynomial.legendre.leggauss(1500)
    m = torch.tensor((nodes + 1) / 2, device=coeffs.device)  # Gauss–Legendre nodes on (0, 1)
    gl_w = torch.tensor(weights / 2, device=coeffs.device)
    eps, derivs, curves = [], [], []
    for c, big_n in enumerate(conductors):
        scale = 2 * math.pi / math.sqrt(big_n)
        u_max = math.log(45 / scale)  # e^{−scale·e^u} < e^{−45} beyond
        u = u_max * m**2  # dense near u = 0, where the large-n terms live
        w = gl_w * 2 * u_max * m  # Gauss–Legendre weights in m times du/dm
        kernel = torch.exp(-scale * n[:, None] * torch.exp(u)[None, :])  # (n, U)
        f_u = torch.einsum("n,nu->u", a[c, 1:], kernel)  # F(e^u)
        # root number from the functional equation F(1/y) = ε y² F(y), at y = 1.2
        y = torch.tensor([1.2, 1 / 1.2], dtype=torch.float64, device=coeffs.device)
        f_y = torch.einsum("n,ny->y", a[c, 1:], torch.exp(-scale * n[:, None] * y[None, :]))
        e = int(torch.round(f_y[1] / (y[0] ** 2 * f_y[0])).item())
        weight = f_u * torch.exp(u) * w  # F(t) dt with t = e^u
        powers = torch.stack([u**r for r in range(max_order + 1)])  # (R, U): (ln t)^r
        d = 2 * scale * torch.einsum("ru,u->r", powers, weight)  # L^{(r)}(1), valid when ε = (−1)^r
        parity = torch.tensor([(-1) ** r == e for r in range(max_order + 1)], device=coeffs.device)
        derivs.append(torch.where(parity, d, 0.0))
        lam = torch.einsum(
            "su,u->s", torch.exp(torch.outer(s_grid - 1, u)) + e * torch.exp(torch.outer(1 - s_grid, u)), weight
        )
        curves.append(lam * scale**s_grid / torch.exp(torch.lgamma(s_grid)))  # L(s) = Λ(s) (2π/√N)^s / Γ(s)
        eps.append(e)
    derivs = torch.stack(derivs)  # (C, R)
    order = torch.argmax((derivs.abs() > 1e-6).int(), dim=1)  # first derivative that does not vanish
    fact = torch.tensor([math.factorial(r) for r in range(max_order + 1)], dtype=torch.float64, device=coeffs.device)
    leading = derivs.gather(1, order[:, None])[:, 0]
    return {
        "root_number": torch.tensor(eps),
        "derivatives": derivs,
        "order": order,
        "leading": leading,
        "taylor": leading / fact[order],
        "s": s_grid,
        "L": torch.stack(curves),
    }
