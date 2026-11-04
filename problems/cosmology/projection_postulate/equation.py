"""Postulates -> tensors: projection of a high-dimensional state, the projection race, dimension counts, Sorkin kappa.

Shapes in comments. Summed indices: j in the race (product over the other outcomes j != k, done as a sum of logs),
the quadrature variable t (integrated), and the configuration index S in the Sorkin combination.
"""

import math

import torch

# ---------- P1: concentration of measure ----------


def beta_params(d, D, field: str = "complex"):
    """||P Psi||^2 ~ Beta(a, b) for Psi uniform on the sphere of C^D (or R^D), P onto d coordinates.

    complex: each |z_i|^2 ~ Gamma(1) so a = d, b = D - d. real: z_i^2 ~ Gamma(1/2) so a = d/2, b = (D - d)/2.
    """
    D = torch.as_tensor(D, dtype=torch.float64)
    a = torch.full_like(D, float(d))
    a, b = (a, D - d) if field == "complex" else (a / 2, (D - d) / 2)
    return a, b


def beta_mean_var(a: torch.Tensor, b: torch.Tensor):
    """Mean a / (a + b) and variance a b / ((a + b)^2 (a + b + 1))."""
    s = a + b
    return a / s, a * b / (s**2 * (s + 1))


def beta_log_pdf(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log of x^(a-1) (1-x)^(b-1) / B(a, b), broadcast over x (C, N) and a, b (C, 1)."""
    log_norm = torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
    return log_norm + (a - 1) * torch.log(x) + (b - 1) * torch.log1p(-x)


def beta_cdf(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, n_grid: int = 40001) -> torch.Tensor:
    """Regularised incomplete beta I_x(a, b) for x (C, N), a, b (C,), by trapezoid quadrature.

    The grid x = sin^2(pi u / 2) is dense near both ends, which tames the x^(a-1) and (1-x)^(b-1) endpoints.
    """
    u = torch.linspace(0, 1, n_grid, dtype=torch.float64, device=x.device)  # cell edges
    g = torch.sin(math.pi * u / 2) ** 2  # (G,) x at the edges
    um = (u[1:] + u[:-1]) / 2  # midpoints: the endpoints themselves are never evaluated
    gm = torch.sin(math.pi * um / 2) ** 2
    dens = torch.exp(beta_log_pdf(gm, a[:, None], b[:, None])) * (math.pi / 2 * torch.sin(math.pi * um))  # (C, G-1)
    cum = torch.cat([torch.zeros_like(dens[:, :1]), torch.cumsum(dens, 1)], 1) * (u[1] - u[0])  # midpoint rule
    cum = cum / cum[:, -1:]  # integral over [0, 1] is 1
    i = torch.searchsorted(g, x.contiguous()).clamp(1, len(g) - 1)  # (C, N)
    g0, g1 = g[i - 1], g[i]
    c0, c1 = cum.gather(1, i - 1), cum.gather(1, i)
    t = ((x - g0) / (g1 - g0)).clamp(0, 1)
    return c0 + t * (c1 - c0)


def ks_distance(samples: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(C,) Kolmogorov-Smirnov distance sup |F_empirical - I_x(a, b)| for each row of samples (C, N)."""
    xs, _ = torch.sort(samples, dim=1)
    N = xs.shape[1]
    F = beta_cdf(xs, a, b)
    i = torch.arange(1, N + 1, dtype=torch.float64, device=xs.device)
    return torch.maximum(i / N - F, F - (i - 1) / N).amax(1)


# ---------- P2: the projection race ----------


def race_probabilities(w: torch.Tensor, a: torch.Tensor, n_grid: int = 4000) -> torch.Tensor:
    """P(k wins) for the race k* = argmax_k w_k / E_k with E_k ~ Gamma(a) i.i.d.; w (..., K), a broadcast to (...).

    E_k / w_k ~ Gamma(a, rate w_k), and k wins when its value is the smallest:
        P_k = int_0^oo f_a(t; w_k) prod_{j != k} Q(a, w_j t) dt,   Q = regularised upper incomplete gamma.
    The product over j is a sum of logs; t = e^s on a log grid (trapezoid). a = 1 gives P_k = w_k (Born).
    """
    a = torch.as_tensor(a, dtype=torch.float64, device=w.device).expand(w.shape[:-1])  # (...)
    w_pos = torch.where(w > 1e-15 * w.amax(-1, keepdim=True), w, torch.zeros_like(w))
    w_min = torch.where(w_pos > 0, w_pos, torch.full_like(w, math.inf)).amin(-1)  # smallest positive weight
    s_lo = torch.log(torch.tensor(1e-30, dtype=torch.float64)) / a - torch.log(w.amax(-1))  # (w t)^a tiny below
    s_hi = torch.log((a + 60 + 12 * a.sqrt()) / w_min)  # Gamma tail negligible above
    u = torch.linspace(0, 1, n_grid, dtype=torch.float64, device=w.device)
    s = s_lo[..., None] + (s_hi - s_lo)[..., None] * u  # (..., G)
    t = torch.exp(s)
    wt = w_pos[..., :, None] * t[..., None, :]  # (..., K, G)
    aa = a[..., None, None]
    log_q = torch.log(torch.special.gammaincc(aa.expand_as(wt), wt).clamp_min(1e-300))  # (..., K, G)
    log_f_t = aa * torch.log(wt) - wt - torch.lgamma(aa)  # log( f(t) * t ): density in s = ln t
    integrand = torch.exp(log_f_t + log_q.sum(-2, keepdim=True) - log_q)  # sum over j, minus j = k
    ds = (s_hi - s_lo)[..., None] / (n_grid - 1)
    return torch.trapezoid(integrand, dim=-1) * ds  # (..., K)


def race_two_outcomes(w: torch.Tensor, m: int) -> torch.Tensor:
    """Closed form for K = 2, integer m: P_1 = P(Beta(m, m) < w) = I_w(m, m) = P(Binomial(2m - 1, w) >= m)."""
    n = 2 * m - 1
    j = torch.arange(m, n + 1, dtype=torch.float64, device=w.device)  # (J,)
    log_c = torch.lgamma(torch.tensor(n + 1.0)) - torch.lgamma(j + 1) - torch.lgamma(n - j + 1)
    return torch.exp(log_c + j * torch.log(w[..., None]) + (n - j) * torch.log1p(-w[..., None])).sum(-1)


def race_frequencies(w: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Monte Carlo of the race: E (T, K) -> (K,) or E (T, K, M) -> (M, K) winning frequencies."""
    ratio = w[:, None] / E if E.dim() == 3 else w / E  # |c_k|^2 / E_k
    win = ratio.argmax(1)  # (T,) or (T, M)
    K = w.shape[-1]
    one_hot = torch.nn.functional.one_hot(win, K).double()  # (T, [M,] K)
    return one_hot.mean(0)


def total_variation(P: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(...,) total-variation distance (1/2) sum_k |P_k - w_k| from the Born rule."""
    return 0.5 * (P - w).abs().sum(-1)


# ---------- P5: how many dimensions ----------


def min_dim_born(d: int, K: int, m: int = 1) -> int:
    """Born statistics in the race: d observed dimensions plus m hidden complex dimensions for each of K outcomes."""
    return d + K * m


def log2_tensor_product_dim(N: int) -> int:
    """log2 of the dimension of N two-level systems, counted from an explicit Kronecker product."""
    v = torch.ones(1)
    for _ in range(N):
        v = torch.kron(v, torch.ones(2))
    return int(round(math.log2(v.numel())))


def log10_min_dim_entanglement(N: torch.Tensor) -> torch.Tensor:
    """log10 D_min = N log10 2 for N entangled two-level systems (D >= 2^N), safe for N up to 1e80 and beyond."""
    return N * math.log10(2.0)


# ---------- test against data: three-path interference ----------

SUBSETS = torch.tensor(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64
)  # (7, 3) open slits: A, B, C, AB, AC, BC, ABC


def slit_born_weights(amps: torch.Tensor) -> torch.Tensor:
    """(7, X + 1) Born weights for every set of open slits: |sum_s open_s phi_s(x)|^2 on X pixels, plus 'blocked'."""
    psi = torch.einsum("Ss,sx->Sx", SUBSETS.to(amps.dtype), amps)  # repeated slit index s => summed
    pix = psi.abs() ** 2
    return torch.cat([pix, 1 - pix.sum(-1, keepdim=True)], -1)


def sorkin_kappa(P: torch.Tensor, pixel: int) -> torch.Tensor:
    """kappa = epsilon / delta at one pixel; P (..., 7, X + 1) outcome probabilities for the seven slit sets.

    epsilon = P_ABC - P_AB - P_AC - P_BC + P_A + P_B + P_C,  delta = |I_AB| + |I_AC| + |I_BC|, I_AB = P_AB - P_A - P_B.
    """
    p = P[..., pixel]  # (..., 7)
    A, B, C, AB, AC, BC, ABC = p.unbind(-1)
    eps = ABC - AB - AC - BC + A + B + C
    delta = (AB - A - B).abs() + (AC - A - C).abs() + (BC - B - C).abs()
    return eps / delta
