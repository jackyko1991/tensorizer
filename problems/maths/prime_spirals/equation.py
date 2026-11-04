r"""Mathematics → tensor translation: the Hardy–Littlewood conjectures as tensor operations.

1. Conjecture F / Bateman–Horn. For an irreducible quadratic f(k) = a k² + b k + c the number of primes among
   f(1), …, f(K) should be

        Q_f(K) ≈ C(f) Σ_{k≤K} 1 / ln f(k),      C(f) = Π_p (1 − w_f(p)/p) / (1 − 1/p),

   where w_f(p) is the number of roots of f mod p. For an odd prime p ∤ a, w_f(p) = 1 + (D/p) with D = b² − 4ac and
   the Legendre symbol from Euler's criterion (D/p) ≡ D^{(p−1)/2} mod p. The (F, P) table of w over F quadratics and
   P primes is built in one go and the product over p is a sum of logs over the prime axis.

2. Twin primes (Hardy–Littlewood): π₂(x) ≈ 2 C₂ ∫₂^x dt / ln² t,  C₂ = Π_{p>2} p(p−2)/(p−1)² = 0.6601618…

3. Goldbach: r(N) = #{(p, q) : p + q = N} is the self-convolution of the prime indicator, r = 1_P * 1_P (done by FFT),
   predicted to be r(N) ≈ 2 C₂ Π_{p | N, p > 2} (p−1)/(p−2) · ∫₂^{N−2} dt / (ln t · ln(N − t)).

4. Cramér's model: n is "prime" independently with probability 1/ln n, so gaps g_n near x are roughly exponential
   with mean ln x: P(g/ln p > s) ≈ e^{−s}.
"""

import math

import torch


def quadratic(coef: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """f_i(k_j) = a_i k_j² + b_i k_j + c_i for every quadratic i and every k_j.   (F, 3), (K,) → (F, K) int64"""
    a, b, c = coef[:, 0:1], coef[:, 1:2], coef[:, 2:3]
    return a * k[None, :] ** 2 + b * k[None, :] + c


def _power_mod(base: torch.Tensor, exp: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    """base^exp mod mod, elementwise, by repeated squaring (all entries < 2^31 so products fit in int64)."""
    result = torch.ones_like(base)
    base = base % mod
    exp = exp.clone()
    while bool((exp > 0).any()):
        odd = (exp % 2) == 1
        result = torch.where(odd, (result * base) % mod, result)
        base = (base * base) % mod
        exp = exp // 2
    return result


def root_counts(coef: torch.Tensor, primes: torch.Tensor, brute_below: int = 50) -> torch.Tensor:
    """w_f(p): number of x mod p with f(x) ≡ 0, for every quadratic and prime.   (F, 3), (P,) → (F, P)

    Small primes (where p might divide 2a) by trying every residue; larger ones by Euler's criterion."""
    coef, primes = coef.long(), primes.long()
    small = primes[primes < brute_below]
    x = torch.arange(brute_below, device=coef.device)
    vals = quadratic(coef, x)[:, None, :] % small[None, :, None]  # (F, P_small, X)
    in_range = x[None, None, :] < small[None, :, None]  # only residues 0 … p−1
    w_small = ((vals == 0) & in_range).sum(dim=2)
    large = primes[primes >= brute_below]
    disc = coef[:, 1] ** 2 - 4 * coef[:, 0] * coef[:, 2]  # D = b² − 4ac   (F,)
    d_mod = disc[:, None] % large[None, :]  # (F, P_large), in 0 … p−1
    euler = _power_mod(d_mod, ((large - 1) // 2)[None, :].expand_as(d_mod), large[None, :].expand_as(d_mod))
    legendre = torch.where(euler == large[None, :] - 1, -1, euler)  # 1, 0 or −1
    return torch.cat([w_small, 1 + legendre], dim=1)


def bateman_horn_constant(coef: torch.Tensor, primes: torch.Tensor) -> torch.Tensor:
    r"""C(f) = Π_p (1 − w_f(p)/p) / (1 − 1/p), truncated at the largest prime given.   (F, 3) → (F,) float64

    Zero when some prime divides every value (w_f(p) = p)."""
    w = root_counts(coef, primes).double()
    p = primes.double()[None, :]
    factor = (1 - w / p) / (1 - 1 / p)  # (F, P)
    dead = (factor <= 0).any(dim=1)
    log_c = torch.log(factor.clamp(min=1e-300)).sum(dim=1)  # product over p as a sum of logs
    return torch.where(dead, 0.0, torch.exp(log_c))


def twin_constant(primes: torch.Tensor) -> float:
    r"""C₂ = Π_{p>2} p(p − 2)/(p − 1)²."""
    p = primes[primes > 2].double()
    return torch.exp(torch.log(p * (p - 2) / (p - 1) ** 2).sum()).item()


def log_integral(x: torch.Tensor, power: int, n_grid: int = 4000) -> torch.Tensor:
    """∫₂^x dt / ln^power t for every x (trapezoid rule on a log-spaced grid).   (X,) → (X,)"""
    u = torch.linspace(0, 1, n_grid, dtype=torch.float64, device=x.device)[None, :]
    t = 2 * (x.double()[:, None] / 2) ** u  # (X, G) from 2 to x
    return torch.trapezoid(1 / torch.log(t) ** power, t, dim=1)


def twin_prediction(x: torch.Tensor, c2: float) -> torch.Tensor:
    """Hardy–Littlewood: π₂(x) ≈ 2 C₂ ∫₂^x dt / ln² t."""
    return 2 * c2 * log_integral(x, 2)


def twin_count(is_prime: torch.Tensor) -> torch.Tensor:
    """π₂(x) for every x: number of p ≤ x with p and p + 2 both prime.   (X,) bool → (X,) int64"""
    starts = torch.zeros_like(is_prime)
    starts[:-2] = is_prime[:-2] & is_prime[2:]
    return torch.cumsum(starts.long(), dim=0)


def goldbach_count(is_prime: torch.Tensor) -> torch.Tensor:
    """r(N) = Σ_p 1_P(p) 1_P(N − p): the self-convolution of the prime indicator, by FFT.   (X,) → (2X − 1,) int64"""
    n = 2 * len(is_prime) - 1
    spectrum = torch.fft.rfft(is_prime.double(), n=n)
    return torch.fft.irfft(spectrum * spectrum, n=n).round().long()


def goldbach_prediction(n_even: torch.Tensor, primes: torch.Tensor, c2: float, n_grid: int = 400) -> torch.Tensor:
    r"""Hardy–Littlewood prediction for r(N) at every even N.   (E,) int64 → (E,) float64

    The singular series Π_{p|N, p>2} (p−1)/(p−2) is built by multiplying in (p−1)/(p−2) on every multiple of p."""
    n_max = int(n_even.max())
    series = torch.ones(n_max + 1, dtype=torch.float64, device=n_even.device)
    for p in primes[(primes > 2) & (primes <= n_max)].tolist():
        series[p::p] *= (p - 1) / (p - 2)
    n = n_even.double()[:, None]
    t = 2 + (n - 4) * torch.linspace(0, 1, n_grid, dtype=torch.float64, device=n_even.device)[None, :]  # (E, G)
    integral = torch.trapezoid(1 / (torch.log(t) * torch.log(n - t)), t, dim=1)
    return 2 * c2 * series[n_even] * integral


def cramer_model(n_max: int, seed: int, device: torch.device) -> torch.Tensor:
    """Cramér's random primes: n ≥ 3 is kept with probability 1/ln n (2 is kept).   → (n_max + 1,) bool"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.arange(n_max + 1, dtype=torch.float64)
    prob = torch.where(n >= 3, 1 / torch.log(n.clamp(min=2)), 0.0)
    keep = torch.rand(n_max + 1, generator=gen, dtype=torch.float64) < prob
    keep[2] = True
    return keep.to(device)


def normalised_gaps(is_prime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaps g_n = p_{n+1} − p_n and g_n / ln p_n for every consecutive pair.   → (G,), (G,)"""
    p = torch.nonzero(is_prime).flatten()
    gaps = p[1:] - p[:-1]
    return gaps, gaps.double() / torch.log(p[:-1].double().clamp(min=2))


def ln_squared_ratio(is_prime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Record (maximal) gaps and their ratio g / ln² p (Cramér's conjecture: limsup ≤ 1, Granville: 2e^{−γ}?)."""
    p = torch.nonzero(is_prime).flatten()
    gaps = p[1:] - p[:-1]
    record = gaps > torch.cummax(torch.cat([gaps.new_zeros(1), gaps[:-1]]), dim=0).values
    p_rec = p[:-1][record].double()
    return p_rec, gaps[record].double() / torch.log(p_rec) ** 2


EULER_GAMMA = 0.5772156649015329
GRANVILLE = 2 * math.exp(-EULER_GAMMA)  # 1.1229…
