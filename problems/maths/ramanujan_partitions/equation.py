r"""Mathematics → tensor translation: the partition function p(n) and the formulas that describe it.

1. Generating function and Euler's pentagonal theorem:

        Σ_n p(n) qⁿ = Π_k 1/(1 − q^k),      Π_k (1 − q^k) = Σ_k (−1)^k q^{k(3k−1)/2},

   so multiplying the two series and reading off qⁿ gives p(n) = Σ_j s_j p(n − g_j): a signed sum over the
   O(√n) generalised pentagonal numbers g_j. The sum over j is one tensor reduction; the residues mod several m are
   computed in the same step, one column per modulus.

2. Hardy–Ramanujan (1918):  p(n) ~ exp(π √(2n/3)) / (4 n √3).

3. Rademacher (1937), an exact convergent series:

        p(n) = 1/(π√2) Σ_k A_k(n) √k · d/dn [ sinh((π/k) √(2/3 (n − 1/24))) / √(n − 1/24) ],
        A_k(n) = Σ_{0≤h<k, gcd(h,k)=1} exp(π i s(h,k) − 2π i n h / k),

   with the Dedekind sum s(h, k) = Σ_{r=1}^{k−1} (r/k) ((h r / k)) and the sawtooth ((x)) = x − ⌊x⌋ − 1/2 (0 at
   integers). The (N, K, H) table of phases over n, k and h is summed over h.
"""

import math

import torch

C = math.pi * math.sqrt(2 / 3)  # the growth constant: log p(n) ≈ C √n


def euler_step(p: torch.Tensor, n: int, g: torch.Tensor, s: torch.Tensor, moduli: torch.Tensor | None) -> torch.Tensor:
    """p(n) = Σ_j s_j p(n − g_j) over the pentagonal numbers g_j ≤ n (repeated j ⇒ summed).

    p: (N + 1,) or (N + 1, M) with one column per modulus; g, s: (J,) → (,) or (M,)"""
    terms = p[n - g]  # (J,) or (J, M): the earlier values this one depends on
    sign = s.to(p.dtype).reshape(-1, *([1] * (p.dim() - 1)))
    total = (sign * terms).sum(dim=0)
    return total if moduli is None else total % moduli


def garner_digits(residues: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Chinese remainder theorem (Garner's form) for every n at once: find digits v with
    x = v_0 + v_1 m_0 + v_2 m_0 m_1 + …, 0 ≤ v_i < m_i, from the residues x mod m_i.   (N, M) → (N, M)

    v_i = (…((r_i − v_0) m_0⁻¹ − v_1) m_1⁻¹ − …) mod m_i; every product stays below 2^62."""
    v = residues.clone()
    mods = m.tolist()
    for i in range(1, len(mods)):
        for j in range(i):
            inv = pow(mods[j], -1, mods[i])  # m_j⁻¹ mod m_i (a plain integer)
            v[:, i] = (v[:, i] - v[:, j]) % mods[i] * inv % mods[i]
    return v


def digits_to_float(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """x = v_0 + m_0 (v_1 + m_1 (v_2 + …)) by Horner's rule in float64.   (N, M) → (N,)"""
    x = v[:, -1].double()
    for i in range(v.shape[1] - 2, -1, -1):
        x = v[:, i].double() + float(m[i]) * x
    return x


def digits_to_int(v: torch.Tensor, m: torch.Tensor) -> int:
    """The same Horner sum for one n, exactly, with Python integers.   (M,) → int"""
    x = 0
    for d, mi in zip(reversed(v.tolist()), reversed(m.tolist())):
        x = d + mi * x
    return x


def hardy_ramanujan(n: torch.Tensor) -> torch.Tensor:
    """Hardy–Ramanujan asymptotic exp(π√(2n/3)) / (4n√3).   (N,) float64 → (N,)"""
    return torch.exp(C * torch.sqrt(n)) / (4 * n * math.sqrt(3))


def dedekind_sum(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """s(h, k) = Σ_{r=1}^{k−1} (r/k) ((h r / k)) elementwise (h, k broadcast together; repeated r ⇒ summed)."""
    h, k = torch.broadcast_tensors(h.long(), k.long())
    r_max = max(int(k.max()), 2)
    r = torch.arange(1, r_max, device=h.device)  # (R,)
    hr = h[..., None] * r  # (..., R)
    frac = (hr % k[..., None]).double() / k[..., None]  # fractional part of h r / k
    saw = torch.where(frac == 0, 0.0, frac - 0.5)  # ((h r / k))
    in_range = r < k[..., None]
    return (torch.where(in_range, r.double() / k[..., None] * saw, 0.0)).sum(dim=-1)


def kloosterman_a(n: torch.Tensor, n_terms: int) -> torch.Tensor:
    """A_k(n) for k = 1 … n_terms and every n (a real Kloosterman-type sum, repeated h ⇒ summed).   (N,) → (N, K)"""
    k = torch.arange(1, n_terms + 1, device=n.device)  # (K,)
    h = torch.arange(0, n_terms, device=n.device)  # (H,)
    valid = (h[None, :] < k[:, None]) & (torch.gcd(h[None, :], k[:, None]) == 1)  # (K, H)
    s = dedekind_sum(h[None, :], k[:, None])  # (K, H)
    nh_mod_k = (n.long()[:, None, None] * h[None, None, :]) % k[None, :, None]  # exact before the 2π: (N, K, H)
    phase = math.pi * s[None] - 2 * math.pi * nh_mod_k.double() / k[None, :, None]
    return torch.where(valid[None], torch.cos(phase), 0.0).sum(dim=-1)


def rademacher_terms(n: torch.Tensor, n_terms: int) -> torch.Tensor:
    """The first n_terms terms of Rademacher's series for every n; their sum → p(n).   (N,) → (N, K) float64"""
    k = torch.arange(1, n_terms + 1, device=n.device, dtype=torch.float64)[None, :]  # (1, K)
    lam = torch.sqrt(n.double() - 1 / 24)[:, None]  # λ = √(n − 1/24): (N, 1)
    x = C * lam / k
    d_dn = ((C / k) * torch.cosh(x) / lam - torch.sinh(x) / lam**2) / (2 * lam)  # d/dn [sinh(Cλ/k)/λ]
    return kloosterman_a(n, n_terms) * torch.sqrt(k) * d_dn / (math.pi * math.sqrt(2))


def congruence_holds(p_mod: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Is p(a_i k + b_i) ≡ 0 in column i of the residue table for every a_i k + b_i ≤ N?

    p_mod: (N + 1, M), a, b: (M,) → held (M,) bool, number of k checked (M,)"""
    n_max = p_mod.shape[0] - 1
    k = torch.arange(0, n_max + 1, device=p_mod.device)
    n = a[:, None] * k[None, :] + b[:, None]  # (M, K)
    valid = n <= n_max
    col = torch.arange(len(a), device=p_mod.device)[:, None].expand_as(n)
    residue = p_mod[n.clamp(max=n_max), col]  # (M, K)
    return ((residue == 0) | ~valid).all(dim=1), valid.sum(dim=1)
