r"""Mathematics → tensor translation.

1. Hodge numbers of a smooth hypersurface X of degree d in ℙ^{n+1} (Griffiths). The primitive middle cohomology is
   the Jacobian ring ℂ[x]/(∂F) in degree (q + 1)d − (n + 2); for the Fermat F = Σ x_i^d it has the monomial basis
   x^b with 0 ≤ b_i ≤ d − 2, so

        h^{n−q,q}_prim = [t^{(q+1)d − (n+2)}] (1 + t + … + t^{d−2})^{n+2},

   a coefficient of a polynomial power (repeated 1-D convolution). Off the middle row h^{p,p} = 1 (Lefschetz).

2. Complex tori of dimension g: h^{p,q} = C(g, p) C(g, q), an outer product. Products: Künneth, h(X × Y) = h(X) * h(Y)
   as a 2-D convolution. Euler characteristic χ = Σ (−1)^{p+q} h^{p,q}.

3. Hodge classes on Fermat varieties (Shioda, 1979). V(a) has type (n−q, q) with q + 1 = Σ_i ⟨a_i/d⟩, and it lies in
   H^n(X, ℚ) ∩ H^{n/2,n/2} for every Galois conjugate exactly when

        Σ_i ⟨t a_i / d⟩ = n/2 + 1   for every unit t mod d.

   So dim(Hodge classes in H^n) = 1 + #{a : condition}. For surfaces (n = 2) this is the Picard number ρ, and by the
   Lefschetz (1,1) theorem all of these classes are algebraic.

4. Lines on a Fermat surface: two lines meet iff their four spanning vectors are linearly dependent (det = 0);
   L² = 2 − d by adjunction. The rank of the Gram matrix is the rank of the lattice the lines span in NS(X).
"""

import math

import torch

from . import tensors


def jacobian_hodge(n: int, d: int) -> torch.Tensor:
    """h^{n−q,q}_prim for q = 0 … n: coefficients of (1 + t + … + t^{d−2})^{n+2}.   → (n + 1,) int64"""
    if d < 2:
        return torch.zeros(n + 1, dtype=torch.int64)
    digit = torch.ones(d - 1, dtype=torch.float64)  # 1 + t + … + t^{d−2}
    poly = torch.ones(1, dtype=torch.float64)
    for _ in range(n + 2):  # multiply polynomials = full convolution of coefficient vectors
        poly = torch.nn.functional.conv1d(poly[None, None], digit.flip(0)[None, None], padding=d - 2)[0, 0]
    degree = (torch.arange(n + 1) + 1) * d - (n + 2)
    inside = (degree >= 0) & (degree < len(poly))
    return torch.where(inside, poly[degree.clamp(0, len(poly) - 1)], 0.0).round().long()


def fermat_diamond(n: int, d: int) -> torch.Tensor:
    """Full Hodge diamond h[p, q] of the degree-d hypersurface in ℙ^{n+1} (n ≥ 1).   → (n + 1, n + 1) int64"""
    h = torch.eye(n + 1, dtype=torch.int64)  # h^{p,p} = 1: powers of the hyperplane class
    prim = jacobian_hodge(n, d)  # (n + 1,) indexed by q
    q = torch.arange(n + 1)
    h[n - q, q] += prim
    return h


def torus_diamond(g: int) -> torch.Tensor:
    """h^{p,q} = C(g,p) C(g,q) for a complex torus ℂ^g/Λ: an outer product.   → (g + 1, g + 1)"""
    binom = torch.tensor([math.comb(g, p) for p in range(g + 1)], dtype=torch.int64)
    return torch.einsum("p,q->pq", binom, binom)


def kunneth(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """h(X × Y)[p, q] = Σ h1[a, b] h2[p − a, q − b]: a 2-D convolution.   (m+1, m+1), (k+1, k+1) → (m+k+1, m+k+1)"""
    m, k = h1.shape[0], h2.shape[0]
    outer = torch.einsum("ab,cd->abcd", h1, h2)  # every pair of classes
    p = (torch.arange(m)[:, None, None, None] + torch.arange(k)[None, None, :, None]).expand(m, m, k, k)
    q = (torch.arange(m)[None, :, None, None] + torch.arange(k)[None, None, None, :]).expand(m, m, k, k)
    out = torch.zeros(m + k - 1, m + k - 1, dtype=h1.dtype)
    out.index_put_((p.reshape(-1), q.reshape(-1)), outer.reshape(-1), accumulate=True)
    return out


def euler_characteristic(h: torch.Tensor) -> int:
    """χ = Σ_{p,q} (−1)^{p+q} h^{p,q}."""
    p = torch.arange(h.shape[0])
    sign = (-1) ** (p[:, None] + p[None, :])
    return int((sign * h).sum())


def euler_formula(n: int, d: int) -> int:
    """χ of a smooth degree-d hypersurface in ℙ^{n+1}: ((1 − d)^{n+2} − 1)/d + n + 2 (from c(T_X))."""
    return ((1 - d) ** (n + 2) - 1) // d + n + 2


def hodge_type(chars: torch.Tensor, d: int) -> torch.Tensor:
    """q(a) with q + 1 = Σ_i a_i / d (a_i in 1 … d−1), so V(a) ⊂ H^{n−q,q}.   (A, n+2) → (A,)"""
    return chars.sum(dim=1) // d - 1


def hodge_class_mask(chars: torch.Tensor, d: int) -> torch.Tensor:
    """Shioda's condition Σ_i ⟨t a_i/d⟩ = n/2 + 1 for every unit t, on all characters at once.   → (A,) bool"""
    t = tensors.units(d)  # (T,)
    n_half_plus_one = chars.shape[1] // 2  # (n + 2)/2
    conj = (t[:, None, None] * chars[None, :, :]) % d  # (T, A, n+2): the Galois conjugates t·a
    return (conj.sum(dim=2) == d * n_half_plus_one).all(dim=0)


def hodge_class_count(n: int, d: int) -> int:
    """dim of the rational Hodge classes in H^n of the Fermat variety (n even): 1 + #{a satisfying Shioda}."""
    if n % 2:
        return 0
    return 1 + int(hodge_class_mask(tensors.characters(n, d), d).sum())


def line_gram(d: int) -> torch.Tensor:
    """Intersection matrix of the 3d² lines on the Fermat surface of degree d.   → (L, L) int64"""
    lines = tensors.fermat_lines(d)  # (L, 2, 4)
    n_lines = len(lines)
    pair = torch.cat(
        [lines[:, None].expand(-1, n_lines, -1, -1), lines[None, :].expand(n_lines, -1, -1, -1)], dim=2
    )  # (L, L, 4, 4): the four spanning vectors of lines i and j
    meet = torch.linalg.det(pair).abs() < 1e-9  # dependent ⇔ the lines meet (or coincide)
    gram = meet.long()
    gram.fill_diagonal_(2 - d)  # self-intersection from adjunction
    return gram
