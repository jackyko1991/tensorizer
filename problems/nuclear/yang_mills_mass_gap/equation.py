r"""Physics → tensor translation: Wilson's lattice gauge theory with unit-quaternion links.

    S[U] = β Σ_x Σ_{μ<ν} (1 − ½ Tr P_μν(x)),      P_μν(x) = U_μ(x) U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)†

Quaternion product (the group law): (ab)₀ = a₀b₀ − aₖbₖ, (ab)ₖ = a₀bₖ + b₀aₖ + ε_kij a_i b_j (k, i, j summed);
½ Tr(ab) = (ab)₀. Shifts x → x + μ̂ are torch.roll on the periodic lattice, so every plaquette, staple and Wilson
loop of the whole batch is a handful of rolls and products. Under a gauge transformation
U_μ(x) → g(x) U_μ(x) g(x+μ̂)† every closed loop of links keeps its trace: the action is gauge invariant.

Strong coupling (character expansion): each plaquette contributes u(β) = I₁(β)/I₀(β) for U(1) and I₂(β)/I₁(β) for
SU(2). In 2D this is exact: ⟨½ Tr P⟩ = u and W(R, T) = u^{RT} (area law, string tension σ = −ln u).
"""

import torch


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion (group) product of two tensors (..., 4) → (..., 4)."""
    a0, av, b0, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    scalar = a0 * b0 - (av * bv).sum(dim=-1, keepdim=True)  # a₀b₀ − a·b
    cross = torch.linalg.cross(av, bv, dim=-1)  # (a × b)_k = ε_kij a_i b_j
    return torch.cat([scalar, a0 * bv + b0 * av + cross], dim=-1)


def qconj(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion (U†): flip the vector part."""
    return torch.cat([a[..., :1], -a[..., 1:]], dim=-1)


def half_trace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """½ Tr(ab) = a₀b₀ − a·b without forming the product.   (..., 4) → (...)"""
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(dim=-1)


def shift(field: torch.Tensor, mu: int, n: int = 1) -> torch.Tensor:
    """field(x + n μ̂) on the periodic lattice. field (B, L, …, L, 4): site axes are 1 … D."""
    return torch.roll(field, shifts=-n, dims=1 + mu)


def plaquettes(links: torch.Tensor) -> torch.Tensor:
    """½ Tr P_μν(x) for every plane μ < ν and site.   links (B, D, L…, 4) → (B, D(D−1)/2, L…)"""
    dim = links.shape[1]
    out = []
    for mu in range(dim):
        for nu in range(mu + 1, dim):
            u_mu, u_nu = links[:, mu], links[:, nu]
            left = qmul(u_mu, shift(u_nu, mu))  # U_μ(x) U_ν(x+μ̂)
            right = qmul(u_nu, shift(u_mu, nu))  # U_ν(x) U_μ(x+ν̂)
            out.append(half_trace(left, qconj(right)))  # ½ Tr[U_μ U_ν(x+μ̂) (U_ν U_μ(x+ν̂))†]
    return torch.stack(out, dim=1)


def staple(links: torch.Tensor, mu: int) -> torch.Tensor:
    """A_μ(x) = Σ_{ν≠μ} [U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)† + U_ν(x+μ̂−ν̂)† U_μ(x−ν̂)† U_ν(x−ν̂)], so the part of S that
    contains U_μ(x) is −β ½ Tr(U_μ(x) A_μ(x)).   (B, D, L…, 4) → (B, L…, 4)"""
    u_mu = links[:, mu]
    total = torch.zeros_like(u_mu)
    for nu in range(links.shape[1]):
        if nu == mu:
            continue
        u_nu = links[:, nu]
        up = qmul(qmul(shift(u_nu, mu), qconj(shift(u_mu, nu))), qconj(u_nu))
        down_nu = shift(u_nu, nu, -1)  # U_ν(x−ν̂)
        down = qmul(qmul(qconj(shift(down_nu, mu)), qconj(shift(u_mu, nu, -1))), down_nu)
        total = total + up + down
    return total


def gauge_transform(links: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """U_μ(x) → g(x) U_μ(x) g(x+μ̂)† for every μ.   links (B, D, L…, 4), g (B, L…, 4)"""
    return torch.stack([qmul(qmul(g, links[:, mu]), qconj(shift(g, mu))) for mu in range(links.shape[1])], dim=1)


def line(links: torch.Tensor, mu: int, length: int) -> torch.Tensor:
    """Ordered product U_μ(x) U_μ(x+μ̂) ⋯ U_μ(x+(n−1)μ̂) starting at every site.   → (B, L…, 4)"""
    out = links[:, mu]
    for step in range(1, length):
        out = qmul(out, shift(links[:, mu], mu, step))
    return out


def wilson_loops(links: torch.Tensor, r_max: int) -> torch.Tensor:
    """W(R, T) = ⟨½ Tr⟩ of the R × T rectangle, averaged over sites, planes and both orientations.   → (B, R, T)"""
    dim = links.shape[1]
    lines = {(mu, n): line(links, mu, n) for mu in range(dim) for n in range(1, r_max + 1)}
    w = torch.zeros(links.shape[0], r_max, r_max, dtype=links.dtype, device=links.device)
    for mu in range(dim):
        for nu in range(dim):
            if mu == nu:
                continue
            for r in range(1, r_max + 1):
                for t in range(1, r_max + 1):
                    lower = qmul(lines[mu, r], shift(lines[nu, t], mu, r))  # along μ, then up ν
                    upper = qmul(lines[nu, t], shift(lines[mu, r], nu, t))  # up ν, then along μ
                    w[:, r - 1, t - 1] += half_trace(lower, qconj(upper)).flatten(1).mean(dim=1)
    return w / (dim * (dim - 1))


def strong_coupling_u(beta: torch.Tensor, group: str) -> torch.Tensor:
    """Character coefficient u(β): I₁/I₀ for U(1), I₂/I₁ = I₀/I₁ − 2/β for SU(2) (scaled Bessels, no overflow)."""
    i0, i1 = torch.special.i0e(beta), torch.special.i1e(beta)
    return i1 / i0 if group == "U(1)" else i0 / i1 - 2 / beta


def weak_coupling(beta: torch.Tensor, group: str, dim: int) -> torch.Tensor:
    """Equipartition at large β: each of the (N²−1)(D−1) physical link modes per site carries ½, so
    ⟨½ Tr P⟩ ≈ 1 − n_gen/(D β) with n_gen = 3 (SU(2)) or 1 (U(1))."""
    n_gen = 3 if group == "SU(2)" else 1
    return 1 - n_gen / (dim * beta)
