r"""Physics → tensor translation: minimum-image pair vectors, Lennard-Jones forces, virial pressure and g(r).

Indices: s state point, i j atoms, a Cartesian axis, k histogram bin.

    d_saij = x_sia - x_sja - L_s round((x_sia - x_sja) / L_s)              minimum image
    r²_sij = d_saij d_saij                                                 (a summed)
    u(r) = 4 (r⁻¹² - r⁻⁶) - u(r_c)  for r < r_c                            truncated and shifted
    F_sia = Σ_j 24 (2 r⁻¹⁴ - r⁻⁸)_sij d_saij                                (j summed)

The pair tensor is stored axis-first, (S, 3, N, N): the innermost loop then runs over j in contiguous memory,
which on a CPU is several times faster than (S, N, N, 3). For the same reason the two sums that run every step
are written as (d * d).sum(1) and (f * d).sum(-1); for these shapes einsum picks a slower path.
    P_s = ρ_s T_s + (1/3V_s) ½ Σ_ij r_sij f(r_sij) + P_tail                virial theorem
    g_s(r_k) = ⟨pairs in shell k⟩ / (½ N ρ_s 4π r_k² Δr)                   radial distribution function
"""

import math

import torch

R_CUT = 2.5  # σ


def minimum_image(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Pair vectors d_saij = x_sia - x_sja wrapped to the nearest image.   x (S, N, 3), L (S,) → (S, 3, N, N)"""
    xt = x.transpose(1, 2).contiguous()  # (S, 3, N), contiguous so d is laid out (S, 3, N, N) in memory too
    d = xt[:, :, :, None] - xt[:, :, None, :]
    box = L[:, None, None, None]
    return d - box * torch.round(d / box)


def lj_pairs(d: torch.Tensor):
    """r²_sij, pair energies u_sij and f_sij/r_sij for r < r_c (zero on the diagonal and beyond the cutoff)."""
    r2 = (d * d).sum(1)  # (S, N, N), a summed
    inside = (r2 < R_CUT**2) & (r2 > 0)  # r² = 0 only on the diagonal i = j
    inv2 = torch.where(inside, r2.reciprocal(), 0.0)  # 1/r², 0 where no interaction
    inv6 = inv2 * inv2 * inv2
    inv12 = inv6 * inv6
    u_cut = 4 * (R_CUT**-12 - R_CUT**-6)
    u = torch.where(inside, 4 * (inv12 - inv6) - u_cut, 0.0)
    f_over_r = 24 * inv2 * (2 * inv12 - inv6)  # -du/dr / r
    return r2, u, f_over_r


def forces(d: torch.Tensor, f_over_r: torch.Tensor) -> torch.Tensor:
    """F_sia = Σ_j (f/r)_sij d_saij   (j summed): Newton's third law is built in, since d_saji = -d_saij.   (S, N, 3)"""
    return (f_over_r[:, None] * d).sum(-1).transpose(1, 2)


def potential_energy(u: torch.Tensor) -> torch.Tensor:
    """U_s = ½ Σ_ij u_sij (every pair appears twice in the full matrix).   (S,)"""
    return 0.5 * u.sum((1, 2))


def kinetic_temperature(v: torch.Tensor) -> torch.Tensor:
    """T_s = Σ_ia v²_sia / (3N - 3), momentum conserved.   (S,)"""
    return torch.einsum("sia,sia->s", v, v) / (3 * v.shape[1] - 3)


def virial_pressure(r2: torch.Tensor, f_over_r: torch.Tensor, T: torch.Tensor, rho: torch.Tensor, L: torch.Tensor):
    """P_s = ρT + W/3V + P_tail, W = ½ Σ_ij r·F = ½ Σ_ij (f/r) r²; the tail adds the pairs beyond r_c (g ≈ 1 there)."""
    W = 0.5 * torch.einsum("sij,sij->s", f_over_r, r2)
    tail = 16 / 3 * math.pi * rho**2 * (2 / 3 * R_CUT**-9 - R_CUT**-3)
    return rho * T + W / (3 * L**3) + tail


def pair_histogram(r2: torch.Tensor, r_max: float, n_bins: int) -> torch.Tensor:
    """Count every pair i < j into n_bins shells of [0, r_max) for each state point.   (S, K)"""
    S, N, _ = r2.shape
    i, j = torch.triu_indices(N, N, offset=1, device=r2.device)
    r = torch.sqrt(r2[:, i, j])  # (S, P) each pair once
    k = (r / r_max * n_bins).long()
    keep = k < n_bins
    flat = (torch.arange(S, device=r2.device)[:, None] * n_bins + k)[keep]
    return (
        torch.zeros(S * n_bins, device=r2.device)
        .index_add_(0, flat, torch.ones_like(flat, dtype=r.dtype))
        .reshape(S, n_bins)
    )


def radial_distribution(counts: torch.Tensor, n_samples: int, N: int, rho: torch.Tensor, r_max: float):
    """g_s(r_k) = counts / (samples · ½ N ρ_s · shell volume_k): 1 for an ideal gas.   (S, K), centres (K,)"""
    K = counts.shape[1]
    edges = torch.linspace(0, r_max, K + 1, device=counts.device)
    shell = 4 / 3 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)  # (K,)
    ideal = n_samples * 0.5 * N * rho[:, None] * shell[None, :]
    return counts / ideal, 0.5 * (edges[1:] + edges[:-1])


def second_virial(T: float, n: int = 20000) -> float:
    """B₂(T) = -2π ∫ (e^{-u(r)/T} - 1) r² dr for the full LJ potential, by the trapezoid rule."""
    r = torch.linspace(0.3, 30.0, n, dtype=torch.float64)
    u = 4 * (r**-12 - r**-6)
    core = 2 * math.pi * 0.3**3 / 3  # r < 0.3: e^{-u/T} = 0, the integrand is -r²
    return (-2 * math.pi * torch.trapezoid((torch.exp(-u / T) - 1) * r**2, r)).item() + core
