r"""Singular value decomposition → tensor translation.

An image is a matrix A_{ij} (row i, column j). Its SVD writes it as a sum of rank-1 layers,

    A_{ij} = \sum_r U_{ir}\,\sigma_r\,V_{jr},       U, V orthonormal columns, \sigma_1 \ge \sigma_2 \ge \dots \ge 0,

and keeping only the first k layers gives the best rank-k approximation (Eckart–Young):

    (A_k)_{ij} = \sum_{r \le k} U_{ir}\,\sigma_r\,V_{jr},     \|A - A_k\|_F^2 = \sum_{r > k} \sigma_r^2.

A colour image A_{cij} is three matrices, one per channel c: torch.linalg.svd batches over the leading index.
"""

import torch


def svd(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""A = U Σ Vᵀ for one matrix (H, W) or a batch (C, H, W).   Returns U (..., H, R), σ (..., R), V (..., W, R)."""
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    return u, s, vh.mT


def reconstruct(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, k: int) -> torch.Tensor:
    r"""(A_k)_{...ij} = U_{...ir} σ_{...r} V_{...jr}, summed over r ≤ k.   → (..., H, W)"""
    return torch.einsum("...ir,...r,...jr->...ij", u[..., :k], s[..., :k], v[..., :k])


def layer(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, r: int) -> torch.Tensor:
    r"""One rank-1 layer σ_r U_{ir} V_{jr} (an outer product, nothing summed).   (H, R), (R,), (W, R) → (H, W)"""
    return s[r] * torch.einsum("i,j->ij", u[:, r], v[:, r])


def tail_energy(s: torch.Tensor) -> torch.Tensor:
    r"""Eckart–Young error E_k = Σ_{r>k} σ_r² for k = 0..R: a reversed cumulative sum.   (..., R) → (..., R+1)"""
    tail = torch.flip(torch.cumsum(torch.flip(s**2, [-1]), -1), [-1])  # tail[k] = Σ_{r ≥ k} σ_r²  (0-based)
    return torch.cat([tail, tail.new_zeros(*tail.shape[:-1], 1)], dim=-1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Peak signal-to-noise ratio 10 log₁₀(1 / MSE) in dB, for images in [0, 1]."""
    return 10 * torch.log10(1.0 / ((a - b) ** 2).mean())


def compression_ratio(h: int, w: int, k: int) -> float:
    """Numbers stored for the full image divided by those for k layers (k columns of U and V plus k σ's)."""
    return h * w / (k * (h + w + 1))
