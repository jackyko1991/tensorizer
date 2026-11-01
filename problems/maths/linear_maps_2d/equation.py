r"""Maths → tensor translation.

A 2×2 matrix A moves every point of the plane, y_i = A_{ij} x_j (j summed). Stack many points in P_{kj} (k = point)
and many matrices in M_{tij} (t = animation frame) and every image of every point is one einsum,

    Y_{tki} = M_{tij}\,P_{kj}            (j summed).

The animation walks from the identity to A along M(t) = (1 − t) I + t A. Because M(t) is a mix of I and A, an
eigenvector of A (A v = λ v) is also an eigenvector of every M(t): it only slides along its own line.

The determinant, the signed area of the image of the unit square, is

    \det M = \varepsilon_{ij}\,M_{i1}\,M_{j2}            (i, j summed),

with ε the 2-D Levi-Civita symbol. |det| scales every area and a negative det flips orientation (a mirror image).
"""

import torch


def apply(m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""Y_{t…i} = M_{tij} P_{…j}   (j summed).   M (T, 2, 2), P (…, 2) → (T, …, 2)"""
    return torch.einsum("tij,...j->t...i", m, p)


def interpolate(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""M_{tij} = (1 − t) δ_{ij} + t A_{ij}: the straight path from the identity to A.   t (T,) → (T, 2, 2)"""
    eye = torch.eye(2, dtype=a.dtype, device=a.device)
    return (1 - t)[:, None, None] * eye + t[:, None, None] * a


def det(m: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""det M = ε_{ij} M_{i1} M_{j2}   (i, j summed): the signed area spanned by the two columns.   (…, 2, 2) → (…)"""
    return torch.einsum("ij,...i,...j->...", eps, m[..., :, 0], m[..., :, 1])
