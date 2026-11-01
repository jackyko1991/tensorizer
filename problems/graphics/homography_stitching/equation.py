r"""Projective geometry → tensor translation.

Two photos of a plane are related by a homography, a 3 × 3 matrix acting on homogeneous pixel coordinates:

    x'_i ∝ H_{ij} x_j        (j summed; x = (u, v, 1), equality up to the scale x'_3).

Each match x ↔ x' gives two linear equations in the nine entries h_a of H, from x' × (H x) = 0:

    A_{n0a} h_a = 0,   A_{n0} = (−u, −v, −1, 0, 0, 0, u'u, u'v, u')
    A_{n1a} h_a = 0,   A_{n1} = (0, 0, 0, −u, −v, −1, v'u, v'v, v')

The direct linear transform (DLT) takes h as the unit vector minimising |A h|: the right singular vector of A with
the smallest singular value. Hartley's normalisation (shift each point set to its centroid and scale it to mean
distance √2) makes the columns of A comparable and the SVD well conditioned; H = T'^{-1} Ĥ T afterwards.

Everything here takes a leading batch index b, so thousands of 4-point RANSAC hypotheses are solved by one batched
SVD of shape (B, 8, 9).
"""

import torch


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """(..., 2) → (..., 3) by appending 1."""
    return torch.cat([x, torch.ones_like(x[..., :1])], -1)


def apply_homography(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'_{bni} = H_{bij} x_{bnj} / (H_{b3j} x_{bnj})   (j summed).   H (B, 3, 3), x (B, N, 2) or (N, 2) → (B, N, 2)"""
    xh = to_homogeneous(x)
    eq = "bij,bnj->bni" if xh.dim() == 3 else "bij,nj->bni"
    y = torch.einsum(eq, H, xh)
    w = y[..., 2:]
    return y[..., :2] / torch.where(w.abs() < 1e-12, torch.full_like(w, 1e-12), w)


def normalisation(x: torch.Tensor) -> torch.Tensor:
    """Similarity T (B, 3, 3) moving each point set x (B, N, 2) to centroid 0 and mean distance √2."""
    centre = x.mean(dim=1)  # (B, 2)
    scale = 2**0.5 / (x - centre[:, None]).norm(dim=-1).mean(dim=1).clamp(min=1e-12)  # (B,)
    T = torch.zeros(x.shape[0], 3, 3, dtype=x.dtype, device=x.device)
    T[:, 0, 0] = T[:, 1, 1] = scale
    T[:, :2, 2] = -scale[:, None] * centre
    T[:, 2, 2] = 1
    return T


def dlt_matrix(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """A_{b,(n r),a}: two rows per match. x1, x2 (B, N, 2) → (B, 2N, 9)."""
    u, v = x1[..., 0], x1[..., 1]
    u2, v2 = x2[..., 0], x2[..., 1]
    one, zero = torch.ones_like(u), torch.zeros_like(u)
    row_u = torch.stack([-u, -v, -one, zero, zero, zero, u2 * u, u2 * v, u2], -1)  # (B, N, 9)
    row_v = torch.stack([zero, zero, zero, -u, -v, -one, v2 * u, v2 * v, v2], -1)
    return torch.stack([row_u, row_v], dim=2).reshape(x1.shape[0], -1, 9)


def dlt(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    r"""Normalised DLT for a batch of point sets: h = argmin_{|h|=1} |A h| by SVD.   (B, N, 2) × 2 → H (B, 3, 3)

    H is scaled so that H_33 = 1 (a homography is only defined up to scale)."""
    T1, T2 = normalisation(x1), normalisation(x2)
    n1 = apply_homography(T1, x1)  # (B, N, 2) normalised points
    n2 = apply_homography(T2, x2)
    A = dlt_matrix(n1, n2)
    _, _, Vh = torch.linalg.svd(A)  # batched SVD; full_matrices gives Vh (B, 9, 9) even for 4 points (8 rows)
    H_hat = Vh[:, -1].reshape(-1, 3, 3)  # right singular vector of the smallest singular value
    H = torch.linalg.solve(T2, H_hat @ T1)  # undo the normalisation: H = T2⁻¹ Ĥ T1
    h33 = H[:, 2:, 2:]
    return H / torch.where(h33.abs() < 1e-12, torch.full_like(h33, 1e-12), h33)


def transfer_error(H: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    r"""e_{bn} = |x2_n − H_b x1_n| in pixels for every hypothesis b and match n.   H (B, 3, 3), x (N, 2) → (B, N)"""
    err = (apply_homography(H, x1) - x2).norm(dim=-1)
    return torch.nan_to_num(err, nan=1e9, posinf=1e9)
