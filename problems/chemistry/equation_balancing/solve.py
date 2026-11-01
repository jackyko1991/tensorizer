"""Find the coefficients: SVD for the null space, then scale the vector to the smallest whole numbers."""

import torch


def null_vectors(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched SVD M = U diag(sv) Vᵀ.   M: (R, E, S) with E ≥ S → sv (R, S) descending, V (R, S, S).

    The columns of V belong to the singular values; the last ones, with sv ≈ 0, span the null space.
    """
    _, sv, Vh = torch.linalg.svd(M, full_matrices=False)
    return sv, Vh.transpose(-1, -2)


def to_integers(v: torch.Tensor, max_scale: int = 60, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale null vectors to the smallest whole numbers.   v: (..., S) → (ν (..., S) integers, ok (...,) bool).

    1) flip the sign so the largest entry is positive; 2) divide by the smallest non-zero |entry| so it becomes 1;
    3) try multipliers k = 1, 2, … and keep the first one that turns every entry into a whole number.
    The smallest entry is then k and the result has no common factor left.
    """
    big = v.gather(-1, v.abs().argmax(-1, keepdim=True))  # (..., 1) the largest entry, with its sign
    v = v / big
    smallest = torch.where(v.abs() > 1e-9, v.abs(), torch.inf).amin(-1, keepdim=True)
    v = v / smallest  # smallest non-zero entry → ±1
    k = torch.arange(1, max_scale + 1, dtype=v.dtype, device=v.device)  # (K,)
    scaled = v[..., None, :] * k[:, None]  # (..., K, S)
    error = (scaled - scaled.round()).abs().amax(-1)  # (..., K) distance from whole numbers
    good = error < tol
    first = good.int().argmax(-1)  # index of the first multiplier that works (0 if none does)
    nu = scaled.gather(-2, first[..., None, None].expand(*first.shape, 1, v.shape[-1])).squeeze(-2).round()
    return nu, good.any(-1)


def reactions_in_plane(N: torch.Tensor) -> torch.Tensor:
    """For a 2D null space with basis N (S, 2): the S special combinations with one coefficient set to zero.

    W_{ts} = N_{t0} N_{s1} - N_{t1} N_{s0}  → column s is a null vector whose entry t = s vanishes.
    Returns (S, S): row s is that vector (whole numbers, sign fixed so the largest entry is positive).
    """
    W = torch.einsum("t,s->ts", N[:, 0], N[:, 1]) - torch.einsum("t,s->ts", N[:, 1], N[:, 0])
    nu, _ = to_integers(W.T)
    return nu


def physical(nu: torch.Tensor) -> torch.Tensor:
    """A reaction must have every coefficient ≥ 0 (reactants left, products right).   (..., S) → (...,) bool."""
    return (nu >= 0).all(-1)
