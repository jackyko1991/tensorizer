r"""Physics → tensor translation.

Fluorescent molecules diffuse in the 2D membrane:  ∂c/∂t = D ∇²c.  On a grid with spacing h the Laplacian
is the 5-point stencil, a weighted sum over the neighbours k of every grid point (i, j):

    (∇²c)_{bij} = (1/h²) w_k c_{b, i+p_k, j+q_k}          k summed,  w = (−4, 1, 1, 1, 1)

    \dot c_{bij} = D_b (∇²c)_{bij}                         one membrane per species b

The fluorescence in the bleached spot is a sum over the pixels of the mask M:

    F_b(t) = M_{ij} c_{bij} / M_{ij}M_{ij}                 i, j summed
"""

import torch
import torch.nn.functional as F


def neighbours(c: torch.Tensor) -> torch.Tensor:
    """Shifted copies c_{b, i+p_k, j+q_k} for the 5 stencil points.   (B, N, N) → (5, B, N, N).

    The edge is copied outward ("replicate" padding): no flux leaves the patch, so the total dye is conserved.
    """
    n = c.shape[-1]
    p = F.pad(c[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]  # (B, N+2, N+2)
    centre, down, up = p[:, 1 : n + 1, 1 : n + 1], p[:, 2:, 1 : n + 1], p[:, :n, 1 : n + 1]
    right, left = p[:, 1 : n + 1, 2:], p[:, 1 : n + 1, :n]
    return torch.stack([centre, down, up, right, left])  # same order as tensors.OFFSETS


def laplacian(c: torch.Tensor, weights: torch.Tensor, h: float) -> torch.Tensor:
    """(∇²c)_{bij} = w_k c_{b,i+p_k,j+q_k} / h²,  k summed.   (B, N, N)"""
    return torch.einsum("k,kbij->bij", weights, neighbours(c)) / h**2


def rhs(c: torch.Tensor, D: torch.Tensor, weights: torch.Tensor, h: float) -> torch.Tensor:
    """∂c_{bij}/∂t = D_b (∇²c)_{bij}: every species b diffuses with its own D (no index summed)."""
    return torch.einsum("b,bij->bij", D, laplacian(c, weights, h))


def spot_intensity(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """F_b = M_ij c_bij / Σ M_ij: mean fluorescence inside the bleached disk.   (B, N, N) → (B,)"""
    return torch.einsum("bij,ij->b", c, mask) / mask.sum()


def normalized_recovery(F_t: torch.Tensor, F_0: torch.Tensor, F_inf: torch.Tensor) -> torch.Tensor:
    """f = (F(t) − F(0)) / (F(∞) − F(0)): 0 just after the bleach, 1 when fully recovered.   (T, B)"""
    return (F_t - F_0) / (F_inf - F_0)


def diffusion_time(w: float, D: torch.Tensor) -> torch.Tensor:
    """τ_D = w² / (4 D): the time to diffuse across the bleach radius.   (B,)"""
    return w**2 / (4.0 * D)


def soumpasis(t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Soumpasis (1983) recovery of a uniformly bleached disk on an infinite membrane:

    f(t) = e^{−2τ/t} [I₀(2τ/t) + I₁(2τ/t)].   t: (T,), tau: (B,) → (T, B)

    torch.special.i0e(x) = e^{−x} I₀(x) already contains the exponential factor.
    """
    x = 2.0 * tau[None, :] / t.clamp_min(1e-9)[:, None]
    return torch.special.i0e(x) + torch.special.i1e(x)


def soumpasis_half_time(w: float, D: torch.Tensor) -> torch.Tensor:
    """Half-recovery time t½ = 0.224 w² / D ≈ 0.90 τ_D (Soumpasis' rule  D = 0.224 w² / t½).   (B,)"""
    return 0.224 * w**2 / D
