"""Build every partial sum S_1 … S_Nmax and measure the Gibbs overshoot and the L² error of each."""

import torch

from problems._common.progress import steps

from .equation import l2_error_sq, partial_sum


def all_partial_sums(b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """S[N-1] = partial sum with N harmonics, for N = 1 … N_max: (N_max, W, X)."""
    return torch.stack([partial_sum(b, c, n + 1) for n in steps(c.shape[1], "partial sums")])


def overshoot(s: torch.Tensor, f: torch.Tensor, jump: list[float]) -> torch.Tensor:
    """At the highest point of S_N: how far it rises above the wave there, as a fraction of the jump.

    s (N_max, W, X), f (W, X) → (N_max, W). Raw S − f for a wave without a jump.
    """
    peak = s.argmax(dim=-1)  # (N_max, W): grid index of the top of each partial sum
    above = s.amax(dim=-1) - f[torch.arange(f.shape[0]), peak]
    scale = torch.tensor([j if j > 0 else 1.0 for j in jump], dtype=s.dtype, device=s.device)
    return above / scale


def errors(f: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Squared L² error of every partial sum: (N_max, W)."""
    return torch.stack([l2_error_sq(f, s_n, w) for s_n in s])


def decay_rate(err: torch.Tensor) -> torch.Tensor:
    """Fitted p in E_N ∝ N^(−p), using odd N in the upper half of the range (even harmonics of the square are 0).

    err (N_max, W) → (W,)
    """
    n = torch.arange(1, err.shape[0] + 1, dtype=err.dtype, device=err.device)
    keep = (n % 2 == 1) & (n > err.shape[0] // 2)
    x = torch.log(n[keep])  # (K,)
    y = torch.log(err[keep])  # (K, W)
    x = x - x.mean()
    slope = torch.einsum("k,kw->w", x, y - y.mean(dim=0)) / (x @ x)
    return -slope
