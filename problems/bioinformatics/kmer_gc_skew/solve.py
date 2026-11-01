"""Analyse the genome: k-mer spectrum, GC content and skew in sliding windows, origin from the cumulative skew."""

import torch

from . import equation


def count_kmers(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """k-mer spectrum (4^k,) of a one-hot sequence X (L, 4); the counts add up to L - k + 1."""
    k = P.shape[0]
    U = equation.unfold(X, k)  # (L - k + 1, 4, k)
    return equation.kmer_counts(equation.kmer_index(U, P), k)


def windows(X: torch.Tensor, size: int, is_gc: torch.Tensor, g_minus_c: torch.Tensor) -> dict[str, torch.Tensor]:
    """GC content and skew of windows of `size` bases that start every size/2 bases around the circle."""
    step = max(1, size // 2)
    n = equation.window_base_counts(equation.circular(X, size, step), size, step)  # (W, 4)
    skew = equation.gc_skew(n, is_gc, g_minus_c)  # (W,)
    return {
        "start": torch.arange(n.shape[0], device=X.device) * step,  # (W,) first base of each window
        "gc": equation.gc_content(n, is_gc),  # (W,)
        "skew": skew,
        "cum_skew": torch.cumsum(skew, dim=0),  # (W,) the classic windowed cumulative skew plot
    }


def find_origin(X: torch.Tensor, g_minus_c: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Per-base cumulative skew S (L,), and the predicted origin (argmin S) and terminus (argmax S)."""
    S = equation.cumulative_skew(X, g_minus_c)
    return S, int(S.argmin()), int(S.argmax())


def circular_distance(a: int, b: int, length: int) -> int:
    """Distance between two positions on a circle of `length` bases."""
    d = abs(a - b) % length
    return min(d, length - d)
