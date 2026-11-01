r"""Sequence → tensor translation.

A DNA sequence of length L becomes a one-hot matrix X_{na} (position n, letter a ∈ {A, C, G, T}).

k-mers. Unfolding gives every window of k letters, U_{naj} = X_{n+j, a}. Reading each window as a number in base 4,

    m_n = U_{naj} P_{ja}          (j and a summed),     P_{ja} = a · 4^(k-1-j)

and the k-mer spectrum counts how often each number appears, N_m = Σ_n [m_n = m]. This is a 1-D convolution of X
with a fixed (k, 4) kernel, exactly what a convolutional layer computes.

GC content and skew in windows of w bases, starting every `step` bases (index w for the window, i inside it):

    n_{wa} = U^{(w)}_{wai}        (i summed: base counts per window)
    GC_w = (n_{wG} + n_{wC}) / w,      skew_w = (n_{wG} - n_{wC}) / (n_{wG} + n_{wC})

Cumulative skew. d_n = X_{na} c_a with c = (0, -1, +1, 0) is +1 for G, -1 for C, and S_n = Σ_{n' ≤ n} d_{n'}.
S falls from the terminus to the origin (lagging strand, C-rich) and rises from the origin to the terminus (leading
strand, G-rich), so its minimum marks the replication origin and its maximum the terminus.
"""

import torch


def unfold(X: torch.Tensor, size: int, step: int = 1) -> torch.Tensor:
    """U_{naj} = X_{n·step + j, a}: all windows of `size` letters.   (L, 4) → (n_windows, 4, size), no copy."""
    return X.unfold(0, size, step)


def kmer_index(U: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    r"""m_n = U_{naj} P_{ja}  (j and a summed): the base-4 number of every k-mer.   (n, 4, k), (k, 4) → (n,) long."""
    return torch.einsum("naj,ja->n", U, P).round().long()


def kmer_counts(m: torch.Tensor, k: int) -> torch.Tensor:
    """N_m = Σ_n [m_n = m]: how often each of the 4^k words occurs.   (n,) → (4^k,)"""
    return torch.bincount(m, minlength=4**k)


def circular(X: torch.Tensor, size: int, step: int) -> torch.Tensor:
    """Append the first size - step bases so windows wrap round the circular chromosome.   (L, 4) → (L+size-step, 4)"""
    return torch.cat([X, X[: size - step]])


def window_base_counts(X: torch.Tensor, size: int, step: int) -> torch.Tensor:
    """n_{wa} = U_{wai}  (i summed): how many A, C, G, T are in each window.   (L, 4) → (n_windows, 4)"""
    return torch.einsum("wai->wa", unfold(X, size, step))


def gc_content(n: torch.Tensor, is_gc: torch.Tensor) -> torch.Tensor:
    """GC_w = n_{wa} g_a / Σ_a n_{wa}  (a summed), with g = (0, 1, 1, 0).   (W, 4) → (W,)"""
    return torch.einsum("wa,a->w", n, is_gc) / n.sum(dim=1)


def gc_skew(n: torch.Tensor, is_gc: torch.Tensor, g_minus_c: torch.Tensor) -> torch.Tensor:
    """skew_w = (G - C) / (G + C) = n_{wa} c_a / n_{wa} g_a  (a summed).   (W, 4) → (W,)"""
    return torch.einsum("wa,a->w", n, g_minus_c) / torch.einsum("wa,a->w", n, is_gc).clamp(min=1)


def cumulative_skew(X: torch.Tensor, g_minus_c: torch.Tensor) -> torch.Tensor:
    """S_n = Σ_{n' ≤ n} X_{n'a} c_a: running count of G minus C along the chromosome.   (L, 4) → (L,)"""
    d = torch.einsum("na,a->n", X, g_minus_c)  # +1 for G, -1 for C, 0 for A and T
    return torch.cumsum(d, dim=0)
