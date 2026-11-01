r"""Sequence → tensor translation for a position weight matrix (PWM).

Aligned training sites become a one-hot tensor X_{sja} (site s, column j, letter a). Then

    C_{ja} = X_{sja}                               (s summed: letter counts per column)
    P_{ja} = (C_{ja} + β) / (N + 4β)               (frequencies with a pseudocount β, rows sum to 1)
    W_{ja} = log2(P_{ja} / q_a)                    (log-odds against the background q, in bits)
    IC_j   = P_{ja} log2(P_{ja} / q_a)              (a summed: information content, 0..2 bits for uniform q)

Scanning a sequence X_{na}: unfold it into every window of m letters, U_{naj} = X_{n+j, a}, and score both strands
in one einsum with the stacked matrices W_{sja} (s = 0: the PWM, s = 1: its reverse complement),

    score_{sn} = U_{naj} W_{sja}                    (j and a summed)

The reverse complement reads the other strand: column order reversed and A↔T, C↔G swapped. With the letters ordered
A, C, G, T, the complement of letter a is 3 - a, so it is just W flipped along both axes.
"""

import torch


def counts(X: torch.Tensor) -> torch.Tensor:
    """C_{ja} = X_{sja}  (s summed).   (N, m, 4) → (m, 4)"""
    return torch.einsum("sja->ja", X)


def frequencies(C: torch.Tensor, pseudo: float) -> torch.Tensor:
    """P_{ja} = (C_{ja} + β) / (N + 4β): never exactly 0, so the log-odds stay finite.   (m, 4)"""
    N = C.sum(dim=1, keepdim=True)  # (m, 1) number of sites
    return (C + pseudo) / (N + 4 * pseudo)


def log_odds(P: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """W_{ja} = log2(P_{ja} / q_a): positive for letters the site prefers, negative for avoided ones.   (m, 4)"""
    return torch.log2(P / q)


def information_content(P: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """IC_j = P_{ja} log2(P_{ja} / q_a)  (a summed): bits per column; with q uniform this is 2 - entropy.   (m,)"""
    return torch.einsum("ja,ja->j", P, torch.log2(P / q))


def reverse_complement_matrix(W: torch.Tensor) -> torch.Tensor:
    """W^rc_{ja} = W_{m-1-j, 3-a}: the same site read on the other strand.   (m, 4)"""
    return W.flip(0).flip(1)


def windows(X: torch.Tensor, m: int) -> torch.Tensor:
    """U_{…naj} = X_{…, n+j, a}: every window of m letters.   (…, L, 4) → (…, L-m+1, 4, m), no copy."""
    return X.unfold(-2, m, 1)


def scan(X: torch.Tensor, W_both: torch.Tensor) -> torch.Tensor:
    r"""score_{…sn} = U_{…naj} W_{sja}  (j and a summed).   X (…, L, 4), W_both (2, m, 4) → (…, 2, L-m+1)."""
    U = windows(X, W_both.shape[1])
    return torch.einsum("...naj,sja->...sn", U, W_both)


def p_values(scores: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    """p = (1 + #{null ≥ score}) / (1 + #null): how often a shuffled sequence scores at least as high.  Same shape."""
    null_sorted = null.flatten().sort().values
    n_below = torch.searchsorted(null_sorted, scores.contiguous(), right=False)  # null values < score
    return (1 + null_sorted.numel() - n_below) / (1 + null_sorted.numel())
