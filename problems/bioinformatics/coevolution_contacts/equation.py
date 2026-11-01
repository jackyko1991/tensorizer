r"""Coevolution → tensor translation.

A multiple sequence alignment (MSA) of N sequences, L columns, q letters is a one-hot tensor x_{nia}
(sequence n has letter a at column i). The Potts model gives each sequence the probability

    P(s) ∝ exp( Σ_i h_i(s_i) + Σ_{i<j} J_ij(s_i, s_j) ).

Statistics of the alignment (n summed = average over sequences):

    f_{ia}   = ⟨x_{nia}⟩_n                     single-site frequencies
    f_{iajb} = ⟨x_{nia} x_{njb}⟩_n             pair frequencies
    C_{iajb} = f_{iajb} − f_{ia} f_{jb}        connected correlation (covariance)

Direct couplings, mean-field DCA:  J ≈ −C⁻¹ (C as an L(q−1) × L(q−1) matrix, last letter dropped as gauge).
Pseudo-likelihood:  log P(s_i | s_rest) with the local field  E_{nia} = h_{ia} + J_{iajb} x_{njb}  (j, b summed).
Contact score: Frobenius norm F_ij = ‖J_ij‖ in the zero-sum gauge, then the average-product correction (APC).
"""

import torch


def local_field(x: torch.Tensor, h: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """E_{nia} = h_{ia} + J_{iajb} x_{njb}:  the log-weight of letter a at column i given the rest.  j, b summed.

    x: (N, L, q), h: (L, q), J: (L, q, L, q) with zero diagonal blocks → (N, L, q).
    """
    return h + torch.einsum("iajb,njb->nia", J, x)


def single_frequencies(x: torch.Tensor) -> torch.Tensor:
    """f_{ia} = (1/N) Σ_n x_{nia}.   (N, L, q) → (L, q);  n summed."""
    return torch.einsum("nia->ia", x) / x.shape[0]


def pair_frequencies(x: torch.Tensor) -> torch.Tensor:
    """f_{iajb} = (1/N) Σ_n x_{nia} x_{njb}.   (N, L, q) → (L, q, L, q);  n summed."""
    return torch.einsum("nia,njb->iajb", x, x) / x.shape[0]


def with_pseudocount(f1: torch.Tensor, f2: torch.Tensor, lam: float):
    """Mix the frequencies with a uniform distribution (weight λ) so that C is invertible.

    f_ia → (1−λ) f_ia + λ/q,   f_iajb → (1−λ) f_iajb + λ/q²  (i ≠ j),   f_iaib → f_ia δ_ab  (stay consistent).
    """
    L, q = f1.shape
    f1 = (1 - lam) * f1 + lam / q
    f2 = (1 - lam) * f2 + lam / q**2
    diag = torch.einsum("ia,ab->iab", f1, torch.eye(q, device=f1.device))  # (L, q, q): f_ia δ_ab
    idx = torch.arange(L, device=f1.device)
    f2[idx, :, idx, :] = diag
    return f1, f2


def covariance(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """C_{iajb} = f_{iajb} − f_{ia} f_{jb}.   (L, q), (L, q, L, q) → (L, q, L, q);  outer product, nothing summed."""
    return f2 - torch.einsum("ia,jb->iajb", f1, f1)


def mean_field_couplings(C: torch.Tensor) -> torch.Tensor:
    """Mean-field DCA:  J_{iajb} = −(C⁻¹)_{iajb} for a, b < q−1  (the last letter is the reference, J = 0).

    The connected correlation matrix C (L(q−1) × L(q−1)) mixes direct and indirect effects; its inverse
    keeps only the direct ones (the same idea as partial correlations in a Gaussian graphical model).
    """
    L, q = C.shape[:2]
    Cr = C[:, : q - 1, :, : q - 1].reshape(L * (q - 1), L * (q - 1))
    Jr = -torch.linalg.inv(Cr).reshape(L, q - 1, L, q - 1)
    J = torch.zeros_like(C)
    J[:, : q - 1, :, : q - 1] = Jr
    idx = torch.arange(L, device=C.device)
    J[idx, :, idx, :] = 0.0  # no self-coupling
    return J


def zero_sum_gauge(J: torch.Tensor) -> torch.Tensor:
    """J_ij(a,b) − mean_a − mean_b + mean_ab: the gauge in which the block holds no single-site part."""
    return J - J.mean(1, keepdim=True) - J.mean(3, keepdim=True) + J.mean((1, 3), keepdim=True)


def frobenius_scores(J: torch.Tensor) -> torch.Tensor:
    """F_ij = sqrt( Σ_ab J_ij(a,b)² ) in the zero-sum gauge, diagonal 0.   (L, q, L, q) → (L, L);  a, b summed."""
    Jz = zero_sum_gauge(J)
    F = torch.einsum("iajb,iajb->ij", Jz, Jz).sqrt()
    return F.fill_diagonal_(0.0)


def apc(F: torch.Tensor) -> torch.Tensor:
    """Average-product correction  F^APC_ij = F_ij − F_i· F_·j / F_··  (means over the whole matrix).

    Columns that are very variable (or poorly sampled) have large scores with everybody; APC subtracts that
    background, and afterwards every row and column of F^APC has mean zero.
    """
    row = F.mean(1)  # F_i·
    return F - torch.outer(row, F.mean(0)) / F.mean()


def pseudo_log_likelihood(x: torch.Tensor, h: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """(1/N) Σ_n Σ_i log P(s_ni | s_n,rest) = (1/N) Σ_{n,i,a} x_{nia} log softmax_a E_{nia}.   scalar."""
    logp = torch.log_softmax(local_field(x, h, J), dim=-1)  # (N, L, q)
    return torch.einsum("nia,nia->", x, logp) / x.shape[0]


def top_pairs(score: torch.Tensor, n_top: int, min_sep: int = 3) -> torch.Tensor:
    """The n_top highest-scoring pairs (i < j, |i−j| ≥ min_sep) as an (n_top, 2) index tensor."""
    L = score.shape[0]
    i, j = torch.triu_indices(L, L, min_sep, device=score.device)  # all pairs with j − i ≥ min_sep
    order = torch.argsort(score[i, j], descending=True)[:n_top]
    return torch.stack([i[order], j[order]], dim=1)


def precision(pairs: torch.Tensor, contacts: torch.Tensor) -> float:
    """Fraction of the predicted pairs that are true contacts."""
    return contacts[pairs[:, 0], pairs[:, 1]].float().mean().item()
