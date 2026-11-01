r"""Maths → tensor translation.

A periodic odd function on (−π, π) is a sum of sines, f(x) = \sum_n b_n \sin(nx), with coefficients

    b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\,\sin(nx)\,dx.

On a grid x_j with cell widths w_j the integral becomes a sum. Put the sampled waves in F_{wj} (w = which wave,
j = grid point) and the basis in a matrix B_{jn} = sin(n x_j); then all coefficients of all waves are one einsum,

    c_{wn} = \tfrac{1}{\pi}\,w_j\,F_{wj}\,B_{jn}          (j summed),

and the partial sum with the first N harmonics is one matrix product, S^{(N)}_{wj} = B_{jn} c_{wn} for n ≤ N
(n summed). The squared L² error is E_w = \tfrac{1}{\pi} w_j (F_{wj} − S_{wj})² (j summed).
"""

import math

import torch


def coefficients(f: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""c_{wn} = (1/π) w_j F_{wj} B_{jn}   (j summed).   F (W, X), B (X, N), w (X,) → (W, N)"""
    return torch.einsum("j,wj,jn->wn", w, f, b) / math.pi


def partial_sum(b: torch.Tensor, c: torch.Tensor, n: int) -> torch.Tensor:
    r"""S^{(N)}_{wj} = B_{jn} c_{wn} over the first N harmonics   (n summed).   → (W, X)"""
    return torch.einsum("jn,wn->wj", b[:, :n], c[:, :n])


def l2_error_sq(f: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""E_w = (1/π) w_j (F_{wj} − S_{wj})²   (j summed).   → (W,)"""
    d = f - s
    return torch.einsum("j,wj,wj->w", w, d, d) / math.pi


def energy(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""(1/π) ∫ f² dx = (1/π) w_j F_{wj} F_{wj}: the left side of Parseval's identity   → (W,)"""
    return torch.einsum("j,wj,wj->w", w, f, f) / math.pi
