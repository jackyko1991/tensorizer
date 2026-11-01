r"""Maths → tensor translation.

Every quadrature rule replaces the integral by a weighted sum over nodes x_k,

    \int_{-1}^{1} f(x)\,dx \approx \sum_k w_k\,f(x_k),

so with several integrands stacked in F_{fk} = f_f(x_k) all integrals are one einsum, I_f = w_k F_{fk} (k summed).
The rules differ only in their nodes and weights:

- trapezoid, N intervals of width h:  w = h·(½, 1, 1, …, 1, ½)                      error ∝ h²
- Simpson, N even:                    w = h/3·(1, 4, 2, 4, …, 2, 4, 1)              error ∝ h⁴
- Gauss–Legendre, N nodes: the nodes are the eigenvalues of the Jacobi matrix J and the weights come from the
  first component of each eigenvector, J V = V diag(x),  w_k = 2 V_{0k}²  (Golub–Welsch). Exact for every
  polynomial of degree ≤ 2N − 1, so for smooth f the error falls exponentially.
"""

import torch


def integral(w: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    r"""I_f = w_k F_{fk}   (k summed over the nodes).   w (K,), F (n_f, K) → (n_f,)"""
    return torch.einsum("k,fk->f", w, f)


def trapezoid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights of the composite trapezoid rule with n intervals: (n+1,), (n+1,)."""
    x = torch.linspace(-1, 1, n + 1, dtype=torch.float64, device=device)
    w = torch.full_like(x, 2 / n)
    w[0] = w[-1] = 1 / n  # half weight at the two ends
    return x, w


def simpson(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights of the composite Simpson rule with n (even) intervals: (n+1,), (n+1,)."""
    x = torch.linspace(-1, 1, n + 1, dtype=torch.float64, device=device)
    h = 2 / n
    w = torch.full_like(x, 2 * h / 3)
    w[1::2] = 4 * h / 3  # odd nodes are the midpoints of each pair of intervals
    w[0] = w[-1] = h / 3
    return x, w


def gauss_legendre(jacobi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Golub–Welsch: J V = V diag(x), w_k = μ₀ V_{0k}² with μ₀ = ∫_{-1}^{1} dx = 2.   → nodes (N,), weights (N,)"""
    x, v = torch.linalg.eigh(jacobi)  # eigenvalues in ascending order, eigenvectors in the columns of v
    return x, 2 * v[0] ** 2
