"""Build the concrete tensors: the test integrands with their exact integrals, the N values and the Jacobi matrix."""

import math

import torch

NAMES = ["smooth", "kink", "endpoint"]


def integrands(x: torch.Tensor, kink: float, alpha: float) -> torch.Tensor:
    """f_f(x_k) for the three test functions on [−1, 1]: (3, K).

    smooth   1 / (1 + 4x²)    analytic on [−1, 1]
    kink     |x − a|          continuous, but its slope jumps at x = a
    endpoint (1 + x)^α        its derivatives blow up at x = −1 (for α not a whole number)
    """
    return torch.stack([1 / (1 + 4 * x**2), (x - kink).abs(), (1 + x).clamp(min=0) ** alpha])


def exact(kink: float, alpha: float) -> torch.Tensor:
    """The exact integrals over [−1, 1]: atan 2, 1 + a², 2^(α+1)/(α+1): (3,)."""
    return torch.tensor([math.atan(2.0), 1 + kink**2, 2 ** (alpha + 1) / (alpha + 1)], dtype=torch.float64)


def n_values(n_max: int) -> list[int]:
    """About 30 even N from 2 to n_max, evenly spread on a log scale (Simpson needs an even number of intervals)."""
    steps = torch.logspace(math.log10(2), math.log10(n_max), 30, dtype=torch.float64)
    return sorted({2 * round(float(n) / 2) for n in steps})


def jacobi_matrix(n: int, device: torch.device) -> torch.Tensor:
    """J_{kl}: the symmetric tridiagonal matrix of the Legendre recurrence, β_k = k / √(4k² − 1) off the diagonal.

    Its eigenvalues are the N Gauss–Legendre nodes: (N, N).
    """
    k = torch.arange(1, n, dtype=torch.float64, device=device)
    beta = k / torch.sqrt(4 * k**2 - 1)  # (N−1,)
    return torch.diag(beta, 1) + torch.diag(beta, -1)
