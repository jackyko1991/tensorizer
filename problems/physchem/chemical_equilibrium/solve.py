"""Find the equilibrium extent ξ_b for every condition at once, by bisection on ln Q(ξ) − ln K.

ln Q grows steadily with ξ (more products, fewer reactants), from −∞ at ξ_min to +∞ at ξ_max, so there is exactly
one root in between and halving the bracket is guaranteed to find it: 60 halvings shrink it by 2⁻⁶⁰ ≈ 10⁻¹⁸.
"""

import torch

from problems._common.progress import steps

from . import equation


def equilibrium_extent(
    n0: torch.Tensor, nu: torch.Tensor, P: torch.Tensor, lnK: torch.Tensor, iterations: int = 60
) -> tuple[torch.Tensor, list[float]]:
    """Return ξ_b (B,) with Q_b = K_b, and the largest |ln Q − ln K| after each halving (for the log).

    n0: (B, S) starting moles, nu: (S,), P: (B,) bar, lnK: (B,).
    """
    lo, hi = equation.extent_bounds(n0, nu)  # (B,), (B,)
    history = []
    for _ in steps(iterations, "bisection"):
        mid = 0.5 * (lo + hi)
        f = equation.log_Q(equation.moles(n0, nu, mid), P, nu) - lnK  # (B,) > 0: too far towards products
        hi = torch.where(f > 0, mid, hi)
        lo = torch.where(f > 0, lo, mid)
        history.append(f.abs().max().item())
    return 0.5 * (lo + hi), history


def add_product_and_reequilibrate(
    n_eq: torch.Tensor, nu: torch.Tensor, P: torch.Tensor, lnK: torch.Tensor, extra: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Le Chatelier test: add `extra` mol of the product (last species) to an equilibrium mixture n_eq (B, S).

    Returns (moles just after adding (B, S), ln Q − ln K just after (B,), new equilibrium moles (B, S)).
    """
    n_add = n_eq.clone()
    n_add[:, -1] += extra
    gap = equation.log_Q(n_add, P, nu) - lnK  # > 0: Q > K, so the reaction must run backwards
    xi, _ = equilibrium_extent(n_add, nu, P, lnK)  # extent measured from the new starting point
    return n_add, gap, equation.moles(n_add, nu, xi)
