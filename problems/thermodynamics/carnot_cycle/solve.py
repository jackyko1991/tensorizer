"""Evaluate the cycle: work as the enclosed area, heat on the isotherms, efficiency, and the T–S path."""

import torch

from .equation import entropy, isothermal_heat, temperature, work


def solve(legs: torch.Tensor, n: float, T_h: float, T_c: float, gamma: float) -> dict:
    """legs: (4, K, 2) samples of (V, p).  Returns energies (J) and the path in T–S coordinates."""
    V, p = legs[..., 0], legs[..., 1]  # (4, K) each

    W_legs = work(V, p)  # (4,)  ∫ p dV along each leg (negative when compressed)
    W = W_legs.sum().item()  # ∮ p dV: the area enclosed by the cycle

    V1, V2, V3, V4 = V[:, 0].tolist()  # corner volumes
    Q_h = isothermal_heat(n, T_h, V1, V2)  # absorbed from the hot reservoir (> 0)
    Q_c = -isothermal_heat(n, T_c, V3, V4)  # released to the cold reservoir (> 0)

    T = temperature(V, p, n)  # (4, K)
    S = entropy(V, T, n, gamma)  # (4, K)
    S = S - S.min()  # entropy is only defined up to a constant

    return {
        "W_legs": W_legs,
        "W": W,
        "Q_h": Q_h,
        "Q_c": Q_c,
        "efficiency": W / Q_h,
        "T": T,
        "S": S,
    }
