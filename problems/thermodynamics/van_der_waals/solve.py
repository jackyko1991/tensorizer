"""Isotherms for every temperature, plus Maxwell's equal-area line for those below T_c (bisection on tensors)."""

import torch

from .equation import area_mismatch, pressure, volume_roots


def coexistence_pressure(
    T: torch.Tensor, V: torch.Tensor, V_c: float, n: float, a: float, b: float, n_iter: int = 60
) -> torch.Tensor:
    """Bisection for p* with ∫(p − p*) dV = 0, all temperatures (all below T_c) at once.  (B,) → (B,)

    The answer lies between the bottom of the loop (local minimum, at V < V_c) and its top (local maximum,
    at V > V_c). Each iteration halves every bracket in one tensor operation.
    """
    p = pressure(V, T, n, a, b)  # (B, V)
    low = torch.where(V < V_c, p, torch.inf).min(dim=-1).values  # (B,)  bottom of the loop
    high = torch.where(V > V_c, p, -torch.inf).max(dim=-1).values  # (B,)  top of the loop
    low = low.clamp(min=1e-3 * high)  # at low T the loop dips below p = 0; p* is always positive
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        too_low = area_mismatch(mid, T, n, a, b) > 0  # more area above the line than below → raise it
        low = torch.where(too_low, mid, low)
        high = torch.where(too_low, high, mid)
    return 0.5 * (low + high)


def solve(T: torch.Tensor, V: torch.Tensor, T_c: float, V_c: float, n: float, a: float, b: float) -> dict:
    """Return isotherms p (T, V) and, for T < T_c, the flat line: p* and its ends V_l, V_g (T,) (NaN above T_c)."""
    p = pressure(V, T, n, a, b)  # (T, V)
    below = T < T_c  # (T,)
    p_star = torch.full_like(T, float("nan"))
    V_l, V_g = p_star.clone(), p_star.clone()

    if below.any():
        p_star[below] = coexistence_pressure(T[below], V, V_c, n, a, b)
        roots = volume_roots(p_star[below], T[below], n, a, b)  # (B, 3)
        V_l[below], V_g[below] = roots[:, 0], roots[:, 2]
    return {"p": p, "p_star": p_star, "V_l": V_l, "V_g": V_g}
