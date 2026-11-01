r"""Physics → tensor translation.

Newton's law of cooling: an object loses heat at a rate proportional to how much hotter it is
than its surroundings,

    \frac{dT}{dt} = -k \, (T - T_{env})

For a batch of cups b = 1..B, each with its own cooling constant k_b, the index form is

    \dot T_b = -k_b \, (T_b - T_{env})

No index is repeated, so nothing is summed: it is a plain element-wise operation on the (B,) tensor.
The exact solution at times t_t is

    T_{tb} = T_{env} + (T_{0,b} - T_{env}) \, e^{-k_b t_t}

where k_b t_t (again no repeated index) is an outer product, einsum('t,b->tb').
The gap T - T_env halves every t_{1/2} = \ln 2 / k.
"""

import math

import torch


def cooling_rate(T: torch.Tensor, k: torch.Tensor, T_env: float) -> torch.Tensor:
    r"""\dot T_b = -k_b (T_b - T_env).   T, k: (B,) → (B,)"""
    return -k * (T - T_env)


def exact_temperature(t: torch.Tensor, T0: torch.Tensor, k: torch.Tensor, T_env: float) -> torch.Tensor:
    r"""T_{tb} = T_env + (T_{0,b} - T_env) e^{-k_b t_t}.   t: (T,), T0, k: (B,) → (T, B)"""
    kt = torch.einsum("t,b->tb", t, k)  # (T, B)  every time with every cup
    return T_env + (T0 - T_env) * torch.exp(-kt)  # T0 (B,) broadcasts over the time rows


def half_life(k: torch.Tensor) -> torch.Tensor:
    """Time for the temperature gap to halve: t½ = ln 2 / k.   (B,) → (B,)"""
    return math.log(2) / k
