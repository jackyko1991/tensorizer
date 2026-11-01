"""Physics checks, written first (TDD): what any correct Newton-cooling code must satisfy.

Run with `pytest problems/thermodynamics/newton_cooling/test.py`. The integrated temperature matches the exact
exponential approach to room temperature, and the half-life is ln 2 / k for every cup.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_matches_exponential_and_half_life():
    k = tensors.cooling_constants(4, 0.02, 0.2, CPU).double()
    T0 = tensors.start_temperatures(4, CPU).double()
    times, temps = solve.solve(T0, k, 20.0, 60.0)
    exact = equation.exact_temperature(times, T0, k, 20.0)  # (T, B)
    assert torch.allclose(temps, exact, atol=1e-6)  # T(t) = T_room + (T0 − T_room) e^{-kt}

    # after one half-life the gap to the room has halved, whatever the cup
    t_half = equation.half_life(k)  # (B,)
    gap = equation.exact_temperature(t_half, T0, k, 20.0).diagonal() - 20.0  # cup b at its own t½
    assert torch.allclose(gap, (T0 - 20.0) / 2, rtol=1e-10)
    assert torch.allclose(t_half, math.log(2) / k)  # t½ = ln 2 / k
