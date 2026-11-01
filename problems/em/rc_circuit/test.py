"""Physics checks, written first (TDD): what any correct RC-circuit code must satisfy.

Run with `pytest problems/em/rc_circuit/test.py`. The capacitor charges to 1 − 1/e of the supply after one time
constant RC, and energy balances: the source supplies CV², half is stored as ½CV² and half is dissipated in R.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_capacitor_reaches_63_percent_after_one_time_constant():
    R = tensors.resistances(0.5, 4.0, 4, CPU)
    C, Vs, dt = 2.0, 5.0, 0.001
    tau = R * C
    result = solve.solve(tensors.initial_voltage(4, CPU), R, C, Vs, "step", 10.0, t_max=tau.max().item(), dt=dt)
    steps = torch.round(tau / dt).long()
    V_at_tau = result["V"][steps, torch.arange(4)]
    assert torch.allclose(V_at_tau, torch.full((4,), (1 - math.exp(-1)) * Vs), rtol=1e-4)  # V(τ) = (1 − 1/e) V_s


def test_energy_balance():
    R = tensors.resistances(1.0, 3.0, 3, CPU).double()
    C, Vs = 1.5, 4.0
    V0 = tensors.initial_voltage(3, CPU).double()
    result = solve.solve(V0, R, C, Vs, "step", 10.0, t_max=12 * 3.0 * C, dt=0.002)
    U = equation.stored_energy(result["V"][-1], C)
    assert torch.allclose(U, torch.full((3,), 0.5 * C * Vs**2, dtype=U.dtype), rtol=1e-4)  # full: ½CV²
    current, t = result["I"], result["t"].double()
    supplied = torch.trapezoid(Vs * current, t, dim=0)
    heat = torch.trapezoid(current**2 * R, t, dim=0)
    assert torch.allclose(supplied, torch.full((3,), C * Vs**2, dtype=U.dtype), rtol=1e-3)
    assert torch.allclose(heat, U, rtol=1e-3)  # energy conservation: half is lost as heat, whatever R is
