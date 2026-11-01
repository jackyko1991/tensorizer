"""Physics checks, written first (TDD): what any correct string-junction code must satisfy.

Run with `pytest problems/waves/string_junction/test.py`. Reflection and transmission follow the impedance formulas, R²
+ (Z₂/Z₁) T² = 1 (energy flux balance), and total energy is conserved.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("mu_ratio", [4.0, 0.25, 9.0])
def test_reflection_transmission_and_energy(mu_ratio):
    T, mu1, mu2 = tensors.TENSION, tensors.MU1, mu_ratio
    c1, c2 = tensors.speed(mu1), tensors.speed(mu2)
    x, dx = tensors.grid(1001, CPU)
    mu = tensors.density(x, mu1, mu2)
    dt = 0.9 * dx / max(c1, c2)
    u_prev, u0 = tensors.right_moving_pulse(x, 0.04, c1, dt)
    t_meas = solve.measuring_time(c1, c2, tensors.X0)
    times, traj = solve.solve(u_prev, u0, mu, T, dx, dt, t_meas)
    R_num, T_num = solve.measure(x, traj[-1])
    Z1, Z2 = equation.impedance(T, mu1), equation.impedance(T, mu2)
    R, Tr = equation.reflection_transmission(Z1, Z2)
    assert R_num == pytest.approx(R, abs=0.01)
    assert T_num == pytest.approx(Tr, abs=0.015)
    assert R**2 + Z2 / Z1 * Tr**2 == pytest.approx(1.0)  # exact identity
    assert R_num**2 + Z2 / Z1 * T_num**2 == pytest.approx(1.0, abs=0.02)  # measured flux balance
    E = equation.energy(traj, mu, T, dx, dt)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-3)
