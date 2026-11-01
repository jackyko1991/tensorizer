"""Physics checks, written first (TDD): what any correct double-pendulum code must satisfy.

Run with `pytest problems/mechanics/double_pendulum/test.py`. Energy is conserved, the mass-matrix solve equals the
textbook equations of motion, small swings follow the analytic normal modes, and nearby starts diverge (chaos).
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_energy_is_conserved():
    params = (1.0, 2.0, 1.0, 0.7, 9.81)
    y0 = tensors.initial_state(120.0, -10.0, 0.01, CPU)
    _, traj = solve.solve(y0, *params, t_max=10.0, dt=0.002)
    e = equation.energy(traj, *params)  # (T, B)
    scale = (params[0] + params[1]) * 9.81 * (params[2] + params[3])  # typical energy size
    assert (e - e[0]).abs().max() < 1e-5 * scale  # no friction: total energy is constant


def test_mass_matrix_solve_matches_explicit_formula():
    """torch.linalg.solve on M_ij q̈_j = f_i equals the textbook closed-form angular accelerations."""
    m1, m2, L1, L2, g = 1.3, 0.8, 1.1, 0.9, 9.81
    t1, t2, w1, w2 = 0.7, -1.2, 0.5, -2.0
    y = torch.tensor([[[t1, t2], [w1, w2]]], dtype=torch.float64)
    d = t1 - t2
    den = 2 * m1 + m2 - m2 * math.cos(2 * d)
    a1 = (
        -g * (2 * m1 + m2) * math.sin(t1)
        - m2 * g * math.sin(t1 - 2 * t2)
        - 2 * math.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * math.cos(d))
    ) / (L1 * den)
    a2 = 2 * math.sin(d) * (w1**2 * L1 * (m1 + m2) + g * (m1 + m2) * math.cos(t1) + w2**2 * L2 * m2 * math.cos(d))
    a2 /= L2 * den
    q_ddot = equation.rhs(y, m1, m2, L1, L2, g)[0, 1]
    assert torch.allclose(q_ddot, torch.tensor([a1, a2], dtype=torch.float64), atol=1e-10)


def test_small_angle_normal_modes():
    g, L = 9.81, 1.0
    omega, e = equation.small_angle_modes(1.0, 1.0, L, L, g)
    analytic = torch.tensor([(2 - math.sqrt(2)) * g / L, (2 + math.sqrt(2)) * g / L], dtype=torch.float64).sqrt()
    assert torch.allclose(omega, analytic, rtol=1e-10)  # ω² = (2 ∓ √2) g/L for equal masses
    # start in the slow mode with a tiny amplitude: the full nonlinear motion is a pure cosine at ω_1
    amp = 1e-3
    q0 = amp * e[:, 0] / e[0, 0]
    y0 = torch.stack([q0, torch.zeros(2, dtype=torch.float64)])[None]  # (1, 2, 2)
    times, traj = solve.solve(y0, 1.0, 1.0, L, L, g, t_max=5.0, dt=0.002)
    expected = amp * torch.cos(omega[0] * times)
    assert (traj[:, 0, 0, 0] - expected).abs().max() < 0.01 * amp


def test_is_chaotic():
    y0 = tensors.initial_state(120.0, -10.0, 1e-4, CPU)
    _, traj = solve.solve(y0, 1.0, 1.0, 1.0, 1.0, 9.81, t_max=20.0)
    gap = (traj[:, 0, 0] - traj[:, 1, 0]).norm(dim=-1)
    assert gap[0] < 1e-5 and gap[-1] > 0.1  # sensitive dependence on initial conditions
