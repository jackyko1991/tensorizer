"""Physics checks, written first (TDD): what any correct Lagrangian-mechanics code must satisfy.

Run with `pytest problems/mechanics/lagrangian_autograd/test.py`. Autograd Euler-Lagrange equations match the hand-
derived ones, and the energy q̇·∂L/∂q̇ − L is conserved because L has no explicit time dependence.
"""

import math

import pytest
import torch

from . import equation, solve, tensors


def test_harmonic_oscillator_and_pendulum_match_hand_derived():
    m, k, g, ell = 2.0, 8.0, 9.81, 1.3

    def oscillator(q, q_dot):
        return 0.5 * m * (q_dot**2).sum(-1) - 0.5 * k * (q**2).sum(-1)

    def pendulum(q, q_dot):
        return 0.5 * m * ell**2 * q_dot[..., 0] ** 2 + m * g * ell * torch.cos(q[..., 0])

    for x in (0.3, -1.1, 2.0):
        q = torch.tensor([x], dtype=torch.float64)
        q_dot = torch.tensor([0.7], dtype=torch.float64)
        assert equation.euler_lagrange(oscillator, q, q_dot).item() == pytest.approx(-k / m * x, rel=1e-12)
        assert equation.euler_lagrange(pendulum, q, q_dot).item() == pytest.approx(-g / ell * math.sin(x), rel=1e-12)


def test_spring_pendulum_matches_hand_derived():
    m, k, ell0, g = 1.5, 30.0, 1.0, 9.81
    L = equation.spring_pendulum(m, k, ell0, g)
    for r, th, r_dot, th_dot in [(1.3, 0.4, 0.2, -1.0), (0.8, -2.0, -1.5, 3.0)]:
        q = torch.tensor([r, th], dtype=torch.float64)
        q_dot = torch.tensor([r_dot, th_dot], dtype=torch.float64)
        r_ddot = r * th_dot**2 + g * math.cos(th) - k / m * (r - ell0)
        th_ddot = (-g * math.sin(th) - 2 * r_dot * th_dot) / r
        got = equation.euler_lagrange(L, q, q_dot)
        assert torch.allclose(got, torch.tensor([r_ddot, th_ddot], dtype=torch.float64), atol=1e-12)


def test_spring_pendulum_energy_is_conserved():
    L = equation.spring_pendulum(1.0, 30.0, 1.0, 9.81)
    sol = solve.solve(tensors.initial_state(1.4, 30.0), L, t_max=5.0)
    E = sol["E"]
    assert ((E - E[0]) / E[0]).abs().max() < 1e-6  # L has no explicit t: the energy function is conserved
    # the autograd energy q̇·∂L/∂q̇ − L equals the hand-written T + V
    r, th = sol["q"][:, 0], sol["q"][:, 1]
    r_dot, th_dot = sol["q_dot"][:, 0], sol["q_dot"][:, 1]
    by_hand = 0.5 * (r_dot**2 + r**2 * th_dot**2) - 9.81 * r * torch.cos(th) + 15.0 * (r - 1.0) ** 2
    assert torch.allclose(E, by_hand, atol=1e-12)
