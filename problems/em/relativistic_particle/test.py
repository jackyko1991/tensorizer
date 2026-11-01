"""Physics checks, written first (TDD): what any correct relativistic-particle code must satisfy.

Run with `pytest problems/em/relativistic_particle/test.py`. The four-velocity keeps U·U = 1, the covariant four-force
reduces to the Lorentz force, the cyclotron frequency is qB/(γm), and a constant E field gives hyperbolic motion
that never exceeds c.
"""

import math

import torch

from . import equation, solve, tensors


def setup(config: str, n: int, strength: float = 1.0):
    eta = tensors.metric()
    E, B = tensors.fields(config, strength)
    F_mixed = equation.lower_second(equation.field_tensor(E, B, tensors.levi_civita()), eta)
    qm, y0 = tensors.particles(config, n, 0.95)
    return eta, F_mixed, qm, y0


def test_four_velocity_norm_is_preserved():
    for config in ["cyclotron", "accelerate"]:
        eta, F_mixed, qm, y0 = setup(config, 3)
        _, traj = solve.solve(y0, qm, F_mixed, t_max=5.0)
        U = traj[:, :, 1]  # (T, P, 4)
        assert torch.allclose(equation.minkowski_norm(y0[:, 1], eta), torch.ones(3, dtype=tensors.DTYPE))
        # F antisymmetric ⇒ d(U·U)/dτ = 0: the four-velocity stays on the mass shell U·U = 1
        assert ((equation.minkowski_norm(U, eta) - 1) / U[..., 0] ** 2).abs().max() < 1e-8


def test_four_force_is_the_lorentz_force():
    eta = tensors.metric()
    E = torch.tensor([0.3, -0.2, 0.5], dtype=tensors.DTYPE)
    B = torch.tensor([0.1, 0.4, -0.7], dtype=tensors.DTYPE)
    F_mixed = equation.lower_second(equation.field_tensor(E, B, tensors.levi_civita()), eta)
    v = torch.tensor([0.2, 0.5, -0.1], dtype=tensors.DTYPE)
    gamma = 1 / torch.sqrt(1 - v @ v)
    U = gamma * torch.cat([torch.ones(1, dtype=tensors.DTYPE), v])[None]  # (1, 4)
    dU = equation.four_force(U, torch.ones(1, dtype=tensors.DTYPE), F_mixed)[0]
    assert torch.allclose(dU[1:], gamma * (E + torch.linalg.cross(v, B)))  # spatial part: γ q(E + v × B)
    assert torch.allclose(dU[0], gamma * (E @ v))  # time part: γ q E·v, the power


def test_relativistic_cyclotron_frequency():
    _, F_mixed, qm, y0 = setup("cyclotron", 4, strength=1.5)
    gamma = y0[:, 1, 0]
    t = torch.linspace(0, 30, 3001, dtype=tensors.DTYPE).numpy()
    _, traj = solve.solve(y0, qm, F_mixed, t_max=30.0, dtau=0.005)
    lab = solve.to_lab_time(traj, t)
    omega = torch.tensor(2 * math.pi * lab["turns"][-1] / t[-1])  # average angular frequency in lab time
    assert torch.allclose(omega, 1.5 / gamma, rtol=1e-4)  # qB/(γm), not qB/m
    v = y0[:, 1, 2] / gamma
    radius = torch.tensor(0.5 * (lab["x"][:, :, 0].max(0) - lab["x"][:, :, 0].min(0)))
    assert torch.allclose(radius, gamma * v / 1.5, rtol=1e-3)  # r = γmv/(qB)


def test_constant_field_gives_hyperbolic_motion():
    _, F_mixed, qm, y0 = setup("accelerate", 3, strength=0.8)
    t = torch.linspace(0, 10, 101, dtype=tensors.DTYPE)
    _, traj = solve.solve(y0, qm, F_mixed, t_max=10.0)
    x = torch.tensor(solve.to_lab_time(traj, t.numpy())["x"][:, :, 0])  # (F, P)
    a = qm * 0.8  # (P,)
    exact = (torch.sqrt(1 + (t[:, None] * a) ** 2) - 1) / a
    assert torch.allclose(x, exact, atol=1e-3)  # hyperbolic motion under constant proper acceleration
    assert (x <= t[:, None] + 1e-9).all()  # never faster than light
