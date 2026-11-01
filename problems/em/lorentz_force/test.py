"""Physics checks, written first (TDD): what any correct Lorentz-force code must satisfy.

Run with `pytest problems/em/lorentz_force/test.py`. The Levi-Civita cross product is v × B, charges gyrate at the
cyclotron frequency |q|B/m with radius mv/(|q|B), and crossed fields give the E × B drift for any charge.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_levi_civita_cross_product_matches_torch():
    eps = tensors.levi_civita(CPU)
    v = torch.randn(7, 3)
    B = torch.randn(3)
    expected = torch.linalg.cross(v, B.expand_as(v))
    assert torch.allclose(equation.cross(eps, v, B), expected, atol=1e-6)


def test_cyclotron_period_and_radius():
    eps = tensors.levi_civita(CPU)
    E, B = tensors.fields(0.0, 1.5, CPU)
    q, m = tensors.particles(4, CPU)  # charges ±1, +2 and a heavier particle
    y0 = tensors.initial_state(4, 0.0, CPU)  # |v⊥| = 1, no motion along B
    omega = equation.cyclotron_frequency(q, m, B)
    assert torch.allclose(omega, q.abs() * 1.5 / m)  # ω_c = |q|B/m

    period = 2 * math.pi / omega.min().item()  # longest period in the batch
    _, traj = solve.solve(y0, q, m, E, B, eps, t_max=period, dt=period / 2000)
    pos = traj[:, :, 0, :]  # (T, P, 3)
    radius = 0.5 * (pos[..., 0].amax(0) - pos[..., 0].amin(0))  # half the x extent of each circle
    assert torch.allclose(radius, m * 1.0 / (q.abs() * 1.5), rtol=1e-3)  # Larmor radius r = mv/(|q|B)
    # every particle completes a whole number of turns in the longest period, so all are back at the start
    assert torch.allclose(pos[-1], pos[0], atol=1e-4)


def test_exb_drift_velocity():
    eps = tensors.levi_civita(CPU)
    E, B = tensors.fields(0.4, 2.0, CPU)
    u = equation.drift_velocity(E, B, eps)
    assert torch.allclose(u, torch.linalg.cross(E, B) / (B @ B))
    assert torch.allclose(u, torch.tensor([0.2, 0.0, 0.0]))  # E/B along x, for any charge

    q, m = tensors.particles(4, CPU)
    y0 = tensors.initial_state(4, 0.0, CPU)
    period = 2 * math.pi * 2.0 / 2.0  # 2π m/(|q| B) of the heaviest particle (m=2, q=1) — a multiple of the others
    _, traj = solve.solve(y0, q, m, E, B, eps, t_max=period, dt=period / 2000)
    displacement = traj[-1, :, 0, :] - traj[0, :, 0, :]  # (P, 3) after whole gyrations
    assert torch.allclose(displacement / period, u.expand_as(displacement), atol=1e-3)  # E × B drift, charge-blind
