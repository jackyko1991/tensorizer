"""Physics checks, written first (TDD): what any correct Lagrange-point code must satisfy.

Run with `pytest problems/astro/lagrange_points/test.py`. Each test is a physical law, not an implementation detail:
all five points are equilibria in the rotating frame, L4/L5 form equilateral triangles, the Jacobi constant is
conserved and L4 libration stays bounded.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("mu", [0.001, 0.01, 0.03])
def test_points_are_equilibria_and_L4_L5_equilateral(mu):
    points = solve.lagrange_points(tensors.lagrange_guesses(mu, CPU), mu)
    assert equation.grad_potential(points, mu).abs().max() < 1e-12  # ∇Φ_eff = 0 at the equilibria
    star = torch.tensor([-mu, 0.0, 0.0], dtype=torch.float64)
    planet = torch.tensor([1 - mu, 0.0, 0.0], dtype=torch.float64)
    for k in (3, 4):
        assert (points[k] - star).norm().item() == pytest.approx(1.0, abs=1e-10)
        assert (points[k] - planet).norm().item() == pytest.approx(1.0, abs=1e-10)
    # a particle at rest at any Lagrange point feels no acceleration in the rotating frame
    acc = equation.acceleration(points, torch.zeros_like(points), mu, equation.levi_civita())
    assert acc.abs().max() < 1e-12


def test_jacobi_constant_conserved_and_L4_libration_bounded():
    mu = 0.01
    y0 = tensors.particles_near_l4(mu, [3.0, 10.0], CPU)
    _, traj = solve.solve(y0, mu, t_max=60.0, dt=0.02)
    C = equation.jacobi_constant(traj, mu)
    assert (C - C[0]).abs().max() < 1e-8  # Jacobi integral conserved in the rotating frame
    angle = torch.rad2deg(torch.atan2(traj[:, :, 0, 1], traj[:, :, 0, 0] + mu))
    assert angle.min() > 30 and angle.max() < 90  # stays near L4 at 60°
