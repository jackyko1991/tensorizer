"""Physics checks, written first (TDD): what any correct hydrostatics and buoyancy code must satisfy.

Run with `pytest problems/fluids/hydrostatics_buoyancy/test.py`. Pressure grows linearly with depth (p = p₀ + ρgh),
floating blocks submerge the fraction ρ_block/ρ_water (Archimedes), and blocks denser than water sink.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
G = 9.81


def test_pressure_is_linear_in_depth():
    g = tensors.gravity(G, CPU)
    points = tensors.tank_points(1.0, 2.0, 5, 41, CPU)  # (nz, nx, 2), z from -2 to 0
    p = equation.pressure(points, g, 101_325.0, 1000.0)
    depth = -points[..., 1]
    assert torch.allclose(p, 101_325.0 + 1000.0 * G * depth, rtol=1e-6)  # hydrostatic balance: p = p₀ + ρgh
    assert torch.allclose(p[:, 0], p[:, -1])  # same depth ⇒ same pressure


def test_floating_fraction_matches_archimedes():
    rho_b = tensors.densities(5, 200.0, 900.0, CPU)
    y0 = tensors.initial_state(5, 0.1, CPU)
    _, traj = solve.solve(y0, rho_b, 0.2, depth=1.0, t_max=8.0)
    # Archimedes: weight = weight of displaced water ⇒ submerged fraction ρ_b/ρ_w
    assert torch.allclose(solve.floating_fraction(traj, 0.2), rho_b / 1000.0, atol=2e-3)


def test_dense_blocks_sink_to_the_floor():
    rho_b = torch.tensor([1200.0, 2500.0])
    _, traj = solve.solve(tensors.initial_state(2, 0.1, CPU), rho_b, 0.2, depth=1.0, t_max=5.0)
    assert torch.allclose(traj[-1, :, 0], torch.tensor([-1.0, -1.0]))  # buoyancy < weight: they sink
    assert torch.allclose(traj[-1, :, 1], torch.zeros(2))
