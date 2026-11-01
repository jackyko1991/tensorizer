"""Physics checks, written first (TDD): what any correct Rutherford-scattering code must satisfy.

Run with `pytest problems/nuclear/rutherford_scattering/test.py`. Each test is a physical law, not an implementation
detail: the angle follows cot(θ/2) = 2Eb/k, energy and angular momentum are conserved and the distance of closest
approach matches the analytic value.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def rutherford_run(b: torch.Tensor, energy: float = 5.0):
    k = tensors.coupling(79)
    d = k / energy
    m = tensors.ALPHA_MASS
    start = 30 * d
    y0 = tensors.initial_state(b, energy, k, start, CPU)
    dt = 2.2 * start / math.sqrt(2 * energy / m) / 4000
    paths, y_end, _ = solve.simulate(y0, k, m, dt, 4000, len(b), r_detect=5 * d)
    return k, m, d, y0, paths, y_end


def test_angle_vs_impact_parameter():
    energy = 5.0
    k = tensors.coupling(79)
    b = torch.tensor([0.25, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64) * k / energy
    _, _, _, _, _, y_end = rutherford_run(b, energy)
    theta = equation.scattering_angle(y_end[:, 1])
    expected = equation.rutherford_angle(b, energy, k)  # cot(θ/2) = 2Eb/k
    assert torch.allclose(theta, expected, atol=0.01)


def test_conserves_energy_and_angular_momentum():
    k = tensors.coupling(79)
    b = torch.tensor([0.0, 0.3, 1.0, 2.5], dtype=torch.float64) * k / 5.0
    _, m, _, y0, paths, _ = rutherford_run(b)
    e = equation.energy(paths, k, m)  # (T, B)
    lz = equation.angular_momentum(paths, m)
    assert torch.allclose(e, torch.full_like(e, 5.0), rtol=1e-6)  # conservative force: energy conserved
    # central force: angular momentum conserved
    assert torch.allclose(lz, lz[0].expand_as(lz), atol=1e-6 * m * float(b.max()))
    v_inf = math.sqrt(2 * 5.0 / m)
    assert torch.allclose(lz[0].abs(), m * v_inf * b, rtol=1e-12)


def test_distance_of_closest_approach():
    k = tensors.coupling(79)
    b = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64) * k / 5.0
    _, _, _, _, paths, _ = rutherford_run(b)
    r_min = paths[:, :, 0, :].norm(dim=-1).min(dim=0).values
    assert torch.allclose(r_min, equation.closest_approach(b, 5.0, k), rtol=2e-3)  # energy + angular momentum fix r_min
