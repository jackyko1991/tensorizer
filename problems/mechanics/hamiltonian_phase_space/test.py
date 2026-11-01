"""Physics checks, written first (TDD): what any correct Hamiltonian code must satisfy.

Run with `pytest problems/mechanics/hamiltonian_phase_space/test.py`. Autograd gives Hamilton's equations, symplectic
leapfrog keeps energy bounded and phase-space area constant (Liouville), the separatrix sits at E = mgL, and small
swings have T = 2π√(L/g).
"""

import math

import pytest
import torch

from . import equation, solve, tensors

M, L, G = 1.3, 0.8, 9.81
MGL = M * G * L
P_SEP = 2 * M * L * math.sqrt(G * L)


def test_symplectic_matrix():
    J = equation.symplectic_matrix(2)
    assert torch.equal(J @ J, -torch.eye(4, dtype=J.dtype))
    assert torch.equal(J.T, -J)


def test_autograd_hamilton_equations_match_hand_derivation():
    torch.manual_seed(0)
    z = torch.randn(50, 2, dtype=torch.float64) * torch.tensor([2.0, 5.0], dtype=torch.float64)
    auto = equation.hamilton_rhs(equation.pendulum(M, L, G), z)
    hand = equation.pendulum_rhs_by_hand(z, M, L, G)
    assert torch.allclose(auto, hand, atol=1e-12)


def _energy_error(stepper, n_periods: int, dt: float = 0.05) -> torch.Tensor:
    H = equation.pendulum(M, L, G)
    z0 = tensors.test_orbit(120.0)
    n_steps = int(n_periods * 2 * math.pi * math.sqrt(L / G) / dt)
    traj = solve.integrate(stepper, H, z0, dt, n_steps)  # (T, 1, 2)
    E = H(traj[:, 0])
    return (E - E[0]).abs() / MGL


def test_leapfrog_energy_bounded_while_euler_grows():
    err = _energy_error(solve.leapfrog_step, 60)
    n = len(err)
    first, last = err[: n // 4].max(), err[-n // 4 :].max()
    assert last < 1e-2
    assert last < 1.3 * first  # no secular drift: the error just oscillates
    euler = _energy_error(solve.euler_step, 60)
    assert euler[-1] > 100 * last
    assert euler[n // 2] < euler[-1]  # Euler keeps gaining energy


def test_rk4_drifts_slowly():
    err = _energy_error(solve.rk4_hamilton_step, 60)
    n = len(err)
    assert err[-1] < 1e-3
    assert err[-n // 10 :].mean() > 3 * err[n // 10 : n // 5].mean()  # grows steadily


@pytest.mark.parametrize("factor, rotates", [(0.98, False), (1.02, True)])
def test_separatrix_divides_libration_from_rotation(factor, rotates):
    H = equation.pendulum(M, L, G)
    z0 = torch.tensor([[0.0, factor * P_SEP]], dtype=torch.float64)
    assert (H(z0) > MGL).item() == rotates  # separatrix energy E = mgL
    traj = solve.integrate(solve.leapfrog_step, H, z0, 0.005, 3000)
    went_over = traj[:, 0, 0].abs().max() > math.pi
    assert went_over.item() == rotates


def test_separatrix_curve_has_energy_mgl():
    q = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64)
    z = torch.stack([q, equation.separatrix_momentum(q, M, L, G)], dim=1)
    assert torch.allclose(equation.pendulum(M, L, G)(z), torch.full_like(q, MGL))


def test_shoelace_area_of_ellipse():
    blob = tensors.blob((0.3, 1.0), (0.5, 2.0), k=4000)
    assert equation.shoelace_area(blob).item() == pytest.approx(math.pi * 0.5 * 2.0, rel=1e-5)


def test_leapfrog_conserves_blob_area_euler_does_not():
    H = equation.pendulum(M, L, G)
    blob = tensors.blob((0.0, 0.6 * P_SEP), (0.3, 0.1 * P_SEP), k=2000)
    leap = solve.integrate(solve.leapfrog_step, H, blob, 0.05, 400)
    area = equation.shoelace_area(leap)
    assert ((area / area[0]) - 1).abs().max() < 1e-3
    euler = solve.integrate(solve.euler_step, H, blob, 0.05, 20)
    a_euler = equation.shoelace_area(euler)
    assert a_euler[-1] / a_euler[0] > 1.1


def test_small_oscillation_period():
    H = equation.pendulum(M, L, G)
    period0 = 2 * math.pi * math.sqrt(L / G)
    measured = solve.small_oscillation_period(H, math.radians(1.0), 5.2 * period0, dt=0.002)
    assert measured == pytest.approx(period0, rel=1e-4)


def test_harmonic_period_is_amplitude_independent():
    H = equation.harmonic(M, L, G)
    period0 = 2 * math.pi * math.sqrt(L / G)
    measured = solve.small_oscillation_period(H, math.radians(60.0), 3.2 * period0, dt=0.002)
    assert measured == pytest.approx(period0, rel=1e-4)
