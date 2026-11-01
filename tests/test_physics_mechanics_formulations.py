"""Physics checks for the analytical-mechanics formulations: Hamiltonian phase space and least action."""

import math

import pytest
import torch

from problems.mechanics.hamiltonian_phase_space import equation as ham_eq
from problems.mechanics.hamiltonian_phase_space import solve as ham_solve
from problems.mechanics.hamiltonian_phase_space import tensors as ham_t
from problems.mechanics.least_action import equation as la_eq
from problems.mechanics.least_action import solve as la_solve
from problems.mechanics.least_action import tensors as la_t

M, L, G = 1.3, 0.8, 9.81
MGL = M * G * L
P_SEP = 2 * M * L * math.sqrt(G * L)


# --- Hamiltonian phase space ---------------------------------------------------------------------------------


def test_symplectic_matrix():
    J = ham_eq.symplectic_matrix(2)
    assert torch.equal(J @ J, -torch.eye(4, dtype=J.dtype))
    assert torch.equal(J.T, -J)


def test_autograd_hamilton_equations_match_hand_derivation():
    torch.manual_seed(0)
    z = torch.randn(50, 2, dtype=torch.float64) * torch.tensor([2.0, 5.0], dtype=torch.float64)
    auto = ham_eq.hamilton_rhs(ham_eq.pendulum(M, L, G), z)
    hand = ham_eq.pendulum_rhs_by_hand(z, M, L, G)
    assert torch.allclose(auto, hand, atol=1e-12)


def _energy_error(stepper, n_periods: int, dt: float = 0.05) -> torch.Tensor:
    H = ham_eq.pendulum(M, L, G)
    z0 = ham_t.test_orbit(120.0)
    n_steps = int(n_periods * 2 * math.pi * math.sqrt(L / G) / dt)
    traj = ham_solve.integrate(stepper, H, z0, dt, n_steps)  # (T, 1, 2)
    E = H(traj[:, 0])
    return (E - E[0]).abs() / MGL


def test_leapfrog_energy_bounded_while_euler_grows():
    err = _energy_error(ham_solve.leapfrog_step, 60)
    n = len(err)
    first, last = err[: n // 4].max(), err[-n // 4 :].max()
    assert last < 1e-2
    assert last < 1.3 * first  # no secular drift: the error just oscillates
    euler = _energy_error(ham_solve.euler_step, 60)
    assert euler[-1] > 100 * last
    assert euler[n // 2] < euler[-1]  # Euler keeps gaining energy


def test_rk4_drifts_slowly():
    err = _energy_error(ham_solve.rk4_hamilton_step, 60)
    n = len(err)
    assert err[-1] < 1e-3
    assert err[-n // 10 :].mean() > 3 * err[n // 10 : n // 5].mean()  # grows steadily


@pytest.mark.parametrize("factor, rotates", [(0.98, False), (1.02, True)])
def test_separatrix_divides_libration_from_rotation(factor, rotates):
    H = ham_eq.pendulum(M, L, G)
    z0 = torch.tensor([[0.0, factor * P_SEP]], dtype=torch.float64)
    assert (H(z0) > MGL).item() == rotates  # separatrix energy E = mgL
    traj = ham_solve.integrate(ham_solve.leapfrog_step, H, z0, 0.005, 3000)
    went_over = traj[:, 0, 0].abs().max() > math.pi
    assert went_over.item() == rotates


def test_separatrix_curve_has_energy_mgl():
    q = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64)
    z = torch.stack([q, ham_eq.separatrix_momentum(q, M, L, G)], dim=1)
    assert torch.allclose(ham_eq.pendulum(M, L, G)(z), torch.full_like(q, MGL))


def test_shoelace_area_of_ellipse():
    blob = ham_t.blob((0.3, 1.0), (0.5, 2.0), k=4000)
    assert ham_eq.shoelace_area(blob).item() == pytest.approx(math.pi * 0.5 * 2.0, rel=1e-5)


def test_leapfrog_conserves_blob_area_euler_does_not():
    H = ham_eq.pendulum(M, L, G)
    blob = ham_t.blob((0.0, 0.6 * P_SEP), (0.3, 0.1 * P_SEP), k=2000)
    leap = ham_solve.integrate(ham_solve.leapfrog_step, H, blob, 0.05, 400)
    area = ham_eq.shoelace_area(leap)
    assert ((area / area[0]) - 1).abs().max() < 1e-3
    euler = ham_solve.integrate(ham_solve.euler_step, H, blob, 0.05, 20)
    a_euler = ham_eq.shoelace_area(euler)
    assert a_euler[-1] / a_euler[0] > 1.1


def test_small_oscillation_period():
    H = ham_eq.pendulum(M, L, G)
    period0 = 2 * math.pi * math.sqrt(L / G)
    measured = ham_solve.small_oscillation_period(H, math.radians(1.0), 5.2 * period0, dt=0.002)
    assert measured == pytest.approx(period0, rel=1e-4)


def test_harmonic_period_is_amplitude_independent():
    H = ham_eq.harmonic(M, L, G)
    period0 = 2 * math.pi * math.sqrt(L / G)
    measured = ham_solve.small_oscillation_period(H, math.radians(60.0), 3.2 * period0, dt=0.002)
    assert measured == pytest.approx(period0, rel=1e-4)


# --- least action ----------------------------------------------------------------------------------------------


def _setup(system: str, T: float, N: int = 30, omega: float = 2.0, q_start: float = 0.0, q_end: float = 1.0):
    m, g = 1.0, 9.81
    V, dV = la_eq.gravity(m, g) if system == "gravity" else la_eq.oscillator(m, omega)
    t = la_t.time_grid(T, N)
    D = la_eq.difference_matrix(N)
    exact = la_eq.exact_path(t, q_start, q_end, system, g, omega)
    q0 = la_t.trial_paths(t, q_start, q_end, 4, 1.0)
    return {"m": m, "V": V, "dV": dV, "t": t, "D": D, "dt": T / N, "exact": exact, "q0": q0}


def test_difference_matrix():
    D = la_eq.difference_matrix(4)
    q = torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0]], dtype=torch.float64)
    assert torch.equal(torch.einsum("nk,bk->bn", D, q), torch.tensor([[1.0, 3.0, 5.0, 7.0]], dtype=torch.float64))


def test_trial_paths_keep_end_points():
    s = _setup("gravity", 1.0)
    q0 = s["q0"]
    assert torch.allclose(q0[:, 0], torch.zeros(4, dtype=q0.dtype), atol=1e-12)
    assert torch.allclose(q0[:, -1], torch.ones(4, dtype=q0.dtype), atol=1e-12)


def test_gradient_descent_finds_parabola():
    s = _setup("gravity", 1.2)
    hist = la_solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 400, "gradient")
    # constant force: the Verlet recurrence is exact, so the discrete optimum *is* the sampled parabola
    assert (hist["paths"][-1] - s["exact"]).abs().max() < 1e-6
    S_exact = la_eq.action(s["exact"][None], s["D"], s["dt"], s["m"], s["V"])
    assert torch.allclose(hist["S"][-1], S_exact.expand(4), atol=1e-9)


@pytest.mark.parametrize("N", [20, 40])
def test_newton_finds_oscillator_sine_to_discretisation_accuracy(N):
    s = _setup("oscillator", 1.2, N=N)
    hist = la_solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    err = (hist["paths"][-1] - s["exact"]).abs().max().item()
    assert err < 0.5 * (2.0 * s["dt"]) ** 2  # O(Δt²): halves the step, quarters the error
    if N == 40:
        coarse = _setup("oscillator", 1.2, N=20)
        h2 = la_solve.relax(coarse["q0"], coarse["D"], coarse["dt"], 1.0, coarse["V"], 150, "newton")
        err20 = (h2["paths"][-1] - coarse["exact"]).abs().max().item()
        assert 3.0 < err20 / err < 5.0


def test_gradient_equals_minus_dt_times_euler_lagrange_residual():
    s = _setup("oscillator", 1.0)
    q = s["q0"].clone()
    x = q[:, 1:-1].clone().requires_grad_(True)
    S = la_eq.action(la_solve.with_ends(x, q[:, :1], q[:, -1:]), s["D"], s["dt"], s["m"], s["V"])
    (dS,) = torch.autograd.grad(S.sum(), x)
    resid = la_eq.euler_lagrange_residual(q, s["dt"], s["m"], s["dV"])
    assert torch.allclose(dS, -s["dt"] * resid, atol=1e-12)


def test_gradient_vanishes_at_the_optimum():
    s = _setup("oscillator", 1.2)
    hist = la_solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 600, "gradient")
    assert hist["grad"][-1].abs().max() < 1e-7
    resid = la_eq.euler_lagrange_residual(hist["paths"][-1], s["dt"], s["m"], s["dV"])
    assert resid.abs().max() < 1e-5


def test_optimum_has_least_action_for_short_T():
    s = _setup("oscillator", 1.0)
    hist = la_solve.relax(s["q0"][:1], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    best = hist["paths"][-1]  # (1, N+1)
    S_best = la_eq.action(best, s["D"], s["dt"], s["m"], s["V"])
    torch.manual_seed(1)
    bump = torch.randn(64, best.shape[1], dtype=best.dtype) * 0.05
    bump[:, 0] = bump[:, -1] = 0.0  # keep the end points fixed
    S_pert = la_eq.action(best + bump, s["D"], s["dt"], s["m"], s["V"])
    assert (S_pert > S_best).all()


def test_hessian_tridiagonal_and_positive_definite_for_short_T():
    s = _setup("oscillator", 1.0)
    H = la_solve.hessian(s["q0"], s["D"], s["dt"], s["m"], s["V"])  # (B, N−1, N−1)
    n = H.shape[-1]
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    assert H[:, (i - j).abs() > 1].abs().max() == 0
    m, dt, omega = s["m"], s["dt"], 2.0
    assert torch.allclose(
        torch.diagonal(H, dim1=1, dim2=2), torch.full((4, n), 2 * m / dt - m * omega**2 * dt, dtype=H.dtype)
    )
    assert torch.allclose(torch.diagonal(H, 1, dim1=1, dim2=2), torch.full((4, n - 1), -m / dt, dtype=H.dtype))
    assert torch.linalg.eigvalsh(H).min() > 0


def test_long_T_true_path_is_a_saddle():
    s = _setup("oscillator", 2.5)  # longer than half a period π/ω ≈ 1.57 s
    H = la_solve.hessian(s["exact"][None], s["D"], s["dt"], s["m"], s["V"])[0]
    eig = torch.linalg.eigvalsh(H)
    assert eig.min() < 0 < eig.max()
    hist = la_solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    assert hist["grad"][-1].abs().max() < 1e-8  # Newton still finds the stationary path
    assert (hist["paths"][-1] - s["exact"]).abs().max() < 0.02
