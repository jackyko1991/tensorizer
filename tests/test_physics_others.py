import math

import pytest
import torch

from problems.others.lorenz_attractor import equation as lz_eq
from problems.others.lorenz_attractor import solve as lz_solve
from problems.others.lorenz_attractor import tensors as lz_t
from problems.others.receptor_binding import equation as rb_eq
from problems.others.receptor_binding import solve as rb_solve
from problems.others.receptor_binding import tensors as rb_t

CPU = torch.device("cpu")


def _binding_run(t_assoc=120.0, t_dissoc=240.0, depletion=False, R_T=10.0, log10_L=0.0):
    k_on, k_off = rb_t.rate_constants(CPU)
    L = rb_t.ligand_concentrations(log10_L, CPU)
    y0 = rb_t.initial_state(R_T, len(k_on), len(L), CPU)
    times, traj = rb_solve.solve(y0, L, k_on, k_off, t_assoc, t_dissoc, depletion)
    return times, traj, L, k_on, k_off


def test_receptor_binding_equilibrium_occupancy():
    k_on, k_off = (k.double() for k in rb_t.rate_constants(CPU))  # float64: R_T - C is tiny near saturation
    K_D = rb_eq.dissociation_constant(k_on, k_off)
    assert torch.allclose(K_D, torch.tensor([1e4, 1e3, 50.0, 0.1], dtype=torch.float64), rtol=1e-5)  # nM
    L = torch.tensor([0.0, 1.0, 100.0, 1e4], dtype=torch.float64)
    theta = rb_eq.occupancy(L, K_D)
    assert torch.allclose(theta, L[None, :] / (L[None, :] + K_D[:, None]))
    # half occupancy exactly at L = K_D
    assert torch.allclose(rb_eq.occupancy(K_D, K_D).diagonal(), torch.full((4,), 0.5, dtype=torch.float64))
    # the flux vanishes at the equilibrium occupancy
    R_T = 10.0
    C = R_T * theta
    J = rb_eq.flux(R_T - C, C, L.expand_as(C), k_on, k_off)
    assert J.abs().max() < 1e-12


def test_receptor_binding_association_matches_exponential_approach():
    t_assoc, R_T = 60.0, 10.0
    times, traj, L, k_on, k_off = _binding_run(t_assoc=t_assoc, t_dissoc=10.0, R_T=R_T)
    n_assoc = int(round(t_assoc / (times[1] - times[0]).item()))
    t = times[: n_assoc + 1]
    C_num = traj[: n_assoc + 1, 1]  # (T, B, C)

    K_D = rb_eq.dissociation_constant(k_on, k_off)
    C_eq = R_T * rb_eq.occupancy(L, K_D)  # (B, C)
    k_obs = rb_eq.observed_rate(L, k_on, k_off)  # (B, C)
    C_exact = C_eq * (1 - torch.exp(-k_obs * t[:, None, None]))
    assert (C_num - C_exact).abs().max() < 1e-3 * R_T

    # the observed rate recovered from the numerical curve: ln(1 - C/C_eq) = -k_obs t
    b, c = 1, len(L) - 1  # medium TCR at the sensorgram concentration (1 µM): k_obs = 0.1 s⁻¹
    i = int(round(10.0 / (times[1] - times[0]).item()))
    k_fit = -math.log(1 - (C_num[i, b, c] / C_eq[b, c]).item()) / t[i].item()
    assert math.isclose(k_fit, k_obs[b, c].item(), rel_tol=1e-3)
    assert math.isclose(k_obs[b, c].item(), 5e4 * 1e-6 + 5e-2, rel_tol=1e-5)


def test_receptor_binding_dissociation_half_life():
    t_assoc, t_dissoc = 120.0, 200.0
    times, traj, L, k_on, k_off = _binding_run(t_assoc=t_assoc, t_dissoc=t_dissoc)
    dt = (times[1] - times[0]).item()
    i_wash = int(round(t_assoc / dt))
    C_wash = traj[i_wash, 1]  # (B, C)
    t_off = times[i_wash:] - times[i_wash]
    C_exact = C_wash * torch.exp(-k_off[None, :, None] * t_off[:, None, None])
    assert (traj[i_wash:, 1] - C_exact).abs().max() < 1e-4 * 10.0

    # after one half-life of the high-affinity TCR (139 s), half its complexes remain
    t_half = rb_eq.half_life(k_off)
    assert math.isclose(t_half[2].item(), math.log(2) / 5e-3, rel_tol=1e-6)
    j = int(round(t_half[2].item() / dt))
    ratio = (traj[i_wash + j, 1, 2, -1] / C_wash[2, -1]).item()
    assert abs(ratio - 0.5) < 2e-3


def test_receptor_binding_conserves_receptors():
    for depletion in (False, True):
        R_T = 10.0
        _, traj, *_ = _binding_run(t_assoc=50.0, t_dissoc=50.0, depletion=depletion, R_T=R_T)
        R, C = traj[:, 0], traj[:, 1]
        assert torch.allclose(R + C, torch.full_like(R, R_T), atol=1e-3)  # float32 round-off over ~2000 steps
        assert C.min() > -1e-5 and C.max() < R_T + 1e-5


def test_receptor_binding_depletion_quadratic_steady_state():
    R_T = 10.0  # nM, comparable to the ligand levels → depletion matters
    times, traj, L, k_on, k_off = _binding_run(t_assoc=600.0, t_dissoc=10.0, depletion=True, R_T=R_T)
    i_wash = int(round(600.0 / (times[1] - times[0]).item()))
    K_D = rb_eq.dissociation_constant(k_on, k_off)
    theta_quad = rb_eq.occupancy_with_depletion(L, K_D, R_T)

    # pairs that have relaxed (k_obs t ≫ 1) must sit on the quadratic solution
    k_obs = rb_eq.observed_rate(L, k_on, k_off)
    relaxed = k_obs * 600.0 > 20
    theta_num = traj[i_wash, 1] / R_T
    assert relaxed.sum() > 20
    assert (theta_num - theta_quad)[relaxed].abs().max() < 1e-3

    # the quadratic root solves k_on (R_T − C)(L_T − C) = k_off C ...
    k_on64, k_off64, L64 = k_on.double(), k_off.double(), L.double()  # float64: R_T - C is tiny near saturation
    C = rb_eq.occupancy_with_depletion(L64, rb_eq.dissociation_constant(k_on64, k_off64), R_T) * R_T
    binding = k_on64[:, None] * (R_T - C) * (L64[None, :] - C)
    unbinding = k_off64[:, None] * C
    assert ((binding - unbinding).abs() <= 1e-9 * unbinding + 1e-15).all()
    # ... and depletion can only lower occupancy compared with an inexhaustible ligand supply
    assert (theta_quad <= rb_eq.occupancy(L, K_D) + 1e-6).all()
    # antibody with 1 nM ligand and 10 nM receptors: at most 10 % of receptors can be bound
    c_1nM = int(torch.argmin((L - 1.0).abs()))
    assert theta_quad[3, c_1nM] < 0.1


# --- Lorenz attractor ---

LORENZ = (10.0, 28.0, 8.0 / 3.0)


def test_lorenz_tensor_form_matches_explicit_equations():
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    x = torch.randn(20, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 15
    assert torch.allclose(lz_eq.rhs(x, A, Q), lz_eq.rhs_explicit(x, *LORENZ))


def test_lorenz_fixed_points_have_zero_velocity():
    sigma, rho, beta = LORENZ
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    p = lz_t.fixed_points(*LORENZ)
    c = math.sqrt(beta * (rho - 1))
    expected = torch.tensor([[0, 0, 0], [c, c, rho - 1], [-c, -c, rho - 1]], dtype=torch.float64)
    assert torch.allclose(p, expected)
    assert lz_eq.rhs(p, A, Q).abs().max() < 1e-12
    assert len(lz_t.fixed_points(10.0, 0.5, beta)) == 1  # for ρ < 1 only the origin


def test_lorenz_divergence_is_constant_volume_contraction():
    sigma, rho, beta = LORENZ
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    x = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 10
    div = lz_eq.divergence(x, A, Q)
    assert torch.allclose(div, torch.full((5,), -(sigma + 1 + beta), dtype=torch.float64))
    # same as the trace of the Jacobian computed by autograd
    for xb in x:
        J = torch.autograd.functional.jacobian(lambda v: lz_eq.rhs(v[None], A, Q)[0], xb)
        assert torch.trace(J).item() == pytest.approx(-(sigma + 1 + beta))


def test_lorenz_lyapunov_exponent_is_about_point_nine():
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    center = lz_solve.settle(A, Q)
    x0 = lz_t.initial_ball(center, 1e-6, 8)
    times, traj = lz_solve.solve(x0, A, Q, t_max=25.0)
    s = lz_eq.spread(traj)
    lam, t0, t1, _ = lz_solve.lyapunov_fit(times, s)
    assert 0.7 < lam < 1.1  # literature: 0.906
    assert s[-1] > 1e6 * s[0]  # the butterfly effect: spread over the attractor


def test_lorenz_below_rho_one_decays_to_origin():
    A, Q = lz_t.lorenz_tensors(10.0, 0.5, 8.0 / 3.0)
    x0 = lz_t.initial_ball(torch.tensor([5.0, -3.0, 10.0], dtype=torch.float64), 2.0, 6)
    _, traj = lz_solve.solve(x0, A, Q, t_max=30.0)
    assert traj[-1].norm(dim=1).max() < 1e-3
