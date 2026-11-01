"""Physics checks for three research-inspired problems: the Markovian Mpemba effect, a photonic Haldane
lattice, and a learned flux correction for coarse-grid Burgers."""

import math

import pytest
import torch

from problems.em.topological_photonics import solve as tp_solve
from problems.fluids.ml_accelerated_cfd import equation as cfd_eq
from problems.fluids.ml_accelerated_cfd import solve as cfd_solve
from problems.fluids.ml_accelerated_cfd import tensors as cfd_t
from problems.thermodynamics.mpemba_effect import equation as mp_eq
from problems.thermodynamics.mpemba_effect import solve as mp_solve
from problems.thermodynamics.mpemba_effect import tensors as mp_t

CPU = torch.device("cpu")

# --- Markovian Mpemba effect ---


@pytest.fixture(scope="module")
def mpemba():
    return mp_solve.build(barrier=2.0, T_bath=1.0, device=CPU)


def test_mpemba_rate_matrix_conserves_probability_and_is_metropolis(mpemba):
    W, u = mpemba["W"], mpemba["u"]
    assert W.sum(dim=0).abs().max() < 1e-14  # every column sums to zero
    off = W - torch.diag(torch.diag(W))
    assert (off >= 0).all() and (torch.diag(W) <= 0).all()
    # detailed balance with the Boltzmann weights: W_ij π_j = W_ji π_i
    pi = mpemba["pi"]
    flux = W * pi[None, :]
    assert torch.allclose(flux, flux.T, atol=1e-15)
    assert torch.allclose(pi, torch.softmax(-u, 0))  # T_b = 1


def test_mpemba_stationary_state_is_boltzmann(mpemba):
    W, lam, right = mpemba["W"], mpemba["lam"], mpemba["right"]
    assert abs(lam[0]) < 1e-12 and (lam[1:] < 0).all()
    stationary = right[:, 0] / right[:, 0].sum()
    assert torch.allclose(stationary, mpemba["pi"], atol=1e-12)
    assert (W @ mpemba["pi"]).abs().max() < 1e-15
    # long-time relaxation from any start ends at π
    p0 = mp_t.boltzmann(mpemba["u"], torch.tensor([5.0], dtype=torch.float64))
    p_end = torch.linalg.matrix_exp(W * 50 / -lam[1].item()) @ p0[0]
    assert torch.allclose(p_end, mpemba["pi"], atol=1e-10)


def test_mpemba_spectral_solution_matches_matrix_exponential(mpemba):
    T0 = torch.tensor([1.5, 3.0, 10.0], dtype=torch.float64)
    p0 = mp_t.boltzmann(mpemba["u"], T0)
    a = mp_eq.mode_amplitudes(p0, mpemba["left"])
    times = torch.tensor([0.0, 10.0, 300.0], dtype=torch.float64)
    p = mp_eq.evolve(a, mpemba["lam"], mpemba["right"], times)  # (T, B, N)
    for k, t in enumerate(times):
        expected = torch.einsum("ij,bj->bi", torch.linalg.matrix_exp(mpemba["W"] * t), p0)
        assert torch.allclose(p[k], expected, atol=1e-12)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))


def test_mpemba_a2_changes_sign_at_mpemba_point(mpemba):
    u, left = mpemba["u"], mpemba["left"]
    T_star = mp_solve.mpemba_point(u, 1.0, left)
    assert 2.0 < T_star < 4.0
    a2 = mp_solve.a2_curve(u, torch.tensor([T_star / 1.05, T_star, T_star * 1.05], dtype=torch.float64), left)
    assert a2[0] * a2[2] < 0  # opposite signs on either side
    assert abs(a2[1]) < 1e-10
    assert abs(mp_solve.a2_curve(u, torch.tensor([1.0], dtype=torch.float64), left)[0]) < 1e-12  # a_2(T_b) = 0


def test_mpemba_hot_start_overtakes_warm_start(mpemba):
    u, lam, right, left, pi = (mpemba[k] for k in ["u", "lam", "right", "left", "pi"])
    T_star = mp_solve.mpemba_point(u, 1.0, left)
    T0 = torch.tensor([1.5, T_star, 3.5], dtype=torch.float64)  # warm, strong-Mpemba, hot
    a = mp_eq.mode_amplitudes(mp_t.boltzmann(u, T0), left)
    assert abs(a[2, 1]) < abs(a[0, 1])  # the hotter start has the smaller slow-mode amplitude
    times = torch.tensor([0.0, 3000.0], dtype=torch.float64)
    kl = mp_eq.kl_divergence(mp_eq.evolve(a, lam, right, times), pi)  # (T, B)
    assert kl[0, 2] > kl[0, 0]  # starts farther from equilibrium ...
    assert kl[1, 2] < kl[1, 0]  # ... but ends closer
    assert kl[1, 1] < 1e-3 * kl[1, 0]  # at T* the slow mode is absent: exponentially faster


# --- topological photonics (Haldane model) ---


@pytest.mark.parametrize("phi, M, C", [(math.pi / 2, 0.0, 1), (-math.pi / 2, 0.0, -1), (math.pi / 2, 0.5, 1)])
def test_haldane_chern_number_topological(phi, M, C):
    r = tp_solve.chern_number(1.0, 0.15, phi, M, 36, CPU)
    assert r["C"] == pytest.approx(C, abs=1e-9)
    assert r["omega"].sum().item() * r["cell"] == pytest.approx(2 * math.pi * C, abs=1e-9)


@pytest.mark.parametrize("phi, M", [(math.pi / 2, 1.2), (0.0, 0.3)])
def test_haldane_chern_number_trivial(phi, M):
    r = tp_solve.chern_number(1.0, 0.15, phi, M, 36, CPU)
    assert r["C"] == pytest.approx(0.0, abs=1e-9)
    assert r["gap"] > 0.1


def test_berry_curvature_formula_integrates_to_two_pi_chern():
    for M, C in [(0.0, 1), (0.5, 1), (1.2, 0)]:
        r = tp_solve.chern_number(1.0, 0.15, math.pi / 2, M, 64, CPU)
        omega = tp_solve.berry_curvature_autograd(r["k"], 1.0, 0.15, math.pi / 2, M)
        assert omega.sum().item() * r["cell"] == pytest.approx(2 * math.pi * C, abs=2e-3)
        # and, at the plaquette centres, the continuum formula agrees with the FHS fluxes point by point
        k = r["k"]
        centre = k + 0.5 * (k[1, 0] - k[0, 0]) + 0.5 * (k[0, 1] - k[0, 0])
        omega_c = tp_solve.berry_curvature_autograd(centre, 1.0, 0.15, math.pi / 2, M)
        assert (omega_c - r["omega"]).norm() < 0.02 * r["omega"].norm() + 1e-9


def test_ribbon_has_edge_states_in_gap_only_when_topological():
    window = 0.3 * 1.559  # well inside the bulk gap 2(3√3 t2) for M = 0
    topo = tp_solve.ribbon_bands(20, 1.0, 0.15, math.pi / 2, 0.0, 101, CPU)
    triv = tp_solve.ribbon_bands(20, 1.0, 0.15, math.pi / 2, 1.56, 101, CPU)
    in_gap = topo["E"].abs() < window
    assert in_gap.any()
    assert ((topo["bottom"] + topo["top"])[in_gap] > 0.9).all()  # every mid-gap state lives on an edge
    assert not (triv["E"].abs() < 0.2).any()  # trivial: nothing crosses the middle of the gap
    v, _ = tp_solve.bottom_edge_mode(topo, window)
    v_rev, _ = tp_solve.bottom_edge_mode(tp_solve.ribbon_bands(20, 1.0, 0.15, -math.pi / 2, 0.0, 101, CPU), window)
    assert v * v_rev < 0 and abs(v) > 0.3  # one-way, and the direction flips with φ


def test_edge_transport_is_robust_to_a_notch():
    t2, phi = 0.15, math.pi / 2
    topo = tp_solve.chern_number(1.0, t2, phi, 0.0, 36, CPU)
    window = 0.3 * topo["gap"]
    v, k_c = tp_solve.bottom_edge_mode(tp_solve.ribbon_bands(20, 1.0, t2, phi, 0.0, 101, CPU), window)
    r = tp_solve.edge_transport(1.0, t2, phi, (0.0, 1.56), 0.3, 3.0, v, k_c, window, 40.0, 24.0, 40, CPU)
    norm = r["intensity"].sum(-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-10)  # unitary evolution
    assert r["transmitted"][0, -1] > 0.85  # topological: almost all light goes around the notch
    assert r["transmitted"][1, -1] < 0.35  # trivial: it does not


# --- ML-accelerated CFD (viscous Burgers) ---


def test_burgers_finite_volume_conserves_mean():
    x, dx = cfd_t.grid(64, CPU)
    u0 = cfd_t.random_initial(x, 4, seed=3).double()
    plain = cfd_solve.rollout(u0, 0.02, dx, 200)
    drift = (cfd_eq.mean(plain) - cfd_eq.mean(u0)).abs().max()
    assert drift < 1e-12
    model = cfd_t.FluxCorrection().double()
    torch.nn.init.normal_(model.net[-1].weight, std=0.1)  # any correction, even an untrained random one
    with torch.no_grad():
        learned = cfd_solve.rollout(u0, 0.02, dx, 50, model)
    assert (cfd_eq.mean(learned) - cfd_eq.mean(u0)).abs().max() < 1e-12


def test_burgers_fine_solver_conserves_mean_and_untrained_model_is_plain():
    g = cfd_solve.setup(512, 32, CPU)
    u0 = cfd_t.random_initial(g["x_f"], 3, seed=4)
    fine, coarse = cfd_solve.ground_truth(u0, 0.02, g["dx_f"], g["factor"], 20)
    assert (cfd_eq.mean(fine) - cfd_eq.mean(u0)).abs().max() < 1e-5
    assert torch.allclose(cfd_eq.mean(coarse), cfd_eq.mean(fine), atol=1e-6)  # averaging keeps the mean
    model = cfd_t.FluxCorrection()
    with torch.no_grad():
        same = cfd_solve.rollout(coarse[0], 0.02, g["dx_c"], 10, model)
    assert torch.allclose(same, cfd_solve.rollout(coarse[0], 0.02, g["dx_c"], 10))  # last layer starts at zero


def test_learned_solver_beats_plain_on_held_out_initial_condition():
    torch.manual_seed(0)
    nu = 0.02
    g = cfd_solve.setup(512, 32, CPU)
    _, truth = cfd_solve.ground_truth(cfd_t.random_initial(g["x_f"], 32, seed=1), nu, g["dx_f"], g["factor"], 100)
    _, test = cfd_solve.ground_truth(cfd_t.random_initial(g["x_f"], 4, seed=7), nu, g["dx_f"], g["factor"], 150)
    model = cfd_t.FluxCorrection()
    losses = cfd_solve.train(model, truth, nu, g["dx_c"], n_iter=150, unroll=12)
    assert losses[-1] < 0.3 * losses[0]
    with torch.no_grad():
        err_plain = cfd_solve.rms_error(cfd_solve.rollout(test[0], nu, g["dx_c"], 150), test).mean(0)
        err_learned = cfd_solve.rms_error(cfd_solve.rollout(test[0], nu, g["dx_c"], 150, model), test).mean(0)
    assert (err_learned < 0.6 * err_plain).all()  # every unseen run, out to t = 3 (training stopped at t = 2)
