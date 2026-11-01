"""Physics checks for recent-research problems: EGNN dynamics, metalens inverse design, photonic time crystals."""

import math

import torch

from problems.mechanics.egnn_dynamics import equation as egnn_eq
from problems.mechanics.egnn_dynamics import solve as egnn_solve
from problems.mechanics.egnn_dynamics import tensors as egnn_t
from problems.optics.metalens_inverse_design import equation as lens_eq
from problems.optics.metalens_inverse_design import solve as lens_solve
from problems.optics.metalens_inverse_design import tensors as lens_t
from problems.waves.photonic_time_crystal import equation as ptc_eq
from problems.waves.photonic_time_crystal import solve as ptc_solve
from problems.waves.photonic_time_crystal import tensors as ptc_t

CPU = torch.device("cpu")


# --- EGNN ----------------------------------------------------------------------------------------


def _egnn_data(n_train: int, n_test: int = 6, n_steps: int = 20, dt: float = 0.1):
    gen = torch.Generator().manual_seed(1)
    x0, v0, c = egnn_t.initial_states(n_train + n_test, 5, gen, CPU)
    traj = egnn_solve.simulate(x0, v0, c, 0.3, 0.3, dt, n_steps)
    return gen, egnn_solve.pairs(traj[:, :n_train], c[:n_train]), egnn_solve.pairs(traj[:, n_train:], c[n_train:])


def test_true_forces_obey_newtons_third_law():
    gen = torch.Generator().manual_seed(0)
    x, _, c = egnn_t.initial_states(4, 6, gen, CPU)
    a = egnn_eq.acceleration(x.double(), c.double(), 0.3, 0.3)
    assert torch.allclose(a.sum(dim=1), torch.zeros(4, 3, dtype=torch.float64), atol=1e-12)  # total momentum fixed


def test_egnn_is_rotation_and_translation_equivariant_mlp_is_not():
    gen, _, (now, _, c) = _egnn_data(n_train=2)
    R = egnn_t.random_rotation(gen, CPU)
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6) and torch.det(R) > 0
    shift = torch.tensor([0.5, -1.0, 2.0])
    net = egnn_t.egnn_weights(16, 2, gen, CPU)
    mlp = egnn_t.mlp_weights([35, 64, 64, 30], gen, CPU)
    err_egnn = egnn_solve.equivariance_error("egnn", net, now[:, 0], now[:, 1], c, R, shift, 0.1)
    err_mlp = egnn_solve.equivariance_error("mlp", mlp, now[:, 0], now[:, 1], c, R, shift, 0.1)
    assert err_egnn < 1e-5
    assert err_mlp > 1e-2


def test_trained_egnn_beats_mlp_on_held_out_systems():
    torch.manual_seed(0)
    gen, train_set, test_set = _egnn_data(n_train=8)
    net = egnn_t.egnn_weights(32, 3, gen, CPU)
    mlp = egnn_t.mlp_weights([35, 128, 128, 30], gen, CPU)
    h_egnn = egnn_solve.train("egnn", net, egnn_t.parameters(net), train_set, test_set, 0.1, 250, 3e-3)
    h_mlp = egnn_solve.train("mlp", mlp, egnn_t.parameters(mlp), train_set, test_set, 0.1, 250, 3e-3)
    now, nxt, _ = test_set
    drift = ((now[:, 0] + 0.1 * now[:, 1] - nxt[:, 0]) ** 2).mean() + ((now[:, 1] - nxt[:, 1]) ** 2).mean()
    assert h_egnn["test_final"] < h_mlp["test_final"]
    assert h_egnn["test_final"] < 0.5 * drift.item()  # it learned the forces, not just x + v dt


# --- metalens ------------------------------------------------------------------------------------


def _optics():
    return lens_solve.Optics(torch.tensor([0.532, 0.633]), 0.532, 30.0, 40.0, CPU)


def test_angular_spectrum_propagation_conserves_energy():
    x, kx = lens_t.grid(CPU)
    wl = torch.tensor([0.532, 0.633])
    beam = torch.exp(-(x**2) / (2 * 3.0**2)).to(torch.complex64)  # smooth: no evanescent components
    E0 = beam[None].expand(2, -1)
    E = lens_eq.propagate(E0, lens_eq.kz(kx, wl), torch.tensor([0.0, 10.0, 40.0]))  # (L, Z, X)
    power = (E.abs() ** 2).sum(dim=-1)
    assert torch.allclose(power, power[:, :1].expand_as(power), rtol=1e-4)
    assert not torch.allclose(E[:, 2].abs(), E[:, 0].abs(), atol=1e-2)  # but the beam did spread


def test_analytic_lens_focuses_at_f():
    optics = _optics()
    centres = lens_t.atom_centres(40.0, CPU)
    theta = torch.remainder(lens_eq.hyperbolic_theta(centres, 30.0, 0.532), 1.0)
    peak = lens_solve.on_axis_peak(optics, theta, torch.linspace(10.0, 60.0, 501))
    assert abs(peak[0].item() - 30.0) < 0.03 * 30.0
    # the wrapped (diffractive) lens is chromatic: red light focuses closer, near f λ0/λ1
    assert abs(peak[1].item() - 30.0 * 0.532 / 0.633) < 0.08 * 30.0


def test_optimised_lens_at_least_as_good_as_analytic():
    optics = _optics()
    centres = lens_t.atom_centres(40.0, CPU)
    theta_hyp = torch.remainder(lens_eq.hyperbolic_theta(centres, 30.0, 0.532), 1.0)
    eta_hyp = optics.efficiency(theta_hyp)[0].item()
    weights = torch.tensor([1.0, 0.0])
    history, _, _ = lens_solve.optimise(optics, torch.zeros(len(centres)), weights, 300, 0.02, n_snap=2)
    eta_opt = history[-1, 0].item()
    assert eta_hyp > 0.5
    assert eta_opt >= eta_hyp - 1e-3  # from a flat start it reaches (or beats) the analytic design
    assert history[0, 0].item() < 0.1  # the flat surface does not focus


# --- photonic time crystal -----------------------------------------------------------------------


def test_temporal_fresnel_coefficients_keep_D_and_B_continuous():
    for n1, n2 in [(1.0, 2.0), (1.5, 1.0), (1.0, 1.0), (2.0, 3.0)]:
        Df, Db = ptc_eq.temporal_fresnel(n1, n2)
        assert math.isclose(Df + Db, 1.0, abs_tol=1e-12)  # D before = D after
        assert math.isclose(Df / n2 - Db / n2, 1.0 / n1, abs_tol=1e-12)  # B before = B after (B = ±D/n)


def test_simulated_time_reflection_matches_fresnel():
    x, k = ptc_t.grid(CPU)
    D0, B0 = ptc_t.forward_pulse(x, 1.0, 1.0)
    _, u = ptc_solve.interface(D0, B0, k, 1.0, 4.0, 10.0, 40.0, 0.05, 100)
    D_fwd, D_bwd = ptc_eq.split_directions(ptc_solve.to_field(u[-1]), ptc_solve.to_field(u[-1], 1), 2.0)
    Df, Db = ptc_eq.temporal_fresnel(1.0, 2.0)
    assert abs(ptc_solve.envelope(D_fwd, k).max().item() - abs(Df)) < 0.01
    assert abs(ptc_solve.envelope(D_bwd, k).max().item() - abs(Db)) < 0.01


def _floquet(alpha: float, omega: float = 2.0, n: int = 600):
    period = 2 * math.pi / omega
    k = ptc_t.band_wavenumbers(1.5 * omega / 2, n, CPU)  # first gap only (the second sits near k = Ω)

    def eps(t: float) -> float:
        return ptc_eq.periodic_eps(t, 1.0, alpha, omega)

    M = ptc_solve.monodromy(k, eps, period)
    w = ptc_eq.floquet_frequencies(torch.linalg.eigvals(M), period)
    return k, M, w.imag.max(dim=-1).values


def test_floquet_growth_only_inside_the_gap():
    alpha = 0.1
    k, M, growth = _floquet(alpha)
    assert torch.allclose(torch.linalg.det(M), torch.ones(len(k), dtype=M.dtype), atol=1e-8)  # λ+ λ- = 1
    lo, hi = ptc_eq.perturbative_gap(1.0, alpha, 2.0)
    half = (hi - lo) / 2
    inside = (k > lo + 0.3 * half) & (k < hi - 0.3 * half)
    outside = (k < lo - 0.3 * half) | (k > hi + 0.3 * half)
    assert (growth[inside] > 1e-3).all()
    assert (growth[outside] < 1e-6).all()
    assert math.isclose(growth.max().item(), alpha * 2.0 / 8, rel_tol=0.05)  # peak growth ≈ αΩ/8


def test_gap_edges_match_perturbation_theory():
    for alpha in (0.04, 0.08):
        k, _, growth = _floquet(alpha, n=3000)
        in_gap = k[growth > 1e-6]
        lo, hi = ptc_eq.perturbative_gap(1.0, alpha, 2.0)
        tol = alpha**2 + 2 * (k[1] - k[0]).item()  # first-order theory: error O(α²), plus the k spacing
        assert abs(in_gap.min().item() - lo) < tol
        assert abs(in_gap.max().item() - hi) < tol


def test_simulated_mode_grows_at_floquet_rate():
    alpha, omega = 0.2, 2.0
    x, k = ptc_t.grid(CPU)

    def eps(t: float) -> float:
        return ptc_eq.periodic_eps(t, 1.0, alpha, omega)

    D0, B0 = ptc_t.forward_pulse(x, 1.0, math.sqrt(eps(0.0)))
    times, u = ptc_solve.evolve(ptc_solve.to_modes(D0, B0), k, eps, 0.0, 0.05, 1600, 10)
    i = torch.argmin((k - 1.0).abs()).item()
    rate = ptc_solve.growth_rate_measured(u[..., 0].abs(), torch.tensor(times), i, 40.0)
    M = ptc_solve.monodromy(k[i : i + 1], eps, 2 * math.pi / omega)
    expected = ptc_eq.floquet_frequencies(torch.linalg.eigvals(M), 2 * math.pi / omega).imag.max().item()
    assert math.isclose(rate, expected, rel_tol=0.05)
