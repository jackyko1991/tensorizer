"""Physics checks for the beginner physical-chemistry problems: rate laws + Arrhenius, equilibrium, Beer–Lambert."""

import math

import pytest
import torch

from problems.physchem.arrhenius_kinetics import equation as ar_eq
from problems.physchem.arrhenius_kinetics import solve as ar_solve
from problems.physchem.arrhenius_kinetics import tensors as ar_t
from problems.physchem.beer_lambert_kinetics import equation as bl_eq
from problems.physchem.beer_lambert_kinetics import solve as bl_solve
from problems.physchem.beer_lambert_kinetics import tensors as bl_t
from problems.physchem.chemical_equilibrium import equation as eq_eq
from problems.physchem.chemical_equilibrium import solve as eq_solve
from problems.physchem.chemical_equilibrium import tensors as eq_t

CPU = torch.device("cpu")
F64 = torch.float64


# ---------------------------------------------------------------- rate laws and Arrhenius


def _ar_runs(Ea=75e3, A=1e11):
    T = ar_t.temperatures(290.0, 340.0, CPU)
    k = ar_eq.rate_constant(T, A, Ea)
    n = ar_t.orders(CPU)
    t_end = ar_t.run_lengths(k, n)
    tau, traj = ar_solve.integrate_runs(ar_t.initial_state(3, len(T), CPU), k, n, t_end)
    return T, k, n, t_end, tau, traj


def test_integrated_rate_laws_match_analytic():
    _, k, _, t_end, tau, traj = _ar_runs()
    for o in range(3):
        t = tau[:, None] * t_end[o][None, :]  # (N+1, T) real times
        exact = ar_eq.integrated_law(t, ar_t.A0, k[None, :], o)
        assert (traj[:, o] - exact).abs().max() < 1e-8, f"order {o}"
    assert torch.allclose(traj[-1], torch.full_like(traj[-1], 0.1), atol=1e-8)  # each run ends at 90 % conversion


def test_half_life_method_gives_the_order():
    _, _, _, t_end, tau, traj = _ar_runs()
    t50 = ar_solve.crossing_time(tau, traj, t_end, 0.5)
    t25 = ar_solve.crossing_time(tau, traj, t_end, 0.25)
    n_est = ar_eq.order_from_half_lives(t50, t25 - t50)
    assert torch.allclose(n_est, torch.tensor([0.0, 1.0, 2.0], dtype=F64)[:, None].expand_as(n_est), atol=1e-3)


def test_arrhenius_fit_recovers_Ea_and_A_exactly_from_exact_k():
    T = torch.linspace(280.0, 360.0, 7, dtype=F64)
    for Ea, A in ((50e3, 1e8), (75e3, 1e11), (100e3, 1e14)):
        Ea_fit, A_fit = ar_eq.arrhenius_fit(T, ar_eq.rate_constant(T, A, Ea))
        assert math.isclose(Ea_fit, Ea, rel_tol=1e-9)
        assert math.isclose(A_fit, A, rel_tol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_noisy_experiment_finds_order_and_activation_energy(order):
    T, k, _, t_end, tau, traj = _ar_runs(Ea=80e3, A=3e12)
    tau_s, clean = ar_solve.sample(tau, traj, order)
    data = ar_t.measure(clean, 0.01)
    times = tau_s[None, :] * t_end[order][:, None]
    beta, R2 = ar_solve.fit_linearizations(times, data)
    assert int(R2.mean(1).argmax()) == order  # the right plot is the straightest
    k_fit = (-1.0 if order < 2 else 1.0) * beta[order, :, 1]
    assert ((k_fit - k) / k).abs().max() < 0.06
    Ea_fit, A_fit = ar_eq.arrhenius_fit(T, k_fit)
    assert abs(Ea_fit - 80e3) / 80e3 < 0.03
    assert abs(math.log(A_fit / 3e12)) < 1.0  # ln A is a long extrapolation to 1/T = 0


def test_least_squares_matches_lstsq():
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(4, 30, generator=gen, dtype=F64)
    y = 2.0 - 3.0 * x + 0.1 * torch.randn(4, 30, generator=gen, dtype=F64)
    beta = ar_eq.least_squares(x, y)
    for b in range(4):
        X = torch.stack([torch.ones(30, dtype=F64), x[b]], 1)
        ref = torch.linalg.lstsq(X, y[b, :, None]).solution[:, 0]
        assert torch.allclose(beta[b], ref, atol=1e-10)


# ---------------------------------------------------------------- chemical equilibrium


def _equilibrium(name, T, P, n0=None):
    r = eq_t.reaction(name, CPU)
    T = torch.as_tensor(T, dtype=F64).reshape(-1)
    P = torch.as_tensor(P, dtype=F64).reshape(-1)
    T, P = torch.broadcast_tensors(T, P)
    n0 = (r["n0"] if n0 is None else torch.as_tensor(n0, dtype=F64)).expand(len(T), -1)
    lnK = eq_eq.log_K(T, r["dH"], r["dS"])
    xi, _ = eq_solve.equilibrium_extent(n0, r["nu"], P, lnK)
    n = eq_eq.moles(n0, r["nu"], xi)
    return r, n, xi, lnK, P


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_equilibrium_satisfies_Q_equals_K(name):
    r = eq_t.reaction(name, CPU)
    T_b, P_b = eq_t.condition_grid(r["T_curves"], eq_t.pressures(CPU, 13))
    _, n, _, lnK, P = _equilibrium(name, T_b, P_b)
    assert (eq_eq.log_Q(n, P, r["nu"]) - lnK).abs().max() < 1e-10
    assert n.min() > 0


def test_n2o4_matches_hand_formula():
    """K = 4α²P/((1−α²)P°)  ⇒  α = √(K/(K + 4P))."""
    T = torch.tensor([280.0, 320.0, 360.0], dtype=F64)
    for P in (0.1, 1.0, 30.0):
        r, _, xi, lnK, _ = _equilibrium("n2o4", T, P)
        K = lnK.exp()
        assert torch.allclose(xi, torch.sqrt(K / (K + 4 * P)), rtol=1e-10)


def test_k298_n2o4_textbook_value():
    r = eq_t.reaction("n2o4", CPU)
    K = math.exp(eq_eq.log_K(torch.tensor(298.15, dtype=F64), r["dH"], r["dS"]).item())
    assert 0.1 < K < 0.2  # textbook K_p ≈ 0.15 at 25 °C


def test_le_chatelier_pressure():
    P = torch.logspace(-1, 2, 7, dtype=F64)
    _, _, xi_n2o4, _, _ = _equilibrium("n2o4", 320.0, P)
    _, _, xi_nh3, _, _ = _equilibrium("haber", 700.0, P)
    assert (xi_n2o4.diff() < 0).all()  # Δν = +1: pressure pushes back to N2O4
    assert (xi_nh3.diff() > 0).all()  # Δν = −2: pressure makes more NH3


def test_le_chatelier_temperature():
    T = torch.linspace(280.0, 380.0, 6, dtype=F64)
    _, _, xi_n2o4, _, _ = _equilibrium("n2o4", T, 1.0)
    _, _, xi_nh3, _, _ = _equilibrium("haber", T + 400.0, 100.0)
    assert (xi_n2o4.diff() > 0).all()  # endothermic: heating favours products
    assert (xi_nh3.diff() < 0).all()  # exothermic: heating favours reactants


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_le_chatelier_added_product(name):
    T = 320.0 if name == "n2o4" else 700.0
    r, n_eq, _, lnK, P = _equilibrium(name, T, 5.0)
    n_add, gap, n_new = eq_solve.add_product_and_reequilibrate(n_eq, r["nu"], P, lnK, 0.3)
    assert gap.item() > 0  # Q > K right after adding product
    assert n_new[0, -1] < n_add[0, -1]  # some product turns back
    assert n_new[0, -1] > n_eq[0, -1]  # ... but not all of it
    assert (n_new[0, :-1] > n_eq[0, :-1]).all()  # reactants increase
    assert abs(eq_eq.log_Q(n_new, P, r["nu"]) - lnK).item() < 1e-10


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_van_t_hoff(name):
    r = eq_t.reaction(name, CPU)
    inv_T = (1.0 / torch.linspace(300.0, 800.0, 11, dtype=F64)).requires_grad_(True)
    lnK = eq_eq.log_K(1.0 / inv_T, r["dH"], r["dS"])
    (slope,) = torch.autograd.grad(lnK.sum(), inv_T)
    assert torch.allclose(slope, torch.full_like(slope, eq_eq.van_t_hoff_slope(r["dH"])), rtol=1e-12)
    # integrated form: ln(K2/K1) = −ΔH°/R (1/T2 − 1/T1)
    T1, T2 = torch.tensor(350.0, dtype=F64), torch.tensor(450.0, dtype=F64)
    lhs = eq_eq.log_K(T2, r["dH"], r["dS"]) - eq_eq.log_K(T1, r["dH"], r["dS"])
    assert math.isclose(lhs.item(), -r["dH"] / eq_eq.R_GAS * (1 / 450.0 - 1 / 350.0), rel_tol=1e-12)
    # ΔG° = −RT ln K
    assert math.isclose(eq_eq.standard_gibbs(T1, r["dH"], r["dS"]).item(),
                        -eq_eq.R_GAS * 350.0 * eq_eq.log_K(T1, r["dH"], r["dS"]).item(), rel_tol=1e-12)  # fmt: skip


# ---------------------------------------------------------------- Beer–Lambert and A → B → C


def _bl_setup(k1=0.02, k2=0.003, sigma=0.0):
    lam = bl_t.wavelengths(CPU)
    E = bl_t.absorptivities(lam)
    c0 = bl_t.initial_concentrations(50.0, CPU)
    t = bl_t.sample_times(1200.0, CPU)
    noise = bl_t.noise((len(t), len(lam)), sigma, CPU)
    c, D = bl_solve.simulate(E, c0, t, k1, k2, bl_t.PATH_LENGTH, noise)
    return lam, E, c0, t, c, D


def test_beer_lambert_linearity():
    lam = bl_t.wavelengths(CPU)
    E = bl_t.absorptivities(lam)
    gen = torch.Generator().manual_seed(2)
    c1, c2 = torch.rand(5, 3, generator=gen, dtype=F64) * 1e-4, torch.rand(5, 3, generator=gen, dtype=F64) * 1e-4
    D = lambda c, length=1.0: bl_eq.absorbance(E, c, length)  # noqa: E731
    assert torch.allclose(D(2.5 * c1), 2.5 * D(c1), rtol=1e-12)  # proportional to concentration
    assert torch.allclose(D(c1, 2.0), 2.0 * D(c1), rtol=1e-12)  # proportional to path length
    assert torch.allclose(D(c1 + c2), D(c1) + D(c2), rtol=1e-12)  # absorbances of a mixture add
    # pure A at its band maximum: D = ε l c with ε = 12000 L/(mol cm)
    w = int((lam - 350.0).abs().argmin())
    pure = torch.tensor([[1e-5, 0.0, 0.0]], dtype=F64)
    assert math.isclose(D(pure)[0, w].item(), 12000 * 1e-5, rel_tol=1e-12)


def test_kinetics_matches_analytic_and_conserves_mass():
    _, _, c0, t, c, _ = _bl_setup(k1=0.02, k2=0.003)
    exact = bl_eq.analytic(0.02, 0.003, c0[0].item(), t)
    assert (c - exact).abs().max() < 1e-12 * 1e6 * c0[0]
    assert (c.sum(1) - c0.sum()).abs().max() < 1e-15  # A + B + C = const
    t_peak = math.log(0.02 / 0.003) / (0.02 - 0.003)
    assert abs(t[c[:, 1].argmax()].item() - t_peak) <= 10.0  # B peaks where dB/dt = 0 (grid spacing 10 s)


def test_pseudo_inverse_matches_torch_and_recovers_concentrations():
    _, E, c0, _, c, D = _bl_setup()
    E_pinv = bl_eq.pseudo_inverse(E)
    assert torch.allclose(E_pinv, torch.linalg.pinv(E), atol=1e-14)
    assert torch.allclose(E_pinv @ E, torch.eye(3, dtype=F64), atol=1e-10)
    c_hat, residual, _ = bl_solve.unmix(E, D, bl_t.PATH_LENGTH)  # noise-free spectra
    assert (c_hat - c).abs().max() < 1e-12 * c0[0] * 1e3
    assert residual.abs().max() < 1e-12


def test_recovery_with_noise_and_mass_balance():
    _, E, c0, _, c, D = _bl_setup(sigma=0.003)
    c_hat, residual, _ = bl_solve.unmix(E, D, bl_t.PATH_LENGTH)
    assert (c_hat - c).abs().max() < 0.01 * c0[0]  # within 1 % of [A]0
    assert ((c_hat.sum(1) - c0.sum()).abs() < 0.01 * c0[0]).all()  # recovered A + B + C ≈ const
    assert abs(residual.pow(2).mean().sqrt().item() - 0.003) < 0.0003  # residual ≈ noise level


def test_isosbestic_point_is_exact_for_two_species():
    lam, E, _, _, c, D = _bl_setup(k2=0.0)  # A → B only
    (x,) = bl_eq.crossings(lam, E[:, 0], E[:, 1])
    assert 380 < x < 420
    assert c[:, 2].abs().max() == 0  # no C
    # interpolate D to the crossing wavelength: constant in time, while the band maxima change a lot
    w = int(torch.searchsorted(lam, torch.tensor(x, dtype=F64))) - 1
    f = (x - lam[w]) / (lam[w + 1] - lam[w])
    D_iso = (1 - f) * D[:, w] + f * D[:, w + 1]
    assert (D_iso - D_iso[0]).abs().max() < 2e-3 * D.max()  # linear interpolation error only
    assert (D.max(0).values - D.min(0).values).max() > 0.3
