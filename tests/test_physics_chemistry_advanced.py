"""Checks for the advanced chemistry problems: Gibbs minimisation, the molecular GNN, polymerisation kinetics."""

import math

import pytest
import torch

from problems.chemistry.gibbs_minimisation import equation as gb_eq
from problems.chemistry.gibbs_minimisation import solve as gb_solve
from problems.chemistry.gibbs_minimisation import tensors as gb_t
from problems.chemistry.molecular_gnn import equation as gnn_eq
from problems.chemistry.molecular_gnn import solve as gnn_solve
from problems.chemistry.molecular_gnn import tensors as gnn_t
from problems.chemistry.polymerisation_mwd import equation as pm_eq
from problems.chemistry.polymerisation_mwd import solve as pm_solve
from problems.chemistry.polymerisation_mwd import tensors as pm_t

CPU = torch.device("cpu")
F64 = torch.float64

# --- Gibbs energy minimisation ---


def _sweep(temps, phi=1.0, P=1.0):
    A, th = gb_t.element_matrix(CPU), gb_t.thermo(CPU)
    T = torch.tensor(temps, dtype=F64)
    b = gb_t.feed(torch.tensor([phi], dtype=F64))[0]
    out = gb_solve.temperature_sweep(T, A, b, P, th, n_iter=80)
    return A, th, T, b, out


def test_element_balance_and_lagrange_conditions_hold_at_the_optimum():
    A, th, T, b, out = _sweep([1200.0, 1800.0, 2400.0, 3000.0, 3500.0])
    g = gb_eq.standard_potential(T, th)
    balance, kkt = gb_solve.residuals(A, b.expand(len(T), -1), out["n"], out["pi"], g, 1.0)
    assert balance < 1e-9  # A n = b
    assert kkt < 1e-8  # μ_j / RT = π_i A_ij
    assert (out["n"] > 0).all()


def test_gibbs_energy_is_lower_than_at_nearby_feasible_points():
    A, th, T, b, out = _sweep([2500.0])
    n = out["n"]  # (1, S), every species well above zero at 2500 K
    g = gb_eq.standard_potential(T, th)
    G0 = gb_eq.gibbs(n, g, 1.0).item()
    null = torch.linalg.svd(A).Vh[A.shape[0] :]  # (S - E, S): directions that keep A n = b
    gen = torch.Generator().manual_seed(0)
    for _ in range(25):
        d = torch.randn(null.shape[0], generator=gen, dtype=F64) @ null  # (S,)
        for frac in (0.3, 0.03):
            step = frac * (n[0] / d.abs()).min()  # keeps every n_j > 0
            n_new = n + step * d
            assert torch.allclose(gb_eq.element_totals(A, n_new), b[None], atol=1e-10)  # still feasible
            assert gb_eq.gibbs(n_new, g, 1.0).item() > G0


def test_autograd_gradient_equals_element_potentials():
    A, th, T, b, out = _sweep([2200.0])
    g = gb_eq.standard_potential(T, th)
    n = out["n"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(gb_eq.gibbs(n, g, 1.0).sum(), n)
    assert torch.allclose(grad, out["pi"] @ A, atol=1e-8)


def test_products_dissociate_at_high_temperature():
    temps = [1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0]
    names = gb_t.SPECIES
    *_, out = _sweep(temps)
    x = gb_eq.mole_fractions(out["n"])

    def col(s):
        return x[:, names.index(s)]

    assert col("CO")[0] < 1e-5 and col("H")[0] < 1e-7  # cold: essentially CO2 + 2 H2O + N2
    assert col("CO2")[0] == pytest.approx(1 / 10.52, rel=1e-3)  # 1 CO2 in 1 + 2 + 7.52 mol
    for s in ("CO2", "H2O"):
        assert (col(s).diff() < 0).all()  # break up as T rises
    for s in ("CO", "H2", "OH", "H", "O", "NO"):
        assert (col(s).diff() > 0).all()  # dissociation products grow
    assert col("CO")[-1] > col("CO2")[-1]  # by 3200 K, CO outnumbers CO2


def test_pressure_suppresses_dissociation():
    *_, low = _sweep([2600.0], P=1.0)
    *_, high = _sweep([2600.0], P=20.0)
    j = gb_t.SPECIES.index("CO")
    assert gb_eq.mole_fractions(high["n"])[0, j] < gb_eq.mole_fractions(low["n"])[0, j]  # Le Chatelier


def test_adiabatic_flame_temperature_is_lowered_by_dissociation_and_peaks_slightly_rich():
    A, th = gb_t.element_matrix(CPU), gb_t.thermo(CPU)
    phis = torch.tensor([0.8, 0.9, 1.0, 1.05, 1.1, 1.2], dtype=F64)
    b, H = gb_t.feed(phis), gb_t.reactant_enthalpy(phis)
    T_ad, n_ad = gb_solve.flame_temperature(A, b, H, 1.0, th)
    assert torch.allclose(gb_eq.enthalpy(n_ad, T_ad, th), H, atol=1e-3)  # energy balance closes
    cold = gb_solve.equilibrium(gb_eq.standard_potential(torch.full_like(phis, 1000.0), th), A, b, 1.0)["n"]
    T_frozen = gb_solve.frozen_flame_temperature(cold, H, th)
    assert (T_ad < T_frozen).all()
    assert 2150 < T_ad[2].item() < 2350  # accurate thermochemistry: about 2225 K at φ = 1
    assert phis[T_ad.argmax()].item() in (1.0, 1.05, 1.1)


# --- molecular GNN ---


def _molecule(edges, n, n_pad=8):
    A = torch.zeros(1, n_pad, n_pad)
    for i, j in edges:
        A[0, i, j] = A[0, j, i] = 1.0
    mask = (A.sum(-1) > 0).float()
    return {"A": A, "mask": mask, "x": gnn_t.features(A, mask)}


def test_huckel_energy_matches_textbook_values():
    benzene = _molecule([(i, (i + 1) % 6) for i in range(6)], 6)
    butadiene = _molecule([(0, 1), (1, 2), (2, 3)], 4)
    allyl = _molecule([(0, 1), (1, 2)], 3)  # odd electron in the non-bonding orbital (λ = 0)
    assert gnn_eq.huckel_energy(benzene["A"], benzene["mask"]).item() == pytest.approx(8.0, abs=1e-5)
    assert gnn_eq.huckel_energy(butadiene["A"], butadiene["mask"]).item() == pytest.approx(
        2 * (2 * math.cos(math.pi / 5) + 2 * math.cos(2 * math.pi / 5)), abs=1e-5
    )
    assert gnn_eq.huckel_energy(allyl["A"], allyl["mask"]).item() == pytest.approx(2 * math.sqrt(2), abs=1e-5)


def test_random_skeletons_are_valid_carbon_graphs():
    data = gnn_t.dataset(200, (4, 12), 16, 0, CPU)
    A, mask = data["A"], data["mask"]
    assert torch.equal(A, A.transpose(1, 2))
    assert (A.sum(-1) <= 3).all()  # sp² carbon: at most three C neighbours
    assert ((A.sum(-1) > 0) == (mask > 0)).all()
    reach = torch.matrix_power(torch.eye(16) + A, 16) > 0  # connected: every real atom reaches every other
    real = mask[:, :, None] * mask[:, None, :] > 0
    assert reach[real].all()


def test_gnn_is_permutation_invariant_and_the_mlp_is_not():
    data = gnn_t.dataset(50, (5, 12), 16, 3, CPU)
    w_gnn, w_mlp = gnn_t.init_gnn(16, 3, 0, CPU), gnn_t.init_mlp(16, 32, 0, CPU)
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        moved = gnn_t.permute(data, perm)
        assert torch.allclose(gnn_eq.gnn(moved, w_gnn), gnn_eq.gnn(data, w_gnn), atol=1e-4)
        assert not torch.allclose(gnn_eq.mlp(moved, w_mlp), gnn_eq.mlp(data, w_mlp), atol=1e-2)
    assert torch.allclose(
        gnn_eq.huckel_energy(moved["A"], moved["mask"]), gnn_eq.huckel_energy(data["A"], data["mask"])
    )


def test_gnn_training_loss_falls_below_the_mean_and_bond_counting_baselines():
    train = gnn_t.dataset(600, (4, 10), 16, 0, CPU)
    y = gnn_eq.huckel_energy(train["A"], train["mask"])
    w = gnn_t.init_gnn(24, 3, 0, CPU)
    hist = gnn_solve.train(gnn_eq.gnn, w, {**train, "y": y}, {}, 40, 0.005, 64, 0, "test")
    loss = hist["loss"]
    count_pred, _ = gnn_eq.count_baseline(train, y)
    assert loss[-1] < 0.1 * loss[1]  # it trains
    assert loss[-1] < 0.02 * y.var()  # far better than predicting the mean
    assert loss[-1] < gnn_eq.mse(count_pred, y)  # and better than a linear fit in atoms and bonds


# --- polymerisation ---


@pytest.mark.parametrize("p", [0.5, 0.9, 0.98])
def test_flory_distribution_is_normalised_with_carothers_mean_and_dispersity_one_plus_p(p):
    x = pm_t.lengths(5000, CPU)
    n, w = pm_eq.flory_number(x, p), pm_eq.flory_weight(x, p)
    assert n.sum().item() == pytest.approx(1.0, abs=1e-9)
    assert w.sum().item() == pytest.approx(1.0, abs=1e-9)
    xn = (x * n).sum().item()
    assert xn == pytest.approx(pm_eq.carothers(p), rel=1e-9)
    xn_, xw, disp = pm_eq.averages(n, x)
    assert disp.item() == pytest.approx(1 + p, rel=1e-9)
    assert xw.item() == pytest.approx((1 + p) / (1 - p), rel=1e-9)


def test_population_balance_reproduces_flory_and_poisson():
    L = 400
    idx, mask = pm_t.shift_index(L, CPU)
    x = pm_t.lengths(L, CPU)
    s, traj = pm_solve.population_balance(pm_t.initial_state(L, CPU), idx, mask, 9.0, 0.1)  # X_n = 10, p = 0.9
    P_step, P_living = traj[-1]
    p = s[-1].item() / (1 + s[-1].item())
    assert p == pytest.approx(0.9)
    assert torch.allclose(P_step, (1 - p) ** 2 * p ** (x - 1), atol=1e-6)
    assert torch.allclose(P_living, pm_eq.poisson(x, s[-1]), atol=1e-6)
    m = pm_eq.moments(traj, x)  # (T, 2, 3)
    assert torch.allclose(m[:, 0, 1], torch.ones_like(m[:, 0, 1]), atol=1e-8)  # monomer units conserved
    assert torch.allclose(m[:, 1, 0], torch.ones_like(m[:, 1, 0]), atol=1e-8)  # living chains conserved
    _, _, disp = pm_eq.averages(traj[-1], x)
    assert disp[0].item() == pytest.approx(1 + p, abs=1e-4)


def test_kinetic_monte_carlo_matches_flory():
    n0, p = 50_000, 0.9
    counts = pm_solve.kinetic_monte_carlo(n0, [int(p * n0)], 400, seed=0)[0]  # chains per initial monomer
    x = pm_t.lengths(400, CPU)
    xn, _, disp = pm_eq.averages(counts, x)
    assert xn.item() == pytest.approx(1 / (1 - p), rel=1e-9)  # exact: every event removes one chain
    assert disp.item() == pytest.approx(1 + p, abs=0.05)
    cumulative = counts.cumsum(0) / counts.sum()  # fraction of chains no longer than x
    for k in (1, 3, 10, 30):
        assert cumulative[k - 1].item() == pytest.approx(1 - p**k, abs=0.03)  # Flory: Σ_{x≤k} (1-p) p^(x-1)


def test_living_polymerisation_gives_poisson_dispersity():
    L = 300
    idx, mask = pm_t.shift_index(L, CPU)
    x = pm_t.lengths(L, CPU)
    s, traj = pm_solve.population_balance(pm_t.initial_state(L, CPU), idx, mask, 49.0, 0.1)
    xn, _, disp = pm_eq.averages(traj[:, 1], x)  # living row
    nu = s
    assert torch.allclose(xn, 1 + nu, atol=1e-6)
    assert torch.allclose(disp, 1 + nu / (1 + nu) ** 2, atol=1e-6)
    assert disp[-1].item() == pytest.approx(1 + 1 / xn[-1].item(), abs=1.01 / xn[-1].item() ** 2)
    assert disp[-1].item() < 1.03
