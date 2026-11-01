"""Physics checks for the chemistry problems: Hückel MO theory, the Oregonator and the Chapman mechanism."""

import math

import torch

from problems.chemistry.belousov_zhabotinsky import equation as bz_eq
from problems.chemistry.belousov_zhabotinsky import solve as bz_solve
from problems.chemistry.belousov_zhabotinsky import tensors as bz_t
from problems.chemistry.chapman_ozone import equation as ch_eq
from problems.chemistry.chapman_ozone import solve as ch_solve
from problems.chemistry.chapman_ozone import tensors as ch_t
from problems.chemistry.huckel_mo_theory import equation as hk_eq
from problems.chemistry.huckel_mo_theory import solve as hk_solve
from problems.chemistry.huckel_mo_theory import tensors as hk_t

CPU = torch.device("cpu")


# --- Hückel ---------------------------------------------------------------------------------------------------


def _huckel():
    names, xys, A, pad, n_el = hk_t.batch(CPU)
    return names, xys, hk_solve.solve(A, pad, n_el, hk_t.ALPHA, hk_t.BETA)


def test_benzene_eigenvalues_and_delocalisation():
    alpha, beta = 0.3, -1.7  # any α and β < 0
    names, xys, A, pad, n_el = hk_t.batch(CPU)
    res = hk_solve.solve(A, pad, n_el, alpha, beta)
    b = names.index("benzene")
    E = res["E"][b, :6]
    expected = torch.tensor(
        [alpha + 2 * beta, alpha + beta, alpha + beta, alpha - beta, alpha - beta, alpha - 2 * beta]
    )
    assert torch.allclose(E, expected.double(), atol=1e-10)
    assert math.isclose(res["E_pi"][b].item(), 6 * alpha + 8 * beta, abs_tol=1e-10)
    assert math.isclose(res["deloc"][b].item(), 2 * beta, abs_tol=1e-10)
    # cyclobutadiene: no delocalisation energy and a zero gap (half-filled degenerate pair)
    c = names.index("cyclobutadiene")
    assert abs(res["deloc"][c].item()) < 1e-10
    assert abs(res["gap"][c].item()) < 1e-10
    # padding atoms never take electrons
    assert res["n"].sum(-1).tolist() == [float(n) for n in n_el]


def test_butadiene_bond_orders_and_charges():
    names, xys, res = _huckel()
    b = names.index("butadiene")
    P = res["P"][b]
    bonds = [P[0, 1], P[1, 2], P[2, 3]]
    assert all(0 < p < 1 for p in bonds)
    assert math.isclose(bonds[0].item(), 2 / math.sqrt(5), rel_tol=1e-9)  # 0.894
    assert math.isclose(bonds[1].item(), 1 / math.sqrt(5), rel_tol=1e-9)  # 0.447
    assert math.isclose(bonds[2].item(), bonds[0].item(), rel_tol=1e-9)
    assert math.isclose(res["P"][names.index("benzene"), 0, 1].item(), 2 / 3, rel_tol=1e-9)
    # alternant hydrocarbons: every π charge vanishes
    for k, name in enumerate(names):
        assert res["q"][k, : len(xys[k])].abs().max() < 1e-10, name


def test_cyclic_eigenvalues_and_huckel_rule():
    for N in range(3, 15):
        E_num = torch.linalg.eigvalsh(-hk_t.ring_adjacency(N))  # α = 0, β = -1
        E = hk_eq.cyclic_levels(N, 0.0, -1.0)
        assert torch.allclose(E_num, E, atol=1e-12)
        # neutral rings with N π electrons: closed shell exactly for 4n + 2
        if N % 2 == 0:
            assert hk_eq.is_closed_shell(E, N) == (N % 4 == 2), N
        # 4n + 2 electrons always close a shell (one lowest level, then degenerate pairs): the aromatic ions
        else:
            assert hk_eq.is_closed_shell(E, hk_t.ring_electrons(N)), N
            assert hk_t.ring_electrons(N) % 4 == 2

    # benzene is stabilised against hexatriene, cyclobutadiene destabilised against butadiene
    def ring_minus_chain(N):
        ring, chain = hk_eq.cyclic_levels(N, 0.0, -1.0), hk_eq.chain_levels(N, 0.0, -1.0)
        e = [hk_eq.pi_energy(L[None], hk_eq.occupations(L[None], [N]))[0] for L in (ring, chain)]
        return (e[0] - e[1]).item()

    assert ring_minus_chain(6) < -1.0
    assert ring_minus_chain(4) > 0.4


# --- Oregonator -----------------------------------------------------------------------------------------------


def test_oregonator_steady_state():
    f = torch.linspace(0.2, 3.0, 50, dtype=torch.float64)
    xs = bz_eq.steady_state(f, bz_t.Q)
    dx, dz = bz_eq.rhs(xs, xs, f, 0.04, bz_t.Q)
    assert dx.abs().max() < 1e-9 and dz.abs().max() == 0
    assert (xs > bz_t.Q).all()
    # production − loss form is the same right-hand side
    x, z = torch.rand(20, dtype=torch.float64), torch.rand(20, dtype=torch.float64)
    P, L = bz_eq.production_loss(x, z, f[:20], bz_t.Q)
    assert torch.allclose((P - L * x) / 0.04, bz_eq.rhs(x, z, f[:20], 0.04, bz_t.Q)[0])


def test_oregonator_hopf_onset():
    eps = 0.04
    f_fine = torch.linspace(0.05, 4.0, 4000, dtype=torch.float64)
    hopf = bz_solve.hopf_points(f_fine, eps, bz_t.Q)
    assert len(hopf) == 2
    assert 0.45 < hopf[0] < 0.6 and 2.2 < hopf[1] < 2.45  # near 1/2 and 1 + √2 for small q
    # trace really vanishes there
    xs = bz_eq.steady_state(hopf, bz_t.Q)
    assert bz_eq.trace(bz_eq.jacobian(xs, xs, hopf, eps, bz_t.Q)).abs().max() < 1e-2
    # simulation: oscillation just inside the window, decay to the steady state just outside
    f = torch.tensor([hopf[0] - 0.1, 1.0, hopf[1] - 0.1, hopf[1] + 0.2], dtype=torch.float64)
    x0, z0 = bz_t.initial_state(f)
    t, x, z = bz_solve.integrate(x0, z0, f, eps, bz_t.Q, bz_t.DT, 12000)
    lo, hi = bz_solve.amplitude(x)
    ratio = hi / lo
    assert ratio[0] < 1.05 and ratio[3] < 1.05
    assert ratio[1] > 100 and ratio[2] > 1.5
    assert torch.allclose(x[-1, [0, 3]], bz_eq.steady_state(f[[0, 3]], bz_t.Q), rtol=1e-2)
    assert (x > 0).all()
    period = bz_solve.period(t, x[:, 1])
    assert period is not None and 4.5 < period < 7.0


# --- Chapman --------------------------------------------------------------------------------------------------


def test_chapman_stoichiometry_conserves_oxygen():
    assert torch.equal(ch_eq.ATOMS @ ch_eq.S, torch.zeros(4, dtype=torch.float64))
    # and dc/dt conserves atoms for any state
    c = torch.rand(5, 3, dtype=torch.float64) * torch.tensor([1e8, 1e17, 1e12], dtype=torch.float64)
    co = [torch.full((5,), v, dtype=torch.float64) for v in (1e-11, 1e-3, 1e-33, 1e-15, 1e17)]
    rate = ch_eq.dcdt(c, *co) @ ch_eq.ATOMS
    assert (rate.abs() / (ch_eq.rates(c, *co).abs().sum(-1) * 3)).max() < 1e-14
    # Jacobian matches finite differences
    J = ch_eq.jacobian(c, *co)
    h = 1e-6 * c
    for s in range(3):
        cp = c.clone()
        cp[:, s] += h[:, s]
        fd = (ch_eq.dcdt(cp, *co) - ch_eq.dcdt(c, *co)) / h[:, s : s + 1]
        assert torch.allclose(J[:, :, s], fd, rtol=1e-4, atol=1e-9 * fd.abs().max())


def test_chapman_steady_state_matches_analytic():
    z = torch.tensor([25.0, 30.0, 35.0, 40.0, 45.0, 50.0], dtype=torch.float64)
    times = ch_t.time_grid(1e11, 200)
    c, (j1, j3, k2, k4, M) = ch_solve.solve(z, times, 1.0)
    atoms = c @ ch_eq.ATOMS
    assert ((atoms - atoms[0]) / atoms[0]).abs().max() < 1e-10
    o, O3 = ch_eq.steady_state(j1, j3, k2, k4, M, c[-1, :, 1])
    assert torch.allclose(c[-1, :, 2], O3, rtol=1e-3)
    assert torch.allclose(c[-1, :, 0], o, rtol=1e-3)
    # the textbook square-root formula is close where k4 [O] << j3
    approx = ch_eq.chapman_approx(j1, j3, k2, k4, M, c[-1, :, 1])
    assert torch.allclose(approx, O3, rtol=5e-3)
    # time scales: O ⇄ O3 cycling is far faster than odd-oxygen relaxation
    fast, slow = ch_solve.lifetimes(c[-1], j1, j3, k2, k4, M)
    assert (slow / fast > 1e3).all()
    # the layer: the maximum lies inside the range, not at an edge
    peak = int(c[-1, :, 2].argmax())
    assert 0 < peak < len(z) - 1
