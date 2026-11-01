"""Physics checks for physical chemistry: particle-in-a-box dyes, the HCl rovibrational band, Nernst/Debye–Hückel."""

import math

import torch

from problems.physchem.hcl_rovibrational import equation as hcl_eq
from problems.physchem.hcl_rovibrational import solve as hcl_solve
from problems.physchem.hcl_rovibrational import tensors as hcl_t
from problems.physchem.nernst_debye_huckel import equation as nd_eq
from problems.physchem.nernst_debye_huckel import solve as nd_solve
from problems.physchem.nernst_debye_huckel import tensors as nd_t
from problems.physchem.particle_box_dyes import equation as box_eq
from problems.physchem.particle_box_dyes import solve as box_solve
from problems.physchem.particle_box_dyes import tensors as box_t

CPU = torch.device("cpu")
F64 = torch.float64


# ---------------------------------------------------------------- particle in a box


def _box(series="cyanine", n_max=4, V0=0.0, M=300):
    names, n_bonds, n_el = box_t.molecules(series, n_max, CPU)
    L = box_t.box_length(n_bonds, 0.139, 1.0).to(F64)
    x, dx = box_t.grids(L, M)
    V = box_eq.alternation_potential(x, V0, 0.139, 1.0)
    E, psi = box_solve.solve(box_eq.hamiltonian(dx, V), dx, int(n_el.max()) // 2 + 2)
    return L, n_el, E, psi, dx


def test_numerical_box_levels_scale_as_n_squared():
    L, _, E, psi, dx = _box()
    n = torch.arange(1, E.shape[1] + 1, dtype=F64)
    assert torch.allclose(E / E[:, :1], (n**2).expand_as(E), rtol=2e-3)
    exact = box_eq.energy_levels(n, L)
    assert ((E - exact) / exact).abs().max() < 2e-3
    norm = (psi**2).sum(1) * dx[:, None]  # Σ ψ² dx = 1 for every state
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-10)


def test_box_error_shrinks_with_grid():
    errs = []
    for M in (100, 200):
        L, _, E, _, _ = _box(n_max=1, M=M)
        exact = box_eq.energy_levels(torch.arange(1, E.shape[1] + 1, dtype=F64), L)
        errs.append(((E - exact) / exact).abs().max().item())
    assert 3.5 < errs[0] / errs[1] < 4.5  # second-order finite differences


def test_gap_formula_matches_levels_and_lambda_grows():
    L, n_el, E, _, _ = _box()
    n = torch.arange(1, E.shape[1] + 1, dtype=F64)
    exact = box_eq.energy_levels(n, L)
    homo = (n_el // 2 - 1)[:, None]
    gap_levels = (exact.gather(1, homo + 1) - exact.gather(1, homo))[:, 0]
    gap = box_eq.homo_lumo_gap(n_el, L)
    assert torch.allclose(gap, gap_levels, rtol=1e-6)
    lam = box_eq.wavelength(gap)
    assert (lam[1:] > lam[:-1]).all()  # longer chain → redder absorption
    # cyanine with 3 CH groups (k = 1): L = 6 l, N = 6  →  λ = 8 m c (6 l)² / (7 h) ≈ 328 nm
    assert abs(lam[1].item() - 327.6) < 1.0


def test_alternation_opens_a_gap():
    _, n_el, E0, _, _ = _box("polyene", 6, V0=0.0)
    _, _, E1, _, _ = _box("polyene", 6, V0=1.5)
    homo = (n_el // 2 - 1)[:, None]
    g0 = (E0.gather(1, homo + 1) - E0.gather(1, homo))[:, 0]
    g1 = (E1.gather(1, homo + 1) - E1.gather(1, homo))[:, 0]
    assert (g1 > g0).all()
    assert g1[-1] > 0.6 * 1.5  # the gap stays of order V0 for long chains, while the plain box gap → 0


# ---------------------------------------------------------------- HCl rovibrational band


def _hcl(J_max=15):
    mu = hcl_t.reduced_masses(CPU).to(F64)
    c = hcl_t.constants(mu)
    J = hcl_t.rotational_levels(J_max, CPU).to(F64)
    R, P = hcl_eq.branches(J, c)
    return c, J, R, P


def test_line_positions_match_formulas():
    c, J, R, P = _hcl()
    we, wexe, Be, ae, D = (c[k][0].item() for k in ("we", "wexe", "Be", "ae", "De"))
    nu0 = we - 2 * wexe
    B0, B1 = Be - 0.5 * ae, Be - 1.5 * ae

    def F(B, j):
        return B * j * (j + 1) - D * j**2 * (j + 1) ** 2

    for j in (0, 3, 9):
        assert abs(R[0, j].item() - (nu0 + F(B1, j + 1) - F(B0, j))) < 1e-9
    for j in (1, 4, 10):
        assert abs(P[0, j].item() - (nu0 + F(B1, j - 1) - F(B0, j))) < 1e-9
    # approximate magnitudes for H³⁵Cl
    assert abs(nu0 - 2885.3) < 0.5
    assert 2904 < R[0, 0].item() < 2907 and 2863 < P[0, 1].item() < 2866


def test_branch_spacings_and_band_gap():
    c, J, R, P = _hcl()
    dR = R[0, 1:] - R[0, :-1]
    dP = P[0, 1:-1] - P[0, 2:]  # P(J) − P(J+1), J ≥ 1
    assert (dR[1:] < dR[:-1]).all()  # R lines converge (B₁ < B₀)
    assert (dP[1:] > dP[:-1]).all()  # P lines spread apart
    B0 = hcl_eq.rot_constant(0, c["Be"], c["ae"])[0]
    B1 = hcl_eq.rot_constant(1, c["Be"], c["ae"])[0]
    gap = R[0, 0] - P[0, 1]
    assert abs(gap - 2 * (B0 + B1)).item() < 0.01  # no Q branch: the gap around ν₀ is ≈ 4B
    assert gap > 1.8 * dR[0]


def test_isotope_shift():
    c, J, R, P = _hcl()
    shift0 = (hcl_eq.band_origin(c["we"], c["wexe"])[0] - hcl_eq.band_origin(c["we"], c["wexe"])[1]).item()
    assert 1.9 < shift0 < 2.3  # H³⁷Cl lies about 2 cm⁻¹ lower
    assert (R[0] > R[1]).all() and (P[0, 1:] > P[1, 1:]).all()
    assert torch.allclose(hcl_eq.morse_depth(c["we"], c["wexe"])[1], hcl_eq.morse_depth(c["we"], c["wexe"])[0])


def test_intensities_boltzmann_and_P0_zero():
    c, J, R, P = _hcl(25)
    B0 = hcl_eq.rot_constant(0, c["Be"], c["ae"])
    T = torch.tensor([100.0, 300.0, 1000.0], dtype=F64)
    p = hcl_eq.populations(J, B0, c["De"], T)
    assert torch.allclose(p.sum(-1), torch.ones(3, 2, dtype=F64))
    peaks = p[:, 0].argmax(-1)
    assert (peaks[1:] >= peaks[:-1]).all() and peaks[-1] > peaks[0]  # hotter → higher J
    J_est = math.sqrt(300.0 / (2 * hcl_eq.HC_K * B0[0].item())) - 0.5
    assert abs(peaks[1].item() - J_est) < 1.0
    I_R, I_P = hcl_eq.intensities(J, p, hcl_t.abundances(CPU).to(F64))
    assert (I_P[..., 0] == 0).all()
    ratio = (I_R[:, 0] / I_R[:, 1])[:, 2]
    assert ((ratio - 0.758 / 0.242).abs() < 0.05).all()


def test_combination_difference_fit_recovers_constants():
    c, J, R, P = _hcl()
    fit = hcl_solve.fit_constants(R, P, J)
    nu0 = hcl_eq.band_origin(c["we"], c["wexe"])
    assert torch.allclose(fit["Be"], c["Be"], atol=1e-8)
    assert torch.allclose(fit["ae"], c["ae"], atol=1e-8)
    assert torch.allclose(fit["D"], c["De"], atol=1e-10)
    assert torch.allclose(fit["nu0"], nu0, atol=1e-7)
    # with 0.02 cm⁻¹ noise the constants are still close
    g = torch.Generator().manual_seed(1)
    noisy = hcl_solve.fit_constants(
        R + 0.02 * torch.randn(R.shape, generator=g, dtype=F64),
        P + 0.02 * torch.randn(P.shape, generator=g, dtype=F64),
        J,
    )
    assert (noisy["Be"] - c["Be"]).abs().max() < 0.01
    assert (noisy["nu0"] - nu0).abs().max() < 0.15


# ---------------------------------------------------------------- Nernst and Debye–Hückel


def test_nernst_slope_per_decade():
    T = nd_t.temperatures(CPU).to(F64)
    lq = nd_t.log_ratios(CPU).to(F64)
    E = nd_eq.nernst(1.10, T, lq, 2)
    slope = -(E[:, 1:] - E[:, :-1]) / (lq[1:] - lq[:-1])
    expected = 2.302585 * nd_t.R_GAS * T / (2 * nd_t.F)
    assert torch.allclose(slope, expected[:, None].expand_as(slope), rtol=1e-6)
    s25 = nd_eq.nernst_slope(torch.tensor(298.15, dtype=F64), 2).item()
    assert abs(s25 - 0.02958) < 1e-4  # 29.6 mV per decade for n = 2
    assert torch.allclose(E[:, 40], torch.full((len(T),), 1.10, dtype=F64))  # Q = 1 → E = E°


def test_debye_huckel_constants_and_limiting_law():
    T = torch.tensor(298.15, dtype=F64)
    A, B = nd_eq.debye_A(T).item(), nd_eq.debye_B(T).item()
    assert abs(A - 0.509) < 0.006
    assert abs(1 / B - 0.304) < 0.004  # κ⁻¹ = 0.304 nm / √I
    ion = torch.logspace(-7, -4, 10, dtype=F64)  # ionic strength
    lim = nd_eq.log10_gamma_limiting(2, ion, torch.tensor(A, dtype=F64))
    ext = nd_eq.log10_gamma_extended(2, ion, torch.tensor(A, dtype=F64), torch.tensor(B, dtype=F64), 0.5)
    rel = ((ext - lim) / lim).abs()
    assert rel.max() < 0.02 and (rel[1:] > rel[:-1]).all()  # extended → limiting as I → 0
    # log γ ∝ −√I in the limit: tenfold I gives a √10 larger |log γ|
    assert torch.allclose(lim[1:] / lim[:-1], (ion[1:] / ion[:-1]).sqrt())


def test_poisson_boltzmann_matches_yukawa_and_debye_huckel():
    T = torch.tensor(298.15, dtype=F64)
    m = torch.logspace(-4, -1, 6, dtype=F64)
    ion = nd_t.ionic_strength(m)
    kap = nd_eq.kappa(ion, T)
    a = 0.5
    r, h = nd_t.radial_grids(kap, a, 400)
    phi = nd_solve.solve_pb(r, h, kap, a)
    exact = nd_eq.yukawa(r, kap, a)
    assert ((phi - exact).abs() / exact[:, :1]).max() < 3e-4
    # φ → bare Coulomb 1/r well inside the Debye length at the lowest concentration
    assert abs(phi[0, 0].item() * a - 1) < 0.05
    # the cloud's potential at the ion gives the extended Debye–Hückel γ
    lg = nd_eq.ln_gamma_from_potential(phi[:, 0], a, 2, nd_eq.bjerrum_length(T)) / math.log(10)
    ext = nd_eq.log10_gamma_extended(2, ion, nd_eq.debye_A(T), nd_eq.debye_B(T), a)
    assert (lg - ext).abs().max() < 1e-3


def test_activity_correction_on_emf():
    T = torch.tensor(298.15, dtype=F64)
    A, B = nd_eq.debye_A(T), nd_eq.debye_B(T)
    m = torch.tensor([1e-3, 1e-2, 1e-1], dtype=F64)
    lg = nd_eq.log10_gamma_extended(2, nd_t.ionic_strength(m), A, B, 0.5)
    # equal molalities in both half-cells: the γ cancel and E = E°
    E_eq = nd_eq.emf(1.10, T, m, m, lg, lg, 2)
    assert torch.allclose(E_eq, torch.full_like(m, 1.10))
    # dilute Zn²⁺ against concentrated Cu²⁺: γ_Zn > γ_Cu raises Q and lowers the EMF below the ideal value
    E_ideal = nd_eq.emf(1.10, T, m[:1], m[2:], 0.0, 0.0, 2)
    E_real = nd_eq.emf(1.10, T, m[:1], m[2:], lg[:1], lg[2:], 2)
    assert (E_real < E_ideal).all()
