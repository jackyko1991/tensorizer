"""Physics checks, written first (TDD): what any correct Nernst / Debye–Hückel code must satisfy.

Run with `pytest problems/physchem/nernst_debye_huckel/test.py`. Each test is a physical law, not an implementation
detail: the Nernst slope is ln 10 RT/nF, Debye–Hückel reduces to the limiting law, and Poisson–Boltzmann gives Yukawa.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def test_nernst_slope_per_decade():
    T = tensors.temperatures(CPU).to(F64)
    lq = tensors.log_ratios(CPU).to(F64)
    E = equation.nernst(1.10, T, lq, 2)
    slope = -(E[:, 1:] - E[:, :-1]) / (lq[1:] - lq[:-1])
    expected = 2.302585 * tensors.R_GAS * T / (2 * tensors.F)
    assert torch.allclose(slope, expected[:, None].expand_as(slope), rtol=1e-6)  # Nernst: dE/dlog Q = −ln10 RT/nF
    s25 = equation.nernst_slope(torch.tensor(298.15, dtype=F64), 2).item()
    assert abs(s25 - 0.02958) < 1e-4  # 29.6 mV per decade for n = 2
    assert torch.allclose(E[:, 40], torch.full((len(T),), 1.10, dtype=F64))  # Q = 1 → E = E°


def test_debye_huckel_constants_and_limiting_law():
    T = torch.tensor(298.15, dtype=F64)
    A, B = equation.debye_A(T).item(), equation.debye_B(T).item()
    assert abs(A - 0.509) < 0.006
    assert abs(1 / B - 0.304) < 0.004  # κ⁻¹ = 0.304 nm / √I
    ion = torch.logspace(-7, -4, 10, dtype=F64)  # ionic strength
    lim = equation.log10_gamma_limiting(2, ion, torch.tensor(A, dtype=F64))
    ext = equation.log10_gamma_extended(2, ion, torch.tensor(A, dtype=F64), torch.tensor(B, dtype=F64), 0.5)
    rel = ((ext - lim) / lim).abs()
    assert rel.max() < 0.02 and (rel[1:] > rel[:-1]).all()  # extended → limiting as I → 0
    # log γ ∝ −√I in the limit: tenfold I gives a √10 larger |log γ|
    assert torch.allclose(lim[1:] / lim[:-1], (ion[1:] / ion[:-1]).sqrt())


def test_poisson_boltzmann_matches_yukawa_and_debye_huckel():
    T = torch.tensor(298.15, dtype=F64)
    m = torch.logspace(-4, -1, 6, dtype=F64)
    ion = tensors.ionic_strength(m)
    kap = equation.kappa(ion, T)
    a = 0.5
    r, h = tensors.radial_grids(kap, a, 400)
    phi = solve.solve_pb(r, h, kap, a)
    exact = equation.yukawa(r, kap, a)
    assert ((phi - exact).abs() / exact[:, :1]).max() < 3e-4  # linearised Poisson–Boltzmann: screened Coulomb
    # φ → bare Coulomb 1/r well inside the Debye length at the lowest concentration
    assert abs(phi[0, 0].item() * a - 1) < 0.05
    # the cloud's potential at the ion gives the extended Debye–Hückel γ
    lg = equation.ln_gamma_from_potential(phi[:, 0], a, 2, equation.bjerrum_length(T)) / math.log(10)
    ext = equation.log10_gamma_extended(2, ion, equation.debye_A(T), equation.debye_B(T), a)
    assert (lg - ext).abs().max() < 1e-3


def test_activity_correction_on_emf():
    T = torch.tensor(298.15, dtype=F64)
    A, B = equation.debye_A(T), equation.debye_B(T)
    m = torch.tensor([1e-3, 1e-2, 1e-1], dtype=F64)
    lg = equation.log10_gamma_extended(2, tensors.ionic_strength(m), A, B, 0.5)
    # equal molalities in both half-cells: the γ cancel and E = E°
    E_eq = equation.emf(1.10, T, m, m, lg, lg, 2)
    assert torch.allclose(E_eq, torch.full_like(m, 1.10))
    # dilute Zn²⁺ against concentrated Cu²⁺: γ_Zn > γ_Cu raises Q and lowers the EMF below the ideal value
    E_ideal = equation.emf(1.10, T, m[:1], m[2:], 0.0, 0.0, 2)
    E_real = equation.emf(1.10, T, m[:1], m[2:], lg[:1], lg[2:], 2)
    assert (E_real < E_ideal).all()
