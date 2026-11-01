"""Physics checks, written first (TDD): what any correct HCl rovibrational-band code must satisfy.

Run with `pytest problems/physchem/hcl_rovibrational/test.py`. Each test is a physical law, not an implementation
detail: lines follow rotor-oscillator terms, there is no Q branch, populations are Boltzmann, and isotopes shift.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _hcl(J_max=15):
    mu = tensors.reduced_masses(CPU).to(F64)
    c = tensors.constants(mu)
    J = tensors.rotational_levels(J_max, CPU).to(F64)
    R, P = equation.branches(J, c)
    return c, J, R, P


def test_line_positions_match_formulas():
    c, J, R, P = _hcl()
    we, wexe, Be, ae, D = (c[k][0].item() for k in ("we", "wexe", "Be", "ae", "De"))
    nu0 = we - 2 * wexe
    B0, B1 = Be - 0.5 * ae, Be - 1.5 * ae

    def F(B, j):
        return B * j * (j + 1) - D * j**2 * (j + 1) ** 2

    for j in (0, 3, 9):
        assert abs(R[0, j].item() - (nu0 + F(B1, j + 1) - F(B0, j))) < 1e-9  # ΔJ = +1 selection rule
    for j in (1, 4, 10):
        assert abs(P[0, j].item() - (nu0 + F(B1, j - 1) - F(B0, j))) < 1e-9  # ΔJ = −1 selection rule
    # approximate magnitudes for H³⁵Cl
    assert abs(nu0 - 2885.3) < 0.5
    assert 2904 < R[0, 0].item() < 2907 and 2863 < P[0, 1].item() < 2866


def test_branch_spacings_and_band_gap():
    c, J, R, P = _hcl()
    dR = R[0, 1:] - R[0, :-1]
    dP = P[0, 1:-1] - P[0, 2:]  # P(J) − P(J+1), J ≥ 1
    assert (dR[1:] < dR[:-1]).all()  # R lines converge (B₁ < B₀)
    assert (dP[1:] > dP[:-1]).all()  # P lines spread apart
    B0 = equation.rot_constant(0, c["Be"], c["ae"])[0]
    B1 = equation.rot_constant(1, c["Be"], c["ae"])[0]
    gap = R[0, 0] - P[0, 1]
    assert abs(gap - 2 * (B0 + B1)).item() < 0.01  # no Q branch: the gap around ν₀ is ≈ 4B
    assert gap > 1.8 * dR[0]


def test_isotope_shift():
    c, J, R, P = _hcl()
    origin = equation.band_origin(c["we"], c["wexe"])
    shift0 = (origin[0] - origin[1]).item()
    assert 1.9 < shift0 < 2.3  # H³⁷Cl lies about 2 cm⁻¹ lower (ω ∝ 1/√μ)
    assert (R[0] > R[1]).all() and (P[0, 1:] > P[1, 1:]).all()
    depth = equation.morse_depth(c["we"], c["wexe"])
    assert torch.allclose(depth[1], depth[0])  # Born–Oppenheimer: same potential well for both isotopes


def test_intensities_boltzmann_and_P0_zero():
    c, J, R, P = _hcl(25)
    B0 = equation.rot_constant(0, c["Be"], c["ae"])
    T = torch.tensor([100.0, 300.0, 1000.0], dtype=F64)
    p = equation.populations(J, B0, c["De"], T)
    assert torch.allclose(p.sum(-1), torch.ones(3, 2, dtype=F64))  # populations are normalised
    peaks = p[:, 0].argmax(-1)
    assert (peaks[1:] >= peaks[:-1]).all() and peaks[-1] > peaks[0]  # hotter → higher J
    J_est = math.sqrt(300.0 / (2 * equation.HC_K * B0[0].item())) - 0.5
    assert abs(peaks[1].item() - J_est) < 1.0  # Boltzmann × (2J + 1) peaks at J_max = √(kT / 2hcB) − ½
    I_R, I_P = equation.intensities(J, p, tensors.abundances(CPU).to(F64))
    assert (I_P[..., 0] == 0).all()  # there is no J = −1 level, so P(0) does not exist
    ratio = (I_R[:, 0] / I_R[:, 1])[:, 2]
    assert ((ratio - 0.758 / 0.242).abs() < 0.05).all()


def test_combination_difference_fit_recovers_constants():
    c, J, R, P = _hcl()
    fit = solve.fit_constants(R, P, J)
    nu0 = equation.band_origin(c["we"], c["wexe"])
    assert torch.allclose(fit["Be"], c["Be"], atol=1e-8)
    assert torch.allclose(fit["ae"], c["ae"], atol=1e-8)
    assert torch.allclose(fit["D"], c["De"], atol=1e-10)
    assert torch.allclose(fit["nu0"], nu0, atol=1e-7)
    # with 0.02 cm⁻¹ noise the constants are still close
    g = torch.Generator().manual_seed(1)
    noisy = solve.fit_constants(
        R + 0.02 * torch.randn(R.shape, generator=g, dtype=F64),
        P + 0.02 * torch.randn(P.shape, generator=g, dtype=F64),
        J,
    )
    assert (noisy["Be"] - c["Be"]).abs().max() < 0.01
    assert (noisy["nu0"] - nu0).abs().max() < 0.15
