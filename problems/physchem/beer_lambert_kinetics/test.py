"""Physics checks, written first (TDD): what any correct Beer–Lambert / A → B → C code must satisfy.

Run with `pytest problems/physchem/beer_lambert_kinetics/test.py`. Each test is a physical law, not an implementation
detail: absorbance is linear in concentration and path length, A + B + C is conserved, and isosbestic points are fixed.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _setup(k1=0.02, k2=0.003, sigma=0.0):
    lam = tensors.wavelengths(CPU)
    E = tensors.absorptivities(lam)
    c0 = tensors.initial_concentrations(50.0, CPU)
    t = tensors.sample_times(1200.0, CPU)
    noise = tensors.noise((len(t), len(lam)), sigma, CPU)
    c, D = solve.simulate(E, c0, t, k1, k2, tensors.PATH_LENGTH, noise)
    return lam, E, c0, t, c, D


def test_beer_lambert_linearity():
    lam = tensors.wavelengths(CPU)
    E = tensors.absorptivities(lam)
    gen = torch.Generator().manual_seed(2)
    c1, c2 = torch.rand(5, 3, generator=gen, dtype=F64) * 1e-4, torch.rand(5, 3, generator=gen, dtype=F64) * 1e-4
    D = lambda c, length=1.0: equation.absorbance(E, c, length)  # noqa: E731
    assert torch.allclose(D(2.5 * c1), 2.5 * D(c1), rtol=1e-12)  # proportional to concentration
    assert torch.allclose(D(c1, 2.0), 2.0 * D(c1), rtol=1e-12)  # proportional to path length
    assert torch.allclose(D(c1 + c2), D(c1) + D(c2), rtol=1e-12)  # absorbances of a mixture add
    # pure A at its band maximum: D = ε l c with ε = 12000 L/(mol cm)
    w = int((lam - 350.0).abs().argmin())
    pure = torch.tensor([[1e-5, 0.0, 0.0]], dtype=F64)
    assert math.isclose(D(pure)[0, w].item(), 12000 * 1e-5, rel_tol=1e-12)


def test_kinetics_matches_analytic_and_conserves_mass():
    _, _, c0, t, c, _ = _setup(k1=0.02, k2=0.003)
    exact = equation.analytic(0.02, 0.003, c0[0].item(), t)
    assert (c - exact).abs().max() < 1e-12 * 1e6 * c0[0]  # exact solution of consecutive first-order steps
    assert (c.sum(1) - c0.sum()).abs().max() < 1e-15  # mass balance: A + B + C = const
    t_peak = math.log(0.02 / 0.003) / (0.02 - 0.003)
    assert abs(t[c[:, 1].argmax()].item() - t_peak) <= 10.0  # B peaks where dB/dt = 0 (grid spacing 10 s)


def test_pseudo_inverse_matches_torch_and_recovers_concentrations():
    _, E, c0, _, c, D = _setup()
    E_pinv = equation.pseudo_inverse(E)
    assert torch.allclose(E_pinv, torch.linalg.pinv(E), atol=1e-14)
    assert torch.allclose(E_pinv @ E, torch.eye(3, dtype=F64), atol=1e-10)
    c_hat, residual, _ = solve.unmix(E, D, tensors.PATH_LENGTH)  # noise-free spectra
    assert (c_hat - c).abs().max() < 1e-12 * c0[0] * 1e3
    assert residual.abs().max() < 1e-12


def test_recovery_with_noise_and_mass_balance():
    _, E, c0, _, c, D = _setup(sigma=0.003)
    c_hat, residual, _ = solve.unmix(E, D, tensors.PATH_LENGTH)
    assert (c_hat - c).abs().max() < 0.01 * c0[0]  # within 1 % of [A]0
    assert ((c_hat.sum(1) - c0.sum()).abs() < 0.01 * c0[0]).all()  # recovered A + B + C ≈ const
    assert abs(residual.pow(2).mean().sqrt().item() - 0.003) < 0.0003  # residual ≈ noise level


def test_isosbestic_point_is_exact_for_two_species():
    lam, E, _, _, c, D = _setup(k2=0.0)  # A → B only
    (x,) = equation.crossings(lam, E[:, 0], E[:, 1])
    assert 380 < x < 420
    assert c[:, 2].abs().max() == 0  # no C
    # interpolate D to the crossing wavelength: constant in time, while the band maxima change a lot
    w = int(torch.searchsorted(lam, torch.tensor(x, dtype=F64))) - 1
    f = (x - lam[w]) / (lam[w + 1] - lam[w])
    D_iso = (1 - f) * D[:, w] + f * D[:, w + 1]
    assert (D_iso - D_iso[0]).abs().max() < 2e-3 * D.max()  # isosbestic: ε_A = ε_B and A + B = const
    assert (D.max(0).values - D.min(0).values).max() > 0.3
