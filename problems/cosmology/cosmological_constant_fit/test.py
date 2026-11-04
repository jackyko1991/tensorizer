"""Physics checks, written first (TDD): what any correct supernova-cosmology fit must satisfy.

Run with `pytest problems/cosmology/cosmological_constant_fit/test.py`. The distance–redshift integral must match
the closed forms of three textbook universes, a closed universe must agree with a slow independent quadrature,
a fit to noise-free synthetic data must give back the parameters that made it, and the real Pantheon+ supernovae
must need a positive cosmological constant (an accelerating universe) at high significance.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
C = equation.C_KMS
Z = torch.tensor([0.01, 0.1, 0.5, 1.0, 1.5, 2.2], dtype=torch.float64)


def d_lum(H0, Om, OL):
    nodes = tensors.redshift_nodes(2.3, 2000, CPU)
    return equation.luminosity_distance(Z, nodes, torch.tensor(H0), torch.tensor(Om), torch.tensor(OL))


def test_einstein_de_sitter_closed_form():
    # Ω_m = 1, Ω_Λ = 0:  d_L = (2c/H0) (1 + z) (1 − 1/√(1+z))
    exact = 2 * C / 70 * (1 + Z) * (1 - 1 / torch.sqrt(1 + Z))
    assert torch.allclose(d_lum(70.0, 1.0, 0.0), exact, rtol=1e-6)


def test_empty_universe_closed_form():
    # Ω_m = Ω_Λ = 0 (Milne, Ω_k = 1):  d_L = (c/H0) z (1 + z/2)
    exact = C / 70 * Z * (1 + Z / 2)
    assert torch.allclose(d_lum(70.0, 0.0, 0.0), exact, rtol=1e-6)


def test_de_sitter_closed_form():
    # flat, Ω_Λ = 1:  d_L = (c/H0) z (1 + z)
    exact = C / 70 * Z * (1 + Z)
    assert torch.allclose(d_lum(70.0, 0.0, 1.0), exact, rtol=1e-6)


def test_closed_universe_matches_slow_quadrature():
    # Ω_k = 1 − 1.5 − 0.3 < 0: the sin branch, against Simpson's rule on each redshift separately
    Om, OL = 1.5, 0.3
    Ok = 1 - Om - OL
    ref = []
    for z in Z.tolist():
        x = torch.linspace(0, z, 20001, dtype=torch.float64)
        f = 1 / torch.sqrt(Om * (1 + x) ** 3 + Ok * (1 + x) ** 2 + OL)
        h = z / 20000
        chi = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
        ref.append((1 + z) * C / 70 * math.sin(math.sqrt(-Ok) * chi) / math.sqrt(-Ok))
    assert torch.allclose(d_lum(70.0, Om, OL), torch.tensor(ref, dtype=torch.float64), rtol=1e-6)


def test_loader_reads_bundled_and_custom_csv(tmp_path):
    z, mu, sigma = tensors.load_distance_moduli(device=CPU)
    assert len(z) == 1466 and z.dtype == torch.float64
    assert (sigma > 0).all() and (z > 0.01).all() and (z.diff() >= 0).all()
    assert 30 < mu.min() < mu.max() < 47  # distance moduli of SNe from 40 Mpc to z ≈ 2.3
    # an uploaded file: different column order, extra columns, comments, other spellings of the headers
    path = tmp_path / "my_sne.csv"
    path.write_text("# my survey\nSIGMA_MU,MU,Z,host\n0.2,40.0,0.1,a\n0.3,43.0,0.5,b\n", encoding="utf-8")
    z2, mu2, s2 = tensors.load_distance_moduli(path, device=CPU)
    assert z2.tolist() == [0.1, 0.5] and mu2.tolist() == [40.0, 43.0] and s2.tolist() == [0.2, 0.3]


def test_grid_recovers_synthetic_parameters():
    z, _, sigma = tensors.load_distance_moduli(device=CPU)
    mu = tensors.synthetic_moduli(z, sigma, 70.0, 0.3, 0.7, noise=False)
    Om, OL = tensors.parameter_grid(61, 61, (0.0, 1.2), (-0.2, 1.4), CPU)  # steps of 0.02 and 0.0267
    chi2, H0 = solve.chi2_grid(z, mu, sigma, Om, OL)
    i, j = divmod(int(chi2.argmin()), chi2.shape[1])
    assert abs(Om[i] - 0.3) < 0.011 and abs(OL[j] - 0.7) < 0.014
    assert abs(H0[i, j] - 70.0) < 0.1  # H0 from the analytic shift (the grid misses the truth by < half a step)


def test_adam_fit_recovers_noisy_synthetic_within_errors():
    z, _, sigma = tensors.load_distance_moduli(device=CPU)
    gen = torch.Generator().manual_seed(3)
    mu = tensors.synthetic_moduli(z, sigma, 70.0, 0.3, 0.7, noise=True, generator=gen)
    fit = solve.fit(z, mu, sigma, start=(70.0, 1.0, 0.0), steps=600, lr=0.02)
    best, err = fit["best"], fit["errors"]
    for value, truth, e in zip(best.tolist(), (70.0, 0.3, 0.7), err.tolist()):
        assert abs(value - truth) < 3.5 * e  # within the Hessian error bars
    assert fit["chi2"][-1] <= fit["chi2"][0]


def test_chi2_gradient_matches_finite_difference():
    z, mu, sigma = tensors.load_distance_moduli(device=CPU)
    p = torch.tensor([71.0, 0.35, 0.6], dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(solve.chi2_of(p, z, mu, sigma), p)[0]
    for k in range(3):
        dp = torch.zeros(3, dtype=torch.float64)
        dp[k] = 1e-6
        num = (solve.chi2_of(p.detach() + dp, z, mu, sigma) - solve.chi2_of(p.detach() - dp, z, mu, sigma)) / 2e-6
        assert g[k].item() == pytest.approx(num.item(), rel=1e-4, abs=1e-3)


def test_real_supernovae_need_a_cosmological_constant():
    z, mu, sigma = tensors.load_distance_moduli(device=CPU)
    Om, OL = tensors.parameter_grid(81, 81, (0.0, 1.2), (-0.4, 1.6), CPU)
    chi2, _ = solve.chi2_grid(z, mu, sigma, Om, OL)
    sig = solve.significance(chi2, Om, OL)
    assert sig["sigma_lambda"] > 5  # Ω_Λ ≤ 0 excluded at more than 5σ
    assert sig["sigma_accel"] > 5  # q0 ≥ 0 (decelerating) excluded at more than 5σ
    assert sig["best_q0"] < 0 and sig["p_accel"] > 0.9999
    assert 0.1 < sig["best_Om"] < 0.6 and 0.3 < sig["best_OL"] < 1.1
