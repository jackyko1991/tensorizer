"""Checks, written first (TDD): what any correct calculation of the projection postulates must satisfy.

Run with `pytest problems/cosmology/projection_postulate/test.py`. The hypothesis itself is speculative; these tests
check the mathematics it rests on. A uniform state on the unit sphere of C^D, projected onto d of its coordinates,
keeps a fraction ||P Psi||^2 ~ Beta(d, D - d) of its norm, and the normalised projection is uniform on the smaller
sphere. In the projection race (postulate P2 made concrete) one hidden complex dimension per outcome gives exactly
the Born rule, more hidden dimensions give a computable deviation that grows with m. N qubits need 2^N dimensions.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def shadows():
    Ds = torch.tensor([4, 16, 128, 1024])
    return Ds, tensors.projected_states(Ds, d=2, trials=6000, generator=gen(1), device=CPU)


def test_projected_norm_follows_beta(shadows):
    Ds, sh = shadows
    a, b = equation.beta_params(2, Ds, field="complex")  # Beta(d, D - d)
    q = sh["q"].T  # (n_D, T) fraction of the norm the observer sees
    mean, var = equation.beta_mean_var(a, b)
    assert torch.allclose(mean, 2 / Ds.double())  # mean d / D
    sigma_mean = (var / q.shape[1]).sqrt()
    assert ((q.mean(1) - mean).abs() < 5 * sigma_mean).all()
    assert torch.allclose(q.var(1), var, rtol=0.1)
    ks = equation.ks_distance(q, a, b)
    assert (ks < 1.63 / math.sqrt(q.shape[1])).all()  # Kolmogorov-Smirnov, 1% level
    ks_wrong = equation.ks_distance(q, a + 1.0, b)  # a different Beta law is rejected
    assert (ks_wrong > 3 / math.sqrt(q.shape[1])).all()


def test_real_three_dimensional_shadow_follows_beta_three_halves(shadows):
    Ds, sh = shadows
    r2 = (sh["shadow3"] ** 2).sum(-1).T  # (n_D, T): Psi in R^(2D), 3 real coordinates seen
    a, b = equation.beta_params(3, 2 * Ds, field="real")  # Beta(3/2, D - 3/2)
    assert (equation.ks_distance(r2, a, b) < 1.63 / math.sqrt(r2.shape[1])).all()


def test_beta_cdf_matches_closed_forms():
    x = torch.linspace(0.01, 0.99, 25, dtype=torch.float64)[None, :]
    one = torch.ones(1, dtype=torch.float64)
    assert torch.allclose(equation.beta_cdf(x, one, one), x, atol=1e-8)  # Beta(1, 1) is uniform
    assert torch.allclose(equation.beta_cdf(x, one, 4 * one), 1 - (1 - x) ** 4, atol=1e-8)
    arcsine = 2 / math.pi * torch.asin(x.sqrt())  # Beta(1/2, 1/2)
    assert torch.allclose(equation.beta_cdf(x, 0.5 * one, 0.5 * one), arcsine, atol=1e-6)


def test_normalised_projection_is_uniform_on_the_small_sphere(shadows):
    _, sh = shadows
    psi = sh["psi"]  # (T, d) complex, the observed state P Psi / ||P Psi||
    T, d = psi.shape
    assert torch.allclose(psi.abs().pow(2).sum(1), torch.ones(T, dtype=torch.float64))
    rho = torch.einsum("ti,tj->ij", psi, psi.conj()) / T  # E[psi psi^dagger] = I / d on a uniform sphere
    assert torch.allclose(rho, torch.eye(d, dtype=rho.dtype) / d, atol=5 / math.sqrt(T))
    p1 = psi[:, 0].abs().pow(2)[None, :]  # |psi_1|^2 ~ Beta(1, d - 1)
    one = torch.ones(1, dtype=torch.float64)
    assert equation.ks_distance(p1, one, (d - 1) * one).item() < 1.63 / math.sqrt(T)
    u = sh["dir3"][:, 2][None, :]  # Archimedes: a uniform point on S^2 has a uniform height in [-1, 1]
    assert equation.ks_distance((u + 1) / 2, one, one).item() < 1.63 / math.sqrt(T)
    assert sh["dir3"].mean(0).abs().max() < 5 / math.sqrt(3 * T)


def test_hidden_block_energy_is_gamma_m():
    E = tensors.hidden_energies(40000, 2, 6, "complex", gen(2), CPU)  # (T, K, m)
    m = torch.arange(1, 7, dtype=torch.float64)
    assert torch.allclose(E.mean((0, 1)), m, rtol=0.02)  # Gamma(m, 1): mean m, variance m
    assert torch.allclose(E.var((0, 1)), m, rtol=0.05)
    Er = tensors.hidden_energies(40000, 2, 4, "real", gen(3), CPU)  # chi-square: 2 Gamma(m / 2)
    assert torch.allclose(Er.mean((0, 1)), torch.arange(1, 5, dtype=torch.float64), rtol=0.02)


def test_exponential_race_is_the_born_rule():
    w = tensors.born_weights(5, CPU)  # |c_k|^2
    P = equation.race_probabilities(w, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(P, w, atol=1e-10)  # m = 1: P(k) = |c_k|^2 exactly
    T = 60000
    E = tensors.hidden_energies(T, 5, 1, "complex", gen(4), CPU)[..., 0]  # (T, K) Exp(1)
    freq = equation.race_frequencies(w, E)
    assert ((freq - w).abs() < 4 * (w * (1 - w) / T).sqrt()).all()


def test_two_outcome_race_closed_forms():
    w = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    W = torch.stack([w, 1 - w], -1)  # (19, 2)
    for m in (1, 2, 3, 5):
        P = equation.race_probabilities(W, torch.tensor(float(m), dtype=torch.float64))
        assert torch.allclose(P[:, 0], equation.race_two_outcomes(w, m), atol=1e-9)  # I_w(m, m)
    P = equation.race_probabilities(W, torch.tensor(0.5, dtype=torch.float64))  # one REAL hidden dimension
    assert torch.allclose(P[:, 0], 2 / math.pi * torch.asin(w.sqrt()), atol=1e-8)  # arcsine law, not Born


def test_race_closed_form_matches_monte_carlo():
    w = tensors.born_weights(4, CPU)
    T, M = 40000, 6
    a = torch.arange(1, M + 1, dtype=torch.float64)
    P = equation.race_probabilities(w.expand(M, -1), a)  # (M, K) by quadrature
    freq = equation.race_frequencies(w, tensors.hidden_energies(T, 4, M, "complex", gen(5), CPU))
    assert ((freq - P).abs() < 4.5 * (P * (1 - P) / T).sqrt() + 1e-4).all()
    Pr = equation.race_probabilities(w.expand(M, -1), a / 2)
    fr = equation.race_frequencies(w, tensors.hidden_energies(T, 4, M, "real", gen(6), CPU))
    assert ((fr - Pr).abs() < 4.5 * (Pr * (1 - Pr) / T).sqrt() + 1e-4).all()


def test_deviation_grows_with_m():
    w = tensors.born_weights(4, CPU)
    a = torch.arange(1, 13, dtype=torch.float64)
    tv = equation.total_variation(equation.race_probabilities(w.expand(12, -1), a), w)
    assert tv[0] < 1e-10
    assert (tv.diff() > 0).all()  # more hidden dimensions per outcome: the largest |c_k|^2 wins more often
    assert tv[-1] < equation.total_variation(torch.nn.functional.one_hot(w.argmax(), 4).double(), w)  # m -> oo limit


def test_qubit_dimension_count():
    for N in range(1, 11):
        assert equation.log2_tensor_product_dim(N) == N  # kron of N two-level spaces
    N = torch.tensor([10.0, 1e3, 1e80], dtype=torch.float64)
    assert torch.allclose(equation.log10_min_dim_entanglement(N), N * math.log10(2.0))
    assert equation.min_dim_born(3, 4, 1) == 3 + 4  # d observed + one hidden complex dimension per outcome


def test_sorkin_parameter_vanishes_for_born_and_not_for_m_2():
    amps = tensors.slit_amplitudes(41, tau=0.05, device=CPU)  # (3, X) one amplitude pattern per slit
    W = equation.slit_born_weights(amps)  # (7, X + 1): pixels plus 'blocked'
    assert torch.allclose(W.sum(-1), torch.ones(7, dtype=torch.float64))
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    kappa = equation.sorkin_kappa(equation.race_probabilities(W.expand(2, -1, -1), a[:, None]), pixel=20)
    assert abs(kappa[0].item()) < 1e-9  # Born rule: no third-order interference
    assert abs(kappa[1].item()) > 1e-3


@pytest.mark.slow
def test_run_accepts_dimensions_as_parameters(tmp_path):
    from . import main

    params = {"D_max": 64, "d": 2, "m_max": 3, "outcomes": 3, "trials": 2000}
    out = main.run(params, tmp_path)
    assert out.exists() and out.stat().st_size > 0
