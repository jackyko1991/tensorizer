"""Physics checks, written first (TDD): what any correct Gray–Scott code must satisfy.

Run with `pytest problems/biophysics/turing_patterns/test.py`. Each test is a physical law: uniform states are
steady, the dispersion relation is unstable only in a finite band (never with equal diffusion), and a small wave
grows at the predicted linear rate.
"""

import math

import torch

from . import equation, solve, tensors


def test_uniform_states_are_steady():
    for F, k in [(0.06, 0.062), (0.078, 0.061), (0.03, 0.055)]:
        for u, v in equation.uniform_states(F, k):
            Ru, Rv = equation.reaction(torch.tensor(u), torch.tensor(v), torch.tensor(F), torch.tensor(k))
            assert abs(Ru.item()) < 1e-6 and abs(Rv.item()) < 1e-6
    assert len(equation.uniform_states(0.029, 0.057)) == 1  # F < 4 (F + k)²: only u = 1, v = 0


def test_dispersion_relation_grows_only_where_expected():
    q = torch.linspace(0, 1.5, 301, dtype=torch.float64)
    Du = tensors.DU
    # the trivial state u = 1, v = 0 decays at every wavenumber
    lam = equation.growth_rate(equation.jacobian(1.0, 0.0, 0.06, 0.062), Du, Du / 2, q)
    assert lam.max() < 0
    # the "Turing" pair: stable at q = 0, unstable in a band of q > 0 that ends again at short wavelengths
    u, v = tensors.turing_state(0.060, 0.062, Du, Du / 2)
    lam = equation.growth_rate(equation.jacobian(u, v, 0.060, 0.062), Du, Du / 2, q)
    band = q[lam > 0]
    assert lam[0] < 0 < lam.max()
    assert 0.3 < band.min() < band.max() < 0.8 and lam[-1] < 0
    # with equal diffusion λ(q) = λ(0) - D q²: no Turing instability is possible
    lam_eq = equation.growth_rate(equation.jacobian(u, v, 0.060, 0.062), Du, Du, q)
    assert torch.allclose(lam_eq, lam_eq[0] - Du * q**2, atol=1e-12)
    assert tensors.turing_state(0.060, 0.062, Du, Du) is None


def test_laplacian_eigenvalue_and_linear_growth_rate():
    n, m = 64, 5
    x = torch.arange(n, dtype=torch.float64)
    qx = 2 * math.pi * m / n
    wave = torch.cos(qx * x)[None, :, None].expand(1, n, n)  # varies along i only
    assert torch.allclose(equation.laplacian(wave), -equation.lattice_q2(torch.tensor(qx)) * wave, atol=1e-12)

    # a tiny wave on the Turing-unstable state grows at the rate the (lattice) dispersion relation predicts
    F, k, Du, Dv = 0.060, 0.062, tensors.DU, tensors.DU / 2
    u0, v0 = tensors.turing_state(F, k, Du, Dv)
    y = torch.stack([torch.full((1, n, n), u0), torch.full((1, n, n), v0)]).double() + 1e-7 * wave
    Ft, kt = torch.tensor(F, dtype=torch.float64), torch.tensor(k, dtype=torch.float64)
    amp = []
    for _ in range(1500):
        y = y + equation.rhs(y, Ft, kt, Du, Dv)  # explicit Euler, dt = 1
        amp.append((y[1] - v0).abs().max().item())
    measured = math.log(amp[1499] / amp[899]) / 600  # late enough for the decaying mode to have gone
    q_eff = torch.sqrt(equation.lattice_q2(torch.tensor([qx], dtype=torch.float64)))
    lam = equation.growth_rate(equation.jacobian(u0, v0, F, k), Du, Dv, q_eff).item()
    assert abs(measured - math.log(1 + lam)) < 0.03 * lam


def test_dominant_wavelength_of_stripes():
    n = 96
    x = torch.arange(n, dtype=torch.float32)
    stripes = torch.sin(2 * math.pi * 8 * x / n)[None, None, :].expand(1, n, n)
    assert solve.dominant_wavelength(stripes).item() == n / 8
