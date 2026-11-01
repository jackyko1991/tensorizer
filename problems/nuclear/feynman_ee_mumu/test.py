"""Physics checks, written first (TDD): what any correct e⁺e⁻ → μ⁺μ⁻ code must satisfy.

Run with `pytest problems/nuclear/feynman_ee_mumu/test.py`. Each test is a physical law, not an implementation
detail: the gamma matrices obey the Clifford algebra, |M|² matches the textbook trace result, the high-energy cross
section is σ = 4πα²/3s with a 1 + cos²θ shape and the threshold is isotropic.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")

G, ETA = tensors.gamma_matrices(CPU), tensors.metric(CPU)
ONE = torch.eye(4, dtype=torch.complex128)


def mink(a, b):
    return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)


def test_clifford_algebra_and_traces():
    anti = torch.einsum("mab,nbc->mnac", G, G) + torch.einsum("nab,mbc->mnac", G, G)
    assert torch.allclose(anti, 2 * ETA[:, :, None, None] * ONE)  # Clifford algebra {γ^μ, γ^ν} = 2η^μν
    assert math.isclose(torch.trace(ONE).real.item(), 4)
    assert torch.allclose(equation.trace_tensor(ONE, ONE, G), 4 * ETA)


def test_slash_squares_to_mass():
    p = torch.tensor([2.0, 0.3, -0.4, 1.2], dtype=torch.float64)
    ps = equation.slash(p, G, ETA)
    assert torch.allclose(ps @ ps, mink(p, p) * ONE)  # p̸² = p² = m²


def test_amplitude_matches_peskin_with_masses():
    theta = tensors.angles(13, CPU)
    sqrt_s = torch.tensor([0.25, 1.0, 7.0], dtype=torch.float64)
    m = tensors.FINAL["muon"][1]
    p, pb, k, kb = tensors.momenta(sqrt_s, theta, m)
    msq = equation.amplitude_squared(p, pb, k, kb, tensors.M_E, m, tensors.ALPHA, G, ETA)
    e4 = (4 * math.pi * tensors.ALPHA) ** 2
    s = mink(p + pb, p + pb)
    me = tensors.M_E
    bracket = mink(p, k) * mink(pb, kb) + mink(p, kb) * mink(pb, k)
    bracket = bracket + m**2 * mink(p, pb) + me**2 * mink(k, kb) + 2 * me**2 * m**2
    expected = 8 * e4 / s**2 * bracket
    assert torch.allclose(msq, expected, rtol=1e-10)


def test_high_energy_angular_shape_and_total():
    theta = tensors.angles(181, CPU)
    sqrt_s = torch.tensor([30.0], dtype=torch.float64)
    m = tensors.FINAL["muon"][1]
    dsdo, sigma = solve.cross_sections(sqrt_s, theta, m, tensors.ALPHA, G, ETA)
    shape = dsdo[0] / dsdo[0, 90]
    assert torch.allclose(shape, 1 + torch.cos(theta) ** 2, rtol=1e-4)  # high energy: dσ/dΩ ∝ 1 + cos²θ
    # point-like σ = 4πα²/3s
    assert torch.allclose(sigma, equation.sigma_point_like(sqrt_s, m, tensors.ALPHA), rtol=1e-4)
    assert abs(solve.forward_backward(dsdo, theta).item()) < 1e-10


def test_threshold_is_isotropic_and_tau_heavier():
    theta = tensors.angles(91, CPU)
    m = tensors.FINAL["tau"][1]
    sqrt_s = torch.tensor([2 * m * 1.0005], dtype=torch.float64)
    dsdo, _ = solve.cross_sections(sqrt_s, theta, m, tensors.ALPHA, G, ETA)
    assert (dsdo[0] / dsdo[0].mean() - 1).abs().max() < 2e-3  # threshold: s-wave, isotropic
