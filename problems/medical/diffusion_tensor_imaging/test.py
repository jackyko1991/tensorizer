"""Physics checks, written first (TDD): what any correct diffusion-tensor code must satisfy.

Run with `pytest problems/medical/diffusion_tensor_imaging/test.py`. Each test is a physical law: signals follow the
Stejskal–Tanner law, MD is trace / 3, FA is 0 for isotropic and 1 for linear diffusion, and tracts follow the
principal eigenvector.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_fit_recovers_true_tensor_without_noise():
    g = tensors.gradient_directions(30, CPU)
    D, fibre, label = tensors.phantom(16, CPU)
    S0, S = solve.measure(D, g, 1.0, noise=0.0)
    D_fit, lam, e1, md, fa = solve.fit(S0, S, tensors.design_matrix(g), 1.0)
    assert (D_fit - D).abs().max() < 1e-8
    single = (label == 1) | (label == 2)
    align = (e1[single] * fibre[single]).sum(-1).abs()
    assert align.min() > 1 - 1e-8  # principal eigenvector along the fibre
    assert torch.allclose(md, torch.einsum("vii->v", D) / 3)  # MD = trace / 3
    assert torch.allclose(md, lam.mean(-1))


def test_fa_limits():
    iso = 2.0 * torch.eye(3, dtype=torch.float64)[None]
    linear = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))[None]
    nearly = torch.diag(torch.tensor([1.0, 1e-4, 1e-4], dtype=torch.float64))[None]
    assert equation.invariants(iso)[3].item() == pytest.approx(0.0, abs=1e-12)
    assert equation.invariants(linear)[3].item() == pytest.approx(1.0, abs=1e-12)
    assert equation.invariants(nearly)[3].item() > 0.999
    bundle = tensors.fibre_tensor(torch.tensor([[1.0, 0, 0]], dtype=torch.float64), 1.7, 0.3)
    lam = torch.tensor([1.7, 0.3, 0.3], dtype=torch.float64)
    fa = math.sqrt(1.5) * (lam - lam.mean()).norm().item() / lam.norm().item()
    assert equation.invariants(bundle)[3].item() == pytest.approx(fa, rel=1e-9)


def test_signal_attenuation_along_direction():
    e = torch.tensor([[0.6, 0.8, 0.0]], dtype=torch.float64)
    D = tensors.fibre_tensor(e, 1.7, 0.3)
    g = torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [0.8, -0.6, 0.0]], dtype=torch.float64)
    b = 1.0  # 1000 s/mm² in units of (10⁻³ mm²/s)⁻¹
    S = equation.signal(D, g, b)[0]
    assert S.tolist() == pytest.approx([math.exp(-1.7), math.exp(-0.3), math.exp(-0.3)], rel=1e-12)


def test_tractography_follows_straight_bundle():
    n = 12
    D = tensors.fibre_tensor(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), 1.7, 0.3).expand(n, n, 3, 3)
    lines = solve.track(D, torch.tensor([[5.0, 5.0]], dtype=torch.float64), h=0.5, n_steps=30)
    pts = lines[:, 0][~lines[:, 0, 0].isnan()]
    assert (pts[:, 0] - 5.0).abs().max() < 1e-9  # stays on its column
    assert pts[:, 1].max() > 10.5  # and runs along the fibre
