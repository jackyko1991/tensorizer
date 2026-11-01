"""Physics checks, written first (TDD): what any correct linear-elasticity code must satisfy.

Run with `pytest problems/fluids/stress_strain/test.py`. The stiffness tensor has minor and major symmetries, Hooke's
law reproduces Young's modulus and Poisson's ratio, and principal stresses are invariant under rotation.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def material():
    lam, mu = equation.lame(200e9, 0.3)
    delta = tensors.kronecker(CPU)
    return lam, mu, delta, equation.stiffness(lam, mu, delta)


def test_stiffness_tensor_symmetries():
    _, _, _, C = material()
    assert torch.allclose(C, C.transpose(0, 1))  # minor symmetry ij (σ symmetric: angular momentum balance)
    assert torch.allclose(C, C.transpose(2, 3))  # minor symmetry kl
    assert torch.allclose(C, C.permute(2, 3, 0, 1))  # major symmetry (a strain-energy function exists)


def test_einsum_hooke_matches_lame_form():
    lam, mu, delta, C = material()
    a = torch.randn(5, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 1e-3
    eps = 0.5 * (a + a.transpose(-1, -2))
    assert torch.allclose(equation.stress(C, eps), equation.lame_stress(eps, lam, mu, delta), rtol=1e-10)


def test_uniaxial_stress_gives_young_and_poisson():
    E, nu = 200e9, 0.3
    _, _, _, C = material()
    e = 1e-3
    eps = torch.diag(torch.tensor([e, -nu * e, -nu * e], dtype=torch.float64))
    sigma = equation.stress(C, eps)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected[0, 0] = E * e
    assert torch.allclose(sigma, expected, atol=1e-3 * E * e)  # Hooke: σ = Eε with lateral strain −νε
    S = solve.compliance(C)
    back = torch.einsum("ijkl,kl->ij", S, sigma)
    assert torch.allclose(back, eps, atol=1e-12)


def test_principal_stresses_invariant_under_rotation():
    sigma = torch.tensor([[100.0, 60.0, 0.0], [60.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    a = 0.7
    R = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=torch.float64)
    rotated = torch.einsum("ia,ab,jb->ij", R, sigma, R)
    v1, _ = equation.principal_stresses(sigma)
    v2, _ = equation.principal_stresses(rotated)
    assert torch.allclose(v1, v2, atol=1e-9)  # eigenvalues are frame invariants
    centre, radius = equation.mohr_circle(sigma)
    assert (centre + radius).item() == pytest.approx(v1.max().item())  # Mohr's circle hits σ1
    assert torch.allclose(equation.mohr_circle(rotated)[1], radius)


def test_strain_from_displacement_gradient_round_trip():
    _, _, _, C = material()
    x = tensors.block_points(7, CPU)
    loads = tensors.loading(5, 100e6, 60e6, CPU)
    result = solve.load_path(loads, C, x)
    assert torch.allclose(result["sigma"], loads, atol=1e-3 * 100e6)  # σ = C : ε(∇u) recovers the applied stress
