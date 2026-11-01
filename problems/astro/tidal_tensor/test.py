"""Physics checks, written first (TDD): what any correct tidal-tensor code must satisfy.

Run with `pytest problems/astro/tidal_tensor/test.py`. Each test is a physical law, not an implementation detail:
the tidal tensor is traceless (vacuum Poisson equation) with eigenvalues (−1, −1, 2)GM/r³, and a falling ring
stretches radially and squeezes transversely as r/r0.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_tensor_is_traceless_with_known_eigenvalues():
    x = torch.randn(20, 3, dtype=torch.float64) * 10
    T = equation.tidal_tensor(x, 1.0)
    r3 = x.norm(dim=-1) ** 3
    # vacuum Poisson equation: tr T = ∇²Φ = 0
    assert torch.allclose(torch.einsum("pii->p", T), torch.zeros(20, dtype=torch.float64), atol=1e-14)
    ev = torch.linalg.eigvalsh(T) * r3[:, None]
    expected = torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64).expand_as(ev)
    assert torch.allclose(ev, expected, atol=1e-9)  # eigenvalues (−1, −1, 2) GM/r³: stretch along r


def test_autograd_matches_formula():
    x = torch.randn(10, 3, dtype=torch.float64) * 5
    assert torch.allclose(equation.tidal_tensor_autograd(x, 2.0), equation.tidal_tensor(x, 2.0), atol=1e-12)


def test_ring_stretches_radially_and_matches_linear_law_when_small():
    y0 = tensors.initial_state(40.0, 0.05, 16, CPU)
    sol = solve.solve(y0, 1.0, r_stop=10.0, n_steps=1500)
    along, across = solve.stretch(sol["xi_exact"])
    assert along[-1] > 2 * along[0] and across[-1] < 0.5 * across[0]
    assert torch.allclose(sol["xi_exact"][-1], sol["xi_tidal"][-1], atol=0.02 * along[-1].item())
    # transverse squeeze is exactly r/r0 for a radial fall (all particles head for the same point)
    assert across[-1].item() / 0.05 == pytest.approx(10.0 / 40.0, rel=1e-3)
