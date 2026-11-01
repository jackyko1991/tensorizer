"""Physics checks, written first (TDD): what any correct reactor-criticality code must satisfy.

Run with `pytest problems/nuclear/reactor_criticality/test.py`. Each test is a physical law, not an implementation
detail: k matches the one-group bare-slab formula, power iteration converges to the cosine flux mode and k = 1 at
the critical width.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")

D, SA, NSF = 0.9, 0.066, 0.070


def slab(width: float, n: int = 200):
    x, h = tensors.grid(width, n, CPU)
    m = equation.loss_operator(tensors.laplacian(n, h, CPU), D, SA)
    f = equation.production_operator(n, NSF, CPU)
    return x, m, f


def test_k_matches_bare_slab_formula():
    for width in (30.0, 47.0, 60.0, 120.0):
        _, m, f = slab(width)
        # one-group diffusion: k = νΣf/(Σa + DB²)
        assert math.isclose(solve.k_eigen(m, f).item(), equation.analytic_k(width, D, SA, NSF), rel_tol=1e-5)


def test_power_iteration_converges_to_cosine_mode():
    width = 60.0
    x, m, f = slab(width)
    ks, fluxes = solve.power_iteration(m, f, tensors.initial_guess(x, width), 200)
    assert math.isclose(ks[-1].item(), solve.k_eigen(m, f).item(), rel_tol=1e-8)
    shape = torch.sin(math.pi * x / width)  # = cos(π x'/L) measured from the centre
    assert torch.allclose(fluxes[-1], shape / shape.max(), atol=1e-4)  # fundamental mode of the bare slab


def test_critical_width_and_batched_sizes():
    l_crit = equation.critical_width(D, SA, NSF)
    _, m, f = slab(l_crit)
    assert math.isclose(solve.k_eigen(m, f).item(), 1.0, rel_tol=1e-5)  # critical: k = 1 exactly at L_crit
    widths = torch.tensor([20.0, 40.0, 80.0], dtype=torch.float64)
    k = solve.k_versus_width(widths, 150, D, SA, NSF)
    assert torch.allclose(k, equation.analytic_k(widths, D, SA, NSF), rtol=1e-4)
    assert k[0] < 1 < k[2]
