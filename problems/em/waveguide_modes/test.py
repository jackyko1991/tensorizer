"""Physics checks, written first (TDD): what any correct waveguide-mode code must satisfy.

Run with `pytest problems/em/waveguide_modes/test.py`. The numerical cutoffs match the exact π√((m/a)² + (n/b)²),
TE₁₀ of WR-90 cuts off at 6.56 GHz, and the modes of the self-adjoint Laplacian are orthonormal.
"""

import torch

from . import equation, solve, tensors


def test_cutoffs_match_exact():
    a, b = 2.286, 1.016  # WR-90
    found = solve.solve(a, b, 24, 8)
    for mode in found:
        assert abs(mode["k_c"] - equation.exact_cutoff(*mode["mn"], a, b)) / mode["k_c"] < 0.01  # k_c = π√(m²/a²+n²/b²)
    assert [(m["kind"], m["mn"]) for m in found[:3]] == [("TE", (1, 0)), ("TE", (2, 0)), ("TE", (0, 1))]
    assert abs(solve.cutoff_frequency_GHz(found[0]["k_c"]) - 6.56) < 0.02  # f_c = c/2a for TE₁₀


def test_dirichlet_cutoffs_and_orthonormal_modes():
    a, b = 2.0, 1.0
    nx, ny = tensors.grid_size(a, b, 20)
    _, _, hx, hy = tensors.grid(a, b, nx, ny, "TM")
    k_c, psi = equation.modes(equation.laplacian(nx, ny, hx, hy, "dirichlet"), 5, skip_zero=False)
    exact = sorted(equation.exact_cutoff(m, n, a, b) for m in range(1, 6) for n in range(1, 4))[:5]
    assert torch.allclose(k_c, torch.tensor(exact, dtype=torch.float64), rtol=0.01)
    gram = torch.einsum("nk,nl->kl", psi, psi)  # ψ_k · ψ_l, summed over grid points n
    assert torch.allclose(gram, torch.eye(5, dtype=torch.float64), atol=1e-10)  # modes are orthonormal
