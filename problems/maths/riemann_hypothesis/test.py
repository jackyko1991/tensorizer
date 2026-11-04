"""Mathematical checks, written first (TDD): what any correct computation of zeta zeros must satisfy.

Run with `pytest problems/maths/riemann_hypothesis/test.py`. Each test is a known fact, not an implementation detail:
the Riemann–Siegel Z matches an independent high-precision evaluation (mpmath), the zeros sit at the published
heights, there are exactly 649 zeros below t = 1000 (Riemann–von Mangoldt plus S(t)), Gram's law first fails at g_126,
the unfolded spacings repel like GUE eigenvalues, and the explicit formula rebuilds ψ(x) from the zeros.
"""

import functools
import math

import mpmath
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@functools.cache  # several tests share the same zeros
def zeros_up_to(t_max: float) -> torch.Tensor:
    t = tensors.t_grid(t_max, 0.02, CPU)
    return solve.find_zeros(t, solve.z_on_grid(t))


def test_riemann_siegel_matches_high_precision_reference():
    t = torch.tensor([14.5, 37.9, 88.0, 281.3, 1000.37, 4999.8], dtype=torch.float64)
    ref = torch.tensor([float(mpmath.siegelz(x)) for x in t.tolist()], dtype=torch.float64)
    err = (equation.riemann_siegel_z(t) - ref).abs()
    assert err[0] < 1e-3 and err[1] < 1e-4 and err[2:].max() < 1e-5  # corrections C₀, C₁, C₂ improve with t


def test_psi_is_smooth_through_its_removable_points():
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    d = equation.psi_derivatives(p)
    near = equation.psi(p + 1e-4)
    assert torch.allclose(d[0], near, atol=1e-3)  # no 0/0 at p = ¼, ¾
    assert all(torch.isfinite(v).all() for v in d)


def test_zeros_match_published_values_and_count():
    gammas = zeros_up_to(1000.0)
    assert len(gammas) == 649  # N(1000) = 649 (e.g. mpmath.nzeros(1000))
    for k, known in tensors.KNOWN_ZEROS.items():
        if known < 1000:
            assert abs(gammas[k - 1].item() - known) < (2e-4 if k < 10 else 1e-6)  # Riemann–Siegel error ~ t^{-7/4}
    # Riemann–von Mangoldt: N(t) = θ(t)/π + 1 + S(t) with S small and averaging to zero
    t = torch.linspace(15, 1000, 5000, dtype=torch.float64)
    s = solve.s_function(gammas, t)
    assert s.abs().max() < 1.5 and abs(s.mean().item()) < 0.1


def test_gram_law_first_fails_at_126():
    g = tensors.gram_points(200, CPU)
    assert torch.allclose(equation.theta(g), torch.arange(201, dtype=torch.float64) * math.pi, atol=1e-9)
    assert abs(g[0].item() - 17.8455995405) < 1e-6
    bad = solve.gram_violations(equation.riemann_siegel_z(g))
    assert bad[0].item() == 126  # first "bad" Gram point g_126 ≈ 282.455 (Hutchinson 1925)


def test_spacings_repel_like_gue():
    x = solve.unfold(zeros_up_to(3000.0))
    s = x.diff()
    assert abs(s.mean().item() - 1) < 0.01  # unfolding gives mean spacing 1
    grid = torch.linspace(0, 4, 4001, dtype=torch.float64)
    emp = torch.searchsorted(s.sort().values, grid, right=True) / len(s)  # empirical CDF
    cdf_gue = torch.cumsum(equation.wigner_gue(grid), 0) * (grid[1] - grid[0])
    cdf_poisson = 1 - torch.exp(-grid)
    ks_gue, ks_poisson = (emp - cdf_gue).abs().max(), (emp - cdf_poisson).abs().max()
    assert ks_gue < 0.05 < 0.25 < ks_poisson  # GUE statistics, not independent (Poisson) points
    assert (s < 0.1).double().mean() < 0.005  # level repulsion: p(s) ~ s² near 0


def test_pair_correlation_follows_montgomery():
    x = solve.unfold(zeros_up_to(3000.0))
    edges = torch.linspace(0, 3, 31, dtype=torch.float64)
    r2 = solve.pair_correlation(x, edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    assert (r2 - equation.montgomery_r2(mid)).abs().mean() < 0.06
    assert r2[:3].mean() < 0.1 and abs(r2[15:].mean().item() - 1) < 0.08  # repulsion at short range, 1 far away


def test_explicit_formula_rebuilds_psi():
    gammas = zeros_up_to(1000.0)
    x = torch.arange(2.5, 60, 1.0, dtype=torch.float64)  # half-integers, away from the jumps
    exact = tensors.chebyshev_psi(60, CPU)[x.floor().long()]
    psi_k = solve.psi_from_zeros(x, gammas)  # (K + 1, X)
    err = (psi_k - exact).abs().mean(dim=1)
    assert err[-1] < err[100] < err[10] < err[0]
    assert (psi_k[-1] - exact).abs().max() < 0.5
