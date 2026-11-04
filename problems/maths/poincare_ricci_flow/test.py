"""Mathematical checks, written first (TDD): what any correct Ricci-flow code must satisfy.

Run with `pytest problems/maths/poincare_ricci_flow/test.py`. Each test is a theorem, not an implementation detail:
Gauss–Bonnet (∫K dA = 4π on every 2-sphere), the round sphere is a fixed point of the normalised flow, every metric on
S² becomes round (Hamilton 1988, Chow 1991), the round S³ shrinks as r² = r₀² − 4t, and a dumbbell 3-sphere with a
thin enough neck pinches in finite time while the bulbs stay large (Angenent–Knopf 2004).
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_gauss_bonnet_holds_for_any_conformal_factor():
    x, w, lap = tensors.sphere_grid(64, CPU)
    u = 0.3 * torch.randn(5, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(0))  # rough metrics
    k = equation.gauss_curvature(u, lap)  # (B, N)
    total = equation.total_curvature(k, u, w)
    assert torch.allclose(total, torch.full_like(total, 4 * math.pi), atol=1e-11)  # χ(S²) = 2 ⇒ ∫K dA = 4π


def test_round_sphere_is_a_fixed_point():
    x, w, lap = tensors.sphere_grid(64, CPU)
    u0 = torch.zeros(1, 64, dtype=torch.float64)
    out = solve.flow_surface(u0, w, lap, t_max=0.5, n_save=3)
    assert out["u"][-1].abs().max() < 1e-12
    assert torch.allclose(equation.gauss_curvature(u0, lap), torch.ones(1, 64, dtype=torch.float64), atol=1e-12)


@pytest.mark.slow
def test_lumpy_spheres_become_round_and_keep_area_and_total_curvature():
    x, w, lap = tensors.sphere_grid(64, CPU)
    u0 = tensors.lumpy_factors(x)  # (B, N): peanut, pear, lumpy
    out = solve.flow_surface(u0, w, lap, t_max=3.0, n_save=4)
    k_last = out["K"][-1]  # (B, N)
    spread0 = equation.curvature_spread(out["K"][0])
    spread = equation.curvature_spread(k_last)
    assert torch.all(spread0 > 0.3) and torch.all(spread < 1e-3)  # curvature becomes uniform
    area = out["area"]  # (S, B)
    assert torch.allclose(area[-1], area[0], rtol=1e-6)  # normalised flow keeps the area
    total = equation.total_curvature(k_last, out["u"][-1], w)
    assert torch.allclose(total, torch.full_like(total, 4 * math.pi), atol=1e-10)


def test_initial_shapes_are_embeddable_surfaces_of_revolution():
    x, w, lap = tensors.sphere_grid(64, CPU)
    rho, z, slack = equation.embed_profile(tensors.lumpy_factors(x), x)
    assert torch.isfinite(z).all() and torch.all(rho >= 0)
    assert torch.all(slack[:, 1:-1] > 0)  # |dρ/ds| < 1: the metric is realised by a surface in R³


@pytest.mark.slow
def test_round_three_sphere_shrinks_as_r2_minus_4t():
    r0, t_end = 1.0, 0.15
    grid = tensors.s3_grid(48, CPU)
    out = solve.flow_s3(tensors.round_s3(r0, grid[0]), grid, t_max=t_end, psi_stop=0.0)
    r = math.sqrt(r0**2 - 4 * t_end)  # ∂g/∂t = −2 Ric = −4 g / r² on the round S³ of radius r
    assert torch.allclose(torch.exp(out["u"][-1]), torch.full_like(grid[0], r)[None], atol=1e-4)  # stays round


def test_dumbbell_neck_pinches_while_bulbs_survive():
    grid = tensors.s3_grid(96, CPU)
    u0 = tensors.dumbbell(0.15, grid[0])
    out = solve.flow_s3(u0, grid, t_max=0.2, psi_stop=0.05)
    neck, t = out["neck"], out["t"]  # (T,) radius of the thinnest S² and the time of every step
    assert neck[-1] < 0.05 and t[-1] < 0.05  # the neck reaches the surgery scale long before the bulbs could vanish
    psi0 = equation.profile(u0, grid)[1]
    psi1 = equation.profile(out["u"][-1], grid)[1]
    assert psi1.max() > 0.7 * psi0.max()  # the bulbs stay large: a local singularity
    i = int(torch.searchsorted(-neck, torch.tensor(-0.1, dtype=neck.dtype)))  # first step with neck < 0.1
    slope = (neck[-2] ** 2 - neck[i] ** 2) / (t[-2] - t[i])
    assert -2.5 < slope < -1.2  # ψ_neck² falls at roughly the rate 2 of a shrinking cylinder S² × R


def test_fat_neck_does_not_pinch():
    grid = tensors.s3_grid(64, CPU)
    out = solve.flow_s3(tensors.dumbbell(0.7, grid[0]), grid, t_max=0.1, psi_stop=0.05)
    assert out["t"][-1] == pytest.approx(0.1) and torch.isinf(out["neck"][-1])  # the neck opens up and disappears


@pytest.mark.slow
def test_capped_piece_becomes_round_after_surgery():
    grid = tensors.s3_grid(96, CPU)
    out = solve.flow_s3(tensors.dumbbell(0.15, grid[0]), grid, t_max=0.2, psi_stop=0.05)
    capped = solve.surgery(out["u"][-1], grid)
    after = solve.flow_s3(capped, grid, t_max=1.0, psi_stop=0.0, radius_stop=0.36, save_every=10**9)
    k_rad, k_tan = equation.s3_curvatures(after["u"][-1], grid)
    k = torch.cat([k_rad, k_tan], dim=1)
    assert k.min() > 0 and k.max() / k.min() < 1.7  # all sectional curvatures positive and nearly equal (Hamilton)
