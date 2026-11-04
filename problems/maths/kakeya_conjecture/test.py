"""Mathematical checks, written first (TDD): what any correct Kakeya-set code must satisfy.

Run with `pytest problems/maths/kakeya_conjecture/test.py`. Each test is a geometric fact: the Perron tree keeps a
unit segment in every direction of its 60° cone while its area shrinks with refinement (but no faster than Córdoba's
1/log N), Perron's heart lands at height α^k, three turned copies cover all directions, and unions of δ-tubes have
the scaling the Kakeya conjecture is about.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_union_area_matches_brute_force_raster():
    tree = solve.perron_tree(5, CPU)
    poly = solve.tree_polygons(tree)  # (N, 3, 2)
    lo = poly[..., 0].min().item()
    hi = poly[..., 0].max().item()
    nx, ny = 1200, 600
    x = torch.linspace(lo, hi, nx, dtype=torch.float64)
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny
    pts = torch.cartesian_prod(x, y)
    inside = equation.inside_union(pts, tree["base"], tree["w0"], tree["shift"])
    raster = inside.double().mean().item() * (hi - lo)
    assert raster == pytest.approx(tree["area"], rel=0.01)


def test_unrefined_tree_is_the_triangle():
    tree = solve.perron_tree(0, CPU)
    assert tree["area"] == pytest.approx(tensors.HALF_WIDTH, rel=1e-6)  # ½ · base 2h · height 1


def test_heart_has_height_alpha_to_the_k():
    k, alpha = 6, 0.8
    base, w0 = tensors.triangle_bases(k, CPU)
    s = equation.perron_shifts(k, torch.tensor([alpha], dtype=torch.float64), tensors.HALF_WIDTH)[0]
    # left side of the first triangle: x = (b₀ + s₀)(1 − y) + s₀ y
    # right side of the last one:       x = (b + w₀ + s)(1 − y) + s y
    x0_l, x1_l = base[0] + s[0], s[0]
    x0_r, x1_r = base[-1] + w0 + s[-1], s[-1]
    y = (x0_r - x0_l) / ((x0_r - x0_l) - (x1_r - x1_l))  # where the two outer sides cross
    assert y.item() == pytest.approx(alpha**k, rel=1e-9)


def test_area_shrinks_with_refinement_within_perron_bound():
    trees = [solve.perron_tree(k, CPU) for k in range(1, 10)]
    areas = torch.tensor([t["area"] for t in trees]) / tensors.HALF_WIDTH  # fraction of the original triangle
    assert (areas[1:] < areas[:-1]).all()
    assert areas[-1] < 0.25
    for t, a in zip(trees, areas):
        assert a <= t["alpha"] ** (2 * t["k"]) + 2 * (1 - t["alpha"])  # Perron's bound
    k = torch.arange(1, 10, dtype=torch.float64)
    assert (areas * k).min() > 0.5  # Córdoba: |K_δ| ≥ c / log(1/δ); the area cannot fall faster than 1/k


def test_tree_holds_a_unit_segment_in_every_direction_of_its_cone():
    tree = solve.perron_tree(8, CPU)
    h = tensors.HALF_WIDTH
    slopes = torch.linspace(-h, h, 400, dtype=torch.float64)  # direction (b, −1) for every base point b
    i = ((slopes + h) / tree["w0"]).floor().clamp(max=len(tree["base"]) - 1).long()  # triangle holding that direction
    apex = torch.stack([tree["shift"][i], torch.ones_like(slopes)], 1)  # (M, 2)
    foot = torch.stack([slopes + tree["shift"][i], torch.zeros_like(slopes)], 1)
    assert ((foot - apex).norm(dim=1) >= 1 - 1e-12).all()  # every such segment has length ≥ 1
    t = torch.linspace(0, 1, 60, dtype=torch.float64)
    pts = (apex[:, None, :] + t[None, :, None] * (foot - apex)[:, None, :]).reshape(-1, 2)
    assert equation.inside_union(pts, tree["base"], tree["w0"], tree["shift"]).all()


def test_three_copies_cover_every_direction():
    tree = solve.perron_tree(4, CPU)
    poly = solve.kakeya_polygons(tree)  # (3N, 3, 2): sides from the apex cover a 60° cone each
    side = torch.cat([poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0]])
    angle = torch.remainder(torch.atan2(side[:, 1], side[:, 0]), math.pi)  # line directions mod 180°
    angle = torch.sort(angle).values
    gaps = torch.diff(torch.cat([angle, angle[:1] + math.pi]))  # including the wrap-around from 180° to 0°
    # consecutive sides are at most one thin triangle apart (apex angle ≤ w₀), so the cones tile [0°, 180°)
    assert gaps.max() <= tree["w0"] + 1e-9


def test_directions_are_delta_separated_and_cover_the_sphere():
    for n, delta in ((2, 1 / 32), (3, 1 / 8)):
        u = tensors.directions(n, delta, CPU)
        assert torch.allclose(u.norm(dim=1), torch.ones(len(u), dtype=torch.float64))
        probe = torch.nn.functional.normalize(torch.randn(500, n, dtype=torch.float64), dim=1)
        nearest = (probe @ u.T).abs().max(dim=1).values  # |cos| to the closest line direction
        assert torch.arccos(nearest.clamp(max=1)).max() < delta  # every direction is within δ of one


def test_single_tube_volume_scales_like_delta_to_the_n_minus_one():
    for n in (2, 3, 4):
        u = torch.zeros(1, n, dtype=torch.float64)
        u[0, 0] = 1.0
        for delta in (1 / 16, 1 / 64):
            cells = equation.count_cells(equation.segment_cells(-u / 2 + 1e-3, u / 2 + 1e-3, delta))
            assert cells == pytest.approx(1 / delta, abs=2)  # a unit segment meets about 1/δ cubes


def test_bush_of_tubes_has_full_dimension_and_bounded_volume():
    deltas = [1 / 16, 1 / 32, 1 / 64, 1 / 128]
    counts = solve.tube_counts(2, deltas, "bush", CPU)
    assert solve.dimension_slope(deltas, counts) == pytest.approx(2, abs=0.05)
    volumes = [c * d**2 for c, d in zip(counts, deltas)]
    assert min(volumes) > 0.7  # the union stays a fat set: nearly the disc of radius ½ (area π/4 ≈ 0.785)
    counts3 = solve.tube_counts(3, [1 / 8, 1 / 16, 1 / 32], "bush", CPU)
    assert solve.dimension_slope([1 / 8, 1 / 16, 1 / 32], counts3) > 2.7  # tending to 3 (Wang–Zahl: it must)
