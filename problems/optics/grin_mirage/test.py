"""Physics checks, written first (TDD): what any correct graded-index code must satisfy.

Run with `pytest problems/optics/grin_mirage/test.py`. Rays are straight in a uniform medium, the Snell invariant n cos
θ is conserved along each ray, |p| = n, and the mirage image is inverted.
"""

import torch

from problems._common.integrators import integrate

from . import equation, solve, tensors


def _trace(dn: float, n_rays: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    angles = tensors.viewing_angles(10.0, n_rays)
    path = solve.solve(tensors.initial_state(angles, 1.6, 400.0, dn, 0.3), dn, 0.3, 400.0)
    return angles, path


def test_rays_stay_straight_in_uniform_medium():
    angles, path = _trace(0.0)
    for s in range(1, len(path), 50):
        step = path[s] - path[0]  # (R, 2)
        moving = step.norm(dim=-1) > 0
        slope = step[moving, 1] / step[moving, 0]
        assert torch.allclose(slope, -torch.tan(angles[moving]), atol=1e-10)  # heading −x at angle α


def test_snell_invariant_conserved_along_rays():
    dn, h = 3e-5, 0.3
    angles, path = _trace(dn)
    invariant = equation.snell_invariant(path, dn, h)  # (S − 1, R)
    for r in range(len(angles)):
        moving = (path[1:, r] != path[:-1, r]).any(-1)  # ignore the frozen end of each ray
        values = invariant[moving, r][:-1]  # and the last step, which may poke below the road where n is flat
        assert (values.max() - values.min()).item() < 1e-8  # n cos θ conserved ...
    n_along = equation.index(path[:, 2, 1], dn, h)  # ray 2 (−7 mrad) dives into the hot layer
    assert (n_along.max() - n_along.min()).item() > 1e-6  # ... although n itself changes far more


def test_direction_norm_equals_index():
    dn, h = 3e-5, 0.3
    angles = tensors.viewing_angles(10.0, 9)
    traj = integrate(lambda y, s: equation.rhs(y, dn, h), tensors.initial_state(angles, 1.6, 400.0, dn, h), 1.0, 150)
    p_sq = equation.direction_norm_sq(traj[:, :, 1])  # p_j p_j
    n_sq = equation.index(traj[:, :, 0, 1], dn, h) ** 2
    assert torch.allclose(p_sq, n_sq, atol=1e-10)


def test_mirage_image_is_inverted():
    _, path = _trace(3e-5, n_rays=25)
    seen = solve.height_at_object(path)
    assert torch.isnan(seen[:3]).all()  # the steepest rays reach the road
    hit = seen[~torch.isnan(seen)]
    diffs = hit[1:] - hit[:-1]
    assert (diffs[-5:] > 0).all()  # direct view: looking higher shows higher parts (upright)
    assert (diffs[:2] < 0).all()  # mirage: looking lower shows higher parts (inverted)
