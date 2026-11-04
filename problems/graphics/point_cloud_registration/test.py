"""Geometry checks, written first (TDD): what any correct point-cloud registration code must satisfy.

Run with `pytest problems/graphics/point_cloud_registration/test.py`. Each test is a mathematical fact, not an
implementation detail: Kabsch recovers a rigid motion exactly, a transform composed with its inverse is the identity,
plane normals are constant and sphere normals radial, a voxel grid keeps one centroid per occupied voxel, far
outliers have large neighbour distances, FPFH descriptors do not change under a rigid motion, and ICP converges.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

D = torch.float64


def _random_rigid(b: int, seed: int, angle: float = math.pi, trans: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    axis = torch.randn(b, 3, generator=gen, dtype=D)
    omega = axis / axis.norm(dim=1, keepdim=True) * angle * torch.rand(b, 1, generator=gen, dtype=D)
    return equation.rodrigues(omega), trans * torch.randn(b, 3, generator=gen, dtype=D)


def _sphere(n: int, seed: int, radius: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=gen, dtype=D)
    return radius * x / x.norm(dim=1, keepdim=True)


def _wavy_surface(n: int, seed: int, half: float = 0.5):
    """Random samples of the smooth surface z = 0.1 sin(8x) cos(7y) over a square of side 2·half, with normals."""
    gen = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=gen, dtype=D) - 0.5) * 2 * half
    x, y = xy[:, 0], xy[:, 1]
    z = 0.1 * torch.sin(8 * x) * torch.cos(7 * y)
    zx, zy = 0.8 * torch.cos(8 * x) * torch.cos(7 * y), -0.7 * torch.sin(8 * x) * torch.sin(7 * y)
    n_ = torch.stack([-zx, -zy, torch.ones_like(x)], 1)
    return torch.stack([x, y, z], 1), n_ / n_.norm(dim=1, keepdim=True)


def test_kabsch_recovers_a_rigid_transform_exactly():
    R, t = _random_rigid(16, 0)
    P = torch.randn(16, 50, 3, generator=torch.Generator().manual_seed(1), dtype=D)  # (B, N, 3)
    Q = equation.transform(R, t, P)
    R_est, t_est = equation.kabsch(P, Q)
    assert torch.allclose(R_est, R, atol=1e-10)
    assert torch.allclose(t_est, t, atol=1e-10)
    assert torch.allclose(torch.linalg.det(R_est), torch.ones(16, dtype=D), atol=1e-12)  # a rotation, no reflection


def test_transform_composed_with_its_inverse_is_identity():
    R, t = _random_rigid(8, 2)
    Ri, ti = equation.invert(R, t)
    R1, t1 = equation.compose(R, t, Ri, ti)
    R2, t2 = equation.compose(Ri, ti, R, t)
    eye = torch.eye(3, dtype=D).expand(8, 3, 3)
    for Rc, tc in [(R1, t1), (R2, t2)]:
        assert torch.allclose(Rc, eye, atol=1e-12)
        assert tc.abs().max() < 1e-12
    assert torch.allclose(equation.rotation_error_deg(R, R), torch.zeros(8, dtype=D), atol=1e-5)


def test_plane_normals_are_constant_and_flat():
    gen = torch.Generator().manual_seed(3)
    uv = torch.rand(800, 2, generator=gen, dtype=D)
    a, b = torch.tensor([1.0, 0.0, 0.5], dtype=D), torch.tensor([0.0, 1.0, -0.3], dtype=D)
    X = uv[:, :1] * a + uv[:, 1:] * b  # points on a tilted plane
    n_true = torch.linalg.cross(a, b) / torch.linalg.cross(a, b).norm()
    viewpoint = torch.tensor([0.5, 0.5, 5.0], dtype=D)
    normals, curvature = equation.estimate_normals(X, k=12, viewpoint=viewpoint)
    assert torch.allclose(normals, n_true.expand_as(normals), atol=1e-9)  # n_true faces +z, toward the viewpoint
    assert curvature.max() < 1e-12


def test_sphere_normals_are_radial_and_face_the_viewpoint():
    X = _sphere(3000, 4)
    normals, curvature = equation.estimate_normals(X, k=15, viewpoint=torch.zeros(3, dtype=D))
    cos = (normals * X).sum(1)  # X is the unit radial direction
    assert (cos < -0.995).all()  # radial, and flipped toward the viewpoint at the centre
    assert curvature.mean() < 0.01  # nearly flat at the neighbourhood scale


def test_voxel_grid_keeps_one_centroid_per_occupied_voxel():
    gen = torch.Generator().manual_seed(5)
    X = torch.rand(2000, 3, generator=gen, dtype=D) * 1.3 - 0.2
    size = 0.25
    centroids, counts = equation.voxel_downsample(X, size)
    groups: dict[tuple, list] = {}  # slow reference: a Python dictionary of voxels
    for p in X.tolist():
        groups.setdefault(tuple(math.floor(c / size) for c in p), []).append(p)
    ref = torch.tensor([torch.tensor(v, dtype=D).mean(0).tolist() for v in groups.values()], dtype=D)
    assert len(centroids) == len(groups) and int(counts.sum()) == len(X)
    keys = torch.floor(centroids / size)
    assert len(torch.unique(keys, dim=0)) == len(centroids)  # one point per voxel, inside its voxel
    order = lambda P: P[torch.argsort(P[:, 0] * 1e6 + P[:, 1] * 1e3 + P[:, 2])]  # noqa: E731
    assert torch.allclose(order(centroids), order(ref), atol=1e-12)


def test_statistical_outlier_removal_drops_far_outliers():
    gen = torch.Generator().manual_seed(6)
    X = torch.cat([torch.rand(1500, 2, generator=gen, dtype=D), torch.zeros(1500, 1, dtype=D)], 1)
    far = torch.rand(30, 3, generator=gen, dtype=D) * 4 - 1.5
    far[:, 2] = far[:, 2].sign() * (0.5 + far[:, 2].abs())  # at least 0.5 off the plane
    keep = equation.statistical_outlier_mask(torch.cat([X, far]), k=10, alpha=1.0)
    assert not keep[1500:].any()  # every injected outlier is dropped
    assert keep[:1500].float().mean() > 0.95  # almost every surface point stays


def test_knn_matches_full_distance_matrix():
    gen = torch.Generator().manual_seed(7)
    X, Y = torch.rand(300, 3, generator=gen, dtype=D), torch.rand(200, 3, generator=gen, dtype=D)
    d, idx = equation.knn(X, Y, k=4, chunk=64)
    d_ref, idx_ref = torch.cdist(X, Y).topk(4, largest=False)
    assert torch.equal(idx, idx_ref)
    assert torch.allclose(d, d_ref)


def test_fpfh_is_invariant_under_rigid_motion():
    X = tensors.sample_surface(*tensors.load_spot(), 3000, torch.Generator().manual_seed(8))[0]
    viewpoint = torch.tensor([0.0, 0.5, 3.0], dtype=D)
    N, _ = equation.estimate_normals(X, k=15, viewpoint=viewpoint)
    F = equation.fpfh(X, N, k=20)
    R, t = _random_rigid(1, 9)
    Xm = equation.transform(R[0], t[0], X)
    Nm, _ = equation.estimate_normals(Xm, k=15, viewpoint=equation.transform(R[0], t[0], viewpoint[None])[0])
    Fm = equation.fpfh(Xm, Nm, k=20)
    assert F.shape == (3000, 33)
    assert torch.allclose(F.sum(1), torch.full((3000,), 300.0, dtype=D))  # three histograms, each sums to 100
    assert (F - Fm).abs().max() < 1e-6


def test_hidden_points_are_removed():
    X = _sphere(4000, 10)
    eye = torch.tensor([0.0, 0.0, 4.0], dtype=D)
    seen = tensors.visible(X, X, eye)  # sphere normal = position
    cap = X[:, 2] > 1 / 4  # the part of a unit sphere visible from distance 4: cos θ > r/d
    assert (seen & ~cap).sum() == 0  # nothing behind the horizon is kept
    assert seen[cap].float().mean() > 0.9  # most of the visible cap is kept


def test_ransac_recovers_transform_from_half_wrong_matches():
    gen = torch.Generator().manual_seed(11)
    P = torch.rand(400, 3, generator=gen, dtype=D)
    R, t = _random_rigid(1, 12)
    Q = equation.transform(R[0], t[0], P) + 0.002 * torch.randn(400, 3, generator=gen, dtype=D)
    match = torch.arange(400)
    match[200:] = torch.randint(0, 400, (200,), generator=gen)  # half the correspondences are wrong
    R_est, t_est, inliers = solve.ransac(P, Q[match], n_hypotheses=500, eps=0.01, gen=gen)
    assert equation.rotation_error_deg(R_est, R[0]) < 0.5
    assert (t_est - t[0]).norm() < 0.01
    assert inliers[:200].float().mean() > 0.95


def test_icp_converges_from_a_small_perturbation():
    target, normals = _wavy_surface(2000, 16)
    source = target[(target[:, :2].abs() < 0.35).all(1)]  # a subset of the target points: exact pairs exist
    gen = torch.Generator().manual_seed(17)
    axis = torch.randn(2, 3, generator=gen, dtype=D)
    R_true = equation.rodrigues(math.radians(3) * axis / axis.norm(dim=1, keepdim=True))
    t_true = 0.01 * torch.randn(2, 3, generator=gen, dtype=D)
    src = equation.transform(R_true, t_true, source.expand(2, -1, -1))
    Ri, ti = equation.invert(R_true, t_true)
    for method in ["point_to_point", "point_to_plane"]:
        eye = torch.eye(3, dtype=D).expand(2, 3, 3)
        R, t, hist = solve.icp(src, target, normals, eye, torch.zeros(2, 3, dtype=D), 15, method, max_dist=0.2)
        assert (equation.rotation_error_deg(R, Ri) < 1e-3).all()
        assert ((t - ti).norm(dim=1) < 1e-5).all()
        assert hist["rmse"][-1].max() < 1e-6  # every point back on its partner
        assert (hist["rmse"][-1] < hist["rmse"][0]).all()


@pytest.mark.slow
def test_point_to_plane_converges_in_fewer_iterations_than_point_to_point():
    target, normals = _wavy_surface(5000, 13)
    source = _wavy_surface(1000, 14, half=0.35)[0]  # other samples of the inner part: no exact point pairs
    gen = torch.Generator().manual_seed(15)
    axis = torch.randn(3, 3, generator=gen, dtype=D)
    R_true = equation.rodrigues(math.radians(6) * axis / axis.norm(dim=1, keepdim=True))  # 6° perturbations
    t_true = 0.03 * torch.randn(3, 3, generator=gen, dtype=D)
    src = equation.transform(R_true, t_true, source.expand(3, -1, -1))  # (3, N, 3): 3 independent problems
    eye = torch.eye(3, dtype=D).expand(3, 3, 3)
    iters = {}
    for method in ["point_to_point", "point_to_plane"]:
        R, t, hist = solve.icp(src, target, normals, eye, torch.zeros(3, 3, dtype=D), 60, method, max_dist=0.2)
        Ri, ti = equation.invert(R_true, t_true)  # the answer undoes the perturbation
        assert (equation.rotation_error_deg(R, Ri) < 0.5).all()
        assert ((t - ti).norm(dim=1) < 0.005).all()
        errs = torch.stack([equation.rotation_error_deg(Rk, Ri) for Rk in hist["R"]])  # (iters + 1, 4)
        iters[method] = (errs < 0.5).float().argmax(0)  # first iteration within 0.5°
    assert (iters["point_to_plane"] < iters["point_to_point"]).all()


@pytest.mark.slow
def test_full_pipeline_registers_the_two_scans():
    out = solve.register(tensors.DEFAULTS, seed=0)
    assert out["rot_err"] < 2.0
    assert out["trans_err"] < 0.02 * out["size"]
