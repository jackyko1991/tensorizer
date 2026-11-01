"""Physics checks, written first (TDD): what any correct homography code must satisfy.

Run with `pytest problems/graphics/homography_stitching/test.py`. Each test is a mathematical invariant, not an
implementation detail: the DLT recovers an exact homography with zero transfer error from noise-free points, and RANSAC
rejects every outlier.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def normalised(H: torch.Tensor) -> torch.Tensor:
    H = H / H.norm()
    return H * torch.sign(H.flatten()[H.flatten().abs().argmax()])


def test_dlt_recovers_the_homography_exactly_from_noise_free_points():
    gen = torch.Generator().manual_seed(0)
    H_true = torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    H_true[2, :2] *= 0.01  # a mild perspective part, as for real pixel coordinates
    x1 = torch.rand(30, 2, generator=gen, dtype=torch.float64) * 200
    x2 = equation.apply_homography(H_true[None], x1)[0]
    for n in (4, 30):  # the minimal case and the over-determined one
        H = equation.dlt(x1[None, :n], x2[None, :n])[0]
        assert torch.allclose(normalised(H), normalised(H_true), atol=1e-9)
        assert equation.transfer_error(H[None], x1, x2).max() < 1e-7  # x2 ≅ H x1 exactly


def test_ransac_rejects_outliers_and_recovers_the_homography():
    size = (192, 144)
    Hs, _, H_true = tensors.views(size, 170.0, CPU)
    x1, x2, is_out = tensors.matches(Hs, size, 120, 0.5, 0.4, seed=3)
    H_naive = equation.dlt(x1[None], x2[None])[0]
    H, inliers, counts = solve.ransac(x1, x2, 2000, 3.0, torch.Generator().manual_seed(0))
    assert counts.shape == (2000,)
    assert not (inliers & is_out).any()  # no wrong match survives
    assert (inliers & ~is_out).sum() >= 0.9 * (~is_out).sum()  # and almost all good ones are kept
    assert solve.corner_error(H, H_true, size) < 1.5
    assert solve.corner_error(H_naive, H_true, size) > 10  # least squares alone is ruined by the outliers
