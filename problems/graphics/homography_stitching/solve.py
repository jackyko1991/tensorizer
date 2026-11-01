"""Robust estimation by batched RANSAC, then warping and feather-blending the two views into one panorama."""

import torch
import torch.nn.functional as F

from .equation import apply_homography, dlt, transfer_error


def ransac(x1: torch.Tensor, x2: torch.Tensor, n_hyp: int, threshold: float, gen: torch.Generator, rounds: int = 2):
    """Fit n_hyp homographies to random 4-match samples at once, keep the one with most inliers, then refit by DLT
    on all its inliers (repeated `rounds` times). x1, x2 (N, 2) → H (3, 3), inliers (N,) bool, counts (n_hyp,)."""
    n = x1.shape[0]
    sample = torch.rand(n_hyp, n, generator=gen).argsort(dim=1)[:, :4].to(x1.device)  # (B, 4) distinct matches
    H_all = dlt(x1[sample], x2[sample])  # (B, 3, 3): one batched SVD for every hypothesis
    err = transfer_error(H_all, x1, x2)  # (B, N)
    counts = (err < threshold).sum(dim=1)  # consensus of each hypothesis
    best = counts.argmax()
    inliers = err[best] < threshold
    H = H_all[best]
    for _ in range(rounds):  # least-squares refit on the consensus set, then re-classify
        H = dlt(x1[inliers][None], x2[inliers][None])[0]
        inliers = transfer_error(H[None], x1, x2)[0] < threshold
    return H, inliers, counts


def sample_image(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of img (H, W, 3) at pixel coordinates uv (h, w, 2) → (h, w, 3); zero outside."""
    h, w = img.shape[:2]
    grid = torch.stack([(2 * uv[..., 0] + 1) / w - 1, (2 * uv[..., 1] + 1) / h - 1], -1)  # align_corners=False
    out = F.grid_sample(img.permute(2, 0, 1)[None], grid[None], mode="bilinear", align_corners=False)
    return out[0].permute(1, 2, 0)


def feather(uv: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Blending weight: distance (in pixels) to the nearest image border, 0 outside the image.  (h, w, 2) → (h, w)"""
    w, h = size
    d = torch.minimum(
        torch.minimum(uv[..., 0] + 0.5, w - 0.5 - uv[..., 0]), torch.minimum(uv[..., 1] + 0.5, h - 0.5 - uv[..., 1])
    )
    return d.clamp(min=0)


def stitch(imgs: torch.Tensor, H12: torch.Tensor, pad_colour: float = 1.0):
    """Panorama in the frame of view 1: every canvas pixel p looks up view 1 at p and view 2 at H12 p, and the
    two are blended with feathering weights. imgs (2, H, W, 3) → panorama (h, w, 3)."""
    h, w = imgs.shape[1:3]
    size = (w, h)
    corners = torch.tensor([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=imgs.dtype, device=imgs.device)
    warped = apply_homography(torch.linalg.inv(H12)[None], corners)[0]  # view-2 corners in view-1 coordinates
    allc = torch.cat([corners, warped])
    limit = torch.tensor([2.0 * w, 2.0 * h], device=imgs.device)  # a wild H must not ask for a giant canvas
    lo = torch.maximum(allc.min(0).values.floor(), -limit)
    hi = torch.minimum(allc.max(0).values.ceil(), limit + torch.tensor([w, h], device=imgs.device))
    us = torch.arange(int(lo[0]), int(hi[0]) + 1, device=imgs.device, dtype=imgs.dtype)
    vs = torch.arange(int(lo[1]), int(hi[1]) + 1, device=imgs.device, dtype=imgs.dtype)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    p1 = torch.stack([uu, vv], -1)  # (h', w', 2) canvas pixels in view-1 coordinates
    p2 = apply_homography(H12[None], p1.reshape(-1, 2))[0].reshape(p1.shape)
    w1, w2 = feather(p1, size), feather(p2, size)
    blend = (w1[..., None] * sample_image(imgs[0], p1) + w2[..., None] * sample_image(imgs[1], p2)) / (w1 + w2).clamp(
        min=1e-9
    )[..., None]
    covered = (w1 + w2) > 0
    pano = torch.where(covered[..., None], blend, torch.full_like(blend, pad_colour))
    return pano, (int(lo[0]), int(lo[1]))


def corner_error(H: torch.Tensor, H_true: torch.Tensor, size: tuple[int, int]) -> float:
    """Mean distance (pixels) between where H and the true homography send the four image corners."""
    w, h = size
    c = torch.tensor([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=H.dtype, device=H.device)
    return float((apply_homography(H[None], c) - apply_homography(H_true[None], c)).norm(dim=-1).mean())
