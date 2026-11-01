"""Main loop: parameters → two views and noisy matches → DLT and batched RANSAC → panorama → plot."""

import math
import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import dlt, transfer_error

SIZE = (192, 144)  # image width, height in pixels
FOCAL = 170.0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_hyp = 200 if fast_mode() else params["hypotheses"]
    Hs, imgs, H_true = tensors.views(SIZE, FOCAL, dev)
    x1, x2, is_out = tensors.matches(Hs, SIZE, params["matches"], params["noise"], params["outliers"], seed=0)
    eps = float(is_out.float().mean())
    print(f"two {SIZE[0]}×{SIZE[1]} views of a planar poster; {len(x1)} matches, {int(is_out.sum())} of them wrong")
    print("true view-1 → view-2 homography (H₃₃ = 1):")
    for row in H_true.tolist():
        print("   " + "  ".join(f"{v:9.4f}" for v in row))

    H_all = dlt(x1[None], x2[None])[0]  # least squares on every match, outliers included
    p_good = (1 - eps) ** 4
    need = math.log(0.001) / math.log(1 - p_good) if p_good < 1 else 1
    print(f"a random 4-match sample is outlier-free with probability {p_good:.3f}; {need:.0f} samples give 99.9% odds")
    start = time.perf_counter()
    H, inliers, counts = solve.ransac(x1, x2, n_hyp, params["threshold"], torch.Generator().manual_seed(1))
    elapsed = time.perf_counter() - start
    print(f"RANSAC: {n_hyp} hypotheses as one batched SVD of shape ({n_hyp}, 8, 9) in {1000 * elapsed:.0f} ms")
    print(f"best hypothesis has {int(counts.max())} inliers; median hypothesis {int(counts.median())}")
    tp = int((inliers & ~is_out).sum())
    print(
        f"inliers kept {inliers.sum().item()}: {tp} true matches, {int((inliers & is_out).sum())} wrong ones "
        f"(recall {tp / int((~is_out).sum()):.2f})"
    )

    e_ransac = transfer_error(H[None], x1, x2)[0]
    e_all = transfer_error(H_all[None], x1, x2)[0]
    rms = float(e_ransac[inliers].pow(2).mean().sqrt())
    print(
        f"RMS reprojection error on the inliers: {rms:.2f} px (noise σ = {params['noise']} px per axis in both "
        f"images, so ≈ 2σ = {2 * params['noise']:.2f} px expected)"
    )
    ce_r, ce_a = solve.corner_error(H, H_true, SIZE), solve.corner_error(H_all, H_true, SIZE)
    print(f"corner error vs the true H: RANSAC + DLT {ce_r:.2f} px, plain DLT on all matches {ce_a:.1f} px")

    pano_r, origin = solve.stitch(imgs, H)
    pano_a, origin_a = solve.stitch(imgs, H_all)
    print(f"panorama: {pano_r.shape[1]}×{pano_r.shape[0]} pixels from two {SIZE[0]}×{SIZE[1]} views")
    return plot.figure(
        imgs,
        x1,
        x2,
        is_out,
        inliers,
        pano_r,
        pano_a,
        e_ransac,
        e_all,
        params["threshold"],
        (ce_r, ce_a),
        out_dir / "homography_stitching.png",
    )


if __name__ == "__main__":
    cli(run, __file__)
