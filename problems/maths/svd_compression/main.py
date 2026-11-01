"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import svd, tail_energy


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    size = 96 if fast_mode() else params["size"]
    rgb = tensors.test_image(size, dev)  # (3, H, W) in [0, 1]
    a = tensors.grey(rgb)  # (H, W)
    h, w = a.shape
    print(f"synthetic test image: {h}×{w} pixels, 3 colour channels")

    u, s, v = svd(a)  # (H, R), (R,), (W, R)
    u_c, s_c, v_c = svd(rgb)  # batched over channels: (3, H, R), (3, R), (3, W, R)
    print(
        f"grey image: rank {int((s > 1e-5 * s[0]).sum())} of {len(s)}, σ_1..5 = " + ", ".join(f"{x:.2f}" for x in s[:5])
    )
    print(f"orthonormal check: max |UᵀU − I| = {(u.mT @ u - torch.eye(u.shape[1], device=dev)).abs().max().item():.1e}")

    ks = [k for k in (1, 5, 20, 50, 100) if k < len(s)]
    direct = solve.direct_errors(a, u, s, v, ks)
    tail = tail_energy(s)
    total = tail[0].item()
    print("Eckart–Young:  k   ‖A − A_k‖_F²   Σ_{r>k} σ_r²   energy kept")
    for k, d in zip(ks, direct.tolist()):
        print(f"           {k:5d}   {d:12.5f}   {tail[k].item():12.5f}   {1 - tail[k].item() / total:9.3%}")

    rank_ks, psnr, ratio = solve.quality_curve(s_c, h, w)
    for target in (25, 30, 40):
        hit = (psnr >= target).nonzero()
        if len(hit):
            k = int(rank_ks[hit[0, 0]])
            print(f"colour PSNR ≥ {target} dB first at k = {k:3d}: compression ratio {ratio[k - 1].item():5.1f}×")

    k_max = min(params["k_max"], len(s))
    frames = solve.frame_ranks(k_max, params["frames"])
    return plot.animate(
        rgb, (u, s, v), (u_c, s_c, v_c), (rank_ks, psnr, ratio), frames, out_dir / "svd_compression.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
