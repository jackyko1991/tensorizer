"""Main loop: synthetic scene → training images → random Gaussians → Adam on the photometric loss → held-out view."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import psnr, render

SEED = 0
N_SNAP = 30  # checkpoints shown in the animation


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    res, n_views = params["resolution"], params["n_views"]
    m = 60 if fast else params["n_gaussians"]
    iterations = 20 if fast else params["iterations"]
    torch.manual_seed(SEED)

    sc = tensors.scene(dev)
    cams = tensors.training_cameras(n_views, res, dev)
    novel = tensors.novel_camera(n_views, res, dev)
    gt = tensors.render_scene(sc, cams)  # (V, P, 3) the training photographs
    gt_novel = tensors.render_scene(sc, novel)  # (1, P, 3) never used for training
    background = torch.tensor(tensors.BACKGROUND, device=dev)
    print(f"scene: 3 spheres + 1 cube, ray cast into {n_views} training views and 1 held-out view of {res}×{res} px")

    g = tensors.init_gaussians(m, SEED, dev)
    n_params = sum(v.numel() for v in g.values())
    print(f"{m} Gaussians × 14 numbers (μ 3, log s 3, q 4, colour 3, opacity 1) = {n_params:,} parameters")

    t0 = time.perf_counter()
    hist = solve.train(g, cams, gt, novel, gt_novel, background, iterations, params["lr"], params["loss"],
                       params["densify"] == "on", 5 if fast else N_SNAP, SEED)  # fmt: skip
    t_train = time.perf_counter() - t0
    first, last = hist["snaps"][0], hist["snaps"][-1]
    if params["densify"] == "on":
        print(f"densification: {hist['n_split']} splits (faint Gaussians reused as smaller copies of busy ones)")
    print(f"Adam, {iterations} steps × 1 random view, {params['loss'].upper()} loss: "
          f"{hist['loss'][0]:.4f} → {hist['loss'][-1]:.4f}  in {t_train:.1f} s")  # fmt: skip
    print(f"PSNR training views {first['psnr_train']:.1f} → {last['psnr_train']:.1f} dB   ·   "
          f"held-out view {first['psnr_novel']:.1f} → {last['psnr_novel']:.1f} dB")  # fmt: skip

    with torch.no_grad():
        _, splats = render(g, novel, background)
        mean_bg = psnr(background.expand_as(gt_novel), gt_novel).item()
    opaque = (splats["opacity"] > 0.5).sum().item()
    print(f"baseline 'all white' on the held-out view: {mean_bg:.1f} dB   ·   {opaque} of {m} Gaussians end up opaque")
    return plot.animate(gt[0].cpu(), gt_novel[0].cpu(), hist, res, out_dir / "gaussian_splatting.gif")


if __name__ == "__main__":
    cli(run, __file__)
