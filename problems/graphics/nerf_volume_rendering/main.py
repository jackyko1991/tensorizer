"""Main loop: analytic scene → training photos → train two radiance fields (with / without encoding) → novel view."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors

SEED = 0
DEPTH = 3  # hidden layers
N_SNAP = 25


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    res, n_views, n_freq = params["resolution"], params["n_views"], params["n_freq"]
    n_steps = 30 if fast else params["steps"]
    n_samples, batch = params["n_samples"], params["batch"]
    torch.manual_seed(SEED)

    sc = tensors.scene(dev)
    cams = tensors.training_cameras(n_views, res, dev)
    o, d = cams["origins"].reshape(-1, 3), cams["dirs"].reshape(-1, 3)  # (V·P, 3) every training ray
    target, _ = tensors.render_scene(sc, o, d)  # (V·P, 3) the training photos, pixel by pixel
    nov = tensors.novel_camera(n_views, res, dev)
    o_n, d_n = nov["origins"][0], nov["dirs"][0]  # (P, 3)
    rgb_n, t_true = tensors.render_scene(sc, o_n, d_n)  # held-out photo and true depth
    print(f"scene: sphere + checkered cube + ball, {n_views} training views of {res}×{res} px = {o.shape[0]:,} "
          f"rays; sampled on t ∈ [{tensors.NEAR:.1f}, {tensors.FAR:.1f}] with {n_samples} samples each")  # fmt: skip

    runs = {}
    for name, L in [("with encoding", n_freq), ("without encoding", 0)]:
        in_dim = 3 + 6 * L
        layers = tensors.init_mlp(in_dim, params["width"], DEPTH, SEED, dev)
        n_par = sum(p.numel() for p in tensors.parameters(layers))
        t0 = time.perf_counter()
        hist = solve.train(layers, L, o, d, target, (o_n, d_n, rgb_n), n_steps, params["lr"], batch, n_samples,
                           5 if fast else N_SNAP, SEED, label=f"steps ({name})")  # fmt: skip
        dt = time.perf_counter() - t0
        last = hist["snaps"][-1]
        hit = torch.isfinite(t_true).cpu()
        depth_err = (last["depth"][hit] - t_true.cpu()[hit]).abs().median().item()
        print(f"{name:>17} (L = {L}, input {in_dim:2d} → {DEPTH}×{params['width']} → 4, {n_par:,} weights): "
              f"{n_steps} steps × {batch} rays in {dt:.1f} s")  # fmt: skip
        print(f"{'':>19}PSNR training view {last['psnr_train']:.1f} dB, held-out {hist['snaps'][0]['psnr']:.1f} → "
              f"{last['psnr']:.1f} dB, median depth error {depth_err:.3f}")  # fmt: skip
        runs[name] = hist

    t_far = torch.where(torch.isfinite(t_true), t_true, torch.full_like(t_true, tensors.FAR))
    return plot.animate(rgb_n.cpu(), t_far.cpu(), runs, res, n_freq, out_dir / "nerf_volume_rendering.gif")


if __name__ == "__main__":
    cli(run, __file__)
