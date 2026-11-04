"""Main loop: parameters → tensors → the two Ricci flows (and surgery) → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import curvature_spread, profile, s3_curvatures, total_curvature

SURGERY_SCALE = 0.05  # neck radius at which we stop and cut
END_RADIUS = 0.35  # stop the capped piece when its largest S² has this radius (it is about to vanish)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = 48 if fast_mode() else params["cells"] // 2 * 2
    n_frames = 8 if fast_mode() else 60

    # 1. lumpy 2-spheres under normalised Ricci flow
    x, w, lap = tensors.sphere_grid(n, dev)  # (N,), (N,), (N, N)
    u0 = tensors.lumpy_factors(x)  # (B, N)
    surf = solve.flow_surface(u0, w, lap, 0.3 if fast_mode() else params["t_surface"], n_save=n_frames)
    gb = torch.stack([total_curvature(k, u, w) for k, u in zip(surf["K"], surf["u"])])  # (S, B)
    spread = torch.stack([curvature_spread(k) for k in surf["K"]])  # (S, B)
    print(f"surfaces: {u0.shape[0]} lumpy spheres on {n} cells, normalised Ricci flow to t = {surf['t'][-1]:.2f}")
    print(
        f"  Gauss–Bonnet ∫K dA / 4π over the whole run: {(gb / (4 * math.pi)).min():.13f} … "
        f"{(gb / (4 * math.pi)).max():.13f}"
    )
    print(
        f"  curvature spread (max K − min K)/mean K: {', '.join(f'{v:.2f}' for v in spread[0].tolist())} → "
        f"{', '.join(f'{v:.1e}' for v in spread[-1].tolist())};  area drift "
        f"{((surf['area'][-1] / surf['area'][0]) - 1).abs().max():.1e}"
    )

    # 2. a dumbbell 3-sphere: neckpinch, surgery, and the capped piece rounding out
    grid = tensors.s3_grid(n, dev)  # (x, x_η, x_ηη, h)
    t_max = 0.01 if fast_mode() else 1.0
    pre = solve.flow_s3(tensors.dumbbell(params["neck"], grid[0]), grid, t_max, SURGERY_SCALE, 25, END_RADIUS)
    pinched = pre["neck"][-1].item() < SURGERY_SCALE
    print(f"3-sphere: dumbbell with neck radius ε = {params['neck']}, {len(pre['t'])} Euler steps")
    psi0, psi1 = profile(pre["u"][0], grid)[1], profile(pre["u"][-1], grid)[1]
    post = None
    if pinched:
        t_pinch = pre["t"][-1].item()
        neck = pre["neck"]
        i = int(torch.searchsorted(-neck, torch.tensor(-0.1, dtype=neck.dtype)))
        slope = ((neck[-2] ** 2 - neck[i] ** 2) / (pre["t"][-2] - pre["t"][i])).item()
        print(
            f"  neckpinch: neck radius {neck.max():.3f} → {neck[-1]:.3f} at t = {t_pinch:.4f}, while the largest "
            f"S² only went {psi0.max():.2f} → {psi1.max():.2f}"
        )
        print(f"  d(ψ_neck²)/dt over the last stretch = {slope:.2f}  (a shrinking cylinder S² × R: −2)")
        capped = solve.surgery(pre["u"][-1], grid)
        post = solve.flow_s3(capped, grid, 0.002 if fast_mode() else 1.0, 0.0, 400, END_RADIUS)
        post["t"] += t_pinch
        post["t_save"] += t_pinch
        ratio = []
        for u in post["u"]:
            k = torch.cat(s3_curvatures(u, grid), dim=1)
            ratio.append((k.max() / k.min()).item() if k.min() > 0 else float("nan"))
        post["ratio"] = torch.tensor(ratio, dtype=torch.float64)
        print(f"  surgery at ψ = {SURGERY_SCALE}: cut the neck, glue round caps; each piece then flows on its own")
        print(
            f"  capped piece: K_max/K_min = {post['ratio'][~post['ratio'].isnan()][0]:.2f} → {post['ratio'][-1]:.2f}"
            f" by t = {post['t'][-1]:.3f} (radius {END_RADIUS}): it becomes round as it shrinks away"
        )
    elif psi1.max().item() < END_RADIUS:
        print(
            f"  no neckpinch: the neck opened up and the whole sphere rounded out (radius {END_RADIUS} at "
            f"t = {pre['t'][-1]:.3f})"
        )
    else:
        print(f"  stopped at t = {pre['t'][-1]:.3f} before any pinch (neck radius {pre['neck'][-1]:.3f})")
    return plot.animate(x, surf, spread, gb, grid, pre, post, n_frames, out_dir / "poincare_ricci_flow.gif")


if __name__ == "__main__":
    cli(run, __file__)
