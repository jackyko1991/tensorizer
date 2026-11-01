"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    torch.manual_seed(0)
    x = tensors.grid(params["n_pixels"], dev)  # (N,) pixel centres = detector bins
    f = tensors.shepp_logan(x, tensors.ellipse_params(dev))  # (N, N)
    theta = tensors.angles(params["n_angles"], dev)  # (A,)

    sino = solve.scan(f, theta, x)  # (A, S)
    sino = sino + params["noise"] * sino.max() * torch.randn_like(sino)  # optional detector noise
    plain, filtered = solve.per_angle_backprojections(sino, theta, x, params["filter"])  # (A, N, N) each
    recon = equation.backproject(filtered, theta)

    counts = [8, 16, 32, 64, 128, 256]
    errors = solve.error_vs_angles(f, x, counts, params["filter"])

    dx = float(x[1] - x[0])
    mass = (sino.sum(1) * dx).cpu()
    print(f"image mass ∫f dA = {float(f.sum()) * dx * dx:.4f};  ∫p ds per angle: {mass.min():.4f} … {mass.max():.4f}")
    print(f"sinogram {tuple(sino.shape)}, filter {params['filter']}, noise {params['noise']:.3f}")
    print(f"NRMSE of the FBP reconstruction with {len(theta)} angles: {solve.nrmse(recon, f):.3f}")
    for a, e in zip(counts, errors):
        print(f"  {a:4d} angles  NRMSE {e:.3f}")
    return plot.animate(
        f, x, theta, sino, plain, filtered, counts, errors, solve.nrmse(recon, f), out_dir / "radon_ct.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
