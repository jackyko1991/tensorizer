"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import encode, encode_fft


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = params["size"]
    rho = tensors.shepp_logan(n, dev)  # (N, N) proton density
    E = tensors.encoding_matrix(n, dev)  # (N, N) complex
    k, dwell = tensors.readout_k(n, fov=0.24, gradient=10e-3)
    print(f"readout: G = 10 mT/m, FOV = 24 cm -> dwell time {dwell * 1e6:.2f} us, k from {k[0]:.1f} to {k[-1]:.1f} 1/m")

    s = encode(rho, E)  # (N, N) the measured signal, written as the encoding-matrix einsum
    print(f"max |einsum encoding - FFT| = {(s - encode_fft(rho)).abs().max():.2e}")

    masks, images = solve.filling(s, params["n_frames"])
    print(f"full sampling: max |reconstruction - phantom| = {(images[-1] - rho).abs().max():.2e}")
    results = solve.schemes(s, E, params["R"], params["lowpass"], params["random_frac"], params["lam"])
    for name, mask, zf, reg in results:
        print(
            f"{name:16s} lines {int(mask[:, 0].sum()):3d}/{n}  RMSE zero-filled {solve.rmse(zf, rho):.3f}"
            f"  regularised {solve.rmse(reg, rho):.3f}"
        )
    return plot.animate(s, rho, masks, images, results, out_dir / "mri_kspace.gif")


if __name__ == "__main__":
    cli(run, __file__)
