"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    clean = tensors.test_image(params["size"], dev)  # (H, W)
    noisy = tensors.gaussian_noise(clean, params["noise"], seed=0)
    sp = tensors.salt_and_pepper(clean, params["salt_pepper"], seed=1)
    print(f"test image {tuple(clean.shape)}: gradient, disk, square, triangle and a chirp; noise σ = {params['noise']}")

    out, checks = solve.run_filters(clean, noisy, sp, params)
    print(
        f"Gaussian kernel {checks['kernel']}: separable (two 1D passes) vs 2D conv2d: {checks['separable_vs_2d']:.1e}"
    )
    print(f"unfold + einsum vs conv2d:                         {checks['unfold_vs_conv2d']:.1e}")
    print(f"FFT (convolution theorem) vs circular conv2d:      {checks['fft_vs_circular']:.1e}")
    print("time per Gaussian blur: " + ", ".join(f"{k} {v:.2f} ms" for k, v in checks["ms"].items()))
    for name, db in checks["psnr"].items():
        print(f"PSNR {name:15s} {db:5.1f} dB")
    top = out["mag"].max().item()
    print(f"Sobel magnitude: largest {top:.2f}; a sharp step of height h gives 4h, so here h ≈ {top / 4:.2f}")
    return plot.figure(out, checks["responses"], params, out_dir / "image_filtering.png")


if __name__ == "__main__":
    cli(run, __file__)
