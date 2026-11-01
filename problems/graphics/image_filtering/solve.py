"""Apply every filter to the test images and check the three ways of computing a convolution agree."""

import time

import torch

from . import tensors
from .equation import filter2d, filter2d_unfold, filter_fft, frequency_response, gradient, median, separable, unsharp


def psnr(img: torch.Tensor, ref: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]: higher is closer to the reference."""
    return (10 * torch.log10(1 / ((img - ref) ** 2).mean())).item()


def timed(fn, *args) -> tuple[torch.Tensor, float]:
    """Result and run time in ms (after one warm-up call, so first-call overhead is not counted)."""
    fn(*args)
    start = time.perf_counter()
    out = fn(*args)
    return out, 1000 * (time.perf_counter() - start)


def run_filters(clean: torch.Tensor, noisy: torch.Tensor, sp: torch.Tensor, p: dict) -> tuple[dict, dict]:
    """All filtered images (each (H, W)) plus the checks and timings printed by main."""
    dev = clean.device
    g, g2 = tensors.gaussian_1d(p["sigma"], dev), tensors.gaussian_2d(p["sigma"], dev)
    kx, ky = tensors.sobel(dev)
    k_box = tensors.box(p["box"], dev)
    k_log = tensors.laplacian_of_gaussian(p["sigma"], dev)

    out = {"noisy": noisy, "box": filter2d(noisy, k_box)}
    out["gauss"], t_sep = timed(separable, noisy, g)
    gauss_2d, t_2d = timed(filter2d, noisy, g2)
    out["sharp"] = unsharp(noisy, g, p["amount"])
    out["gx"], out["gy"], out["mag"], out["angle"] = gradient(noisy, kx, ky)
    out["log"] = filter2d(noisy, k_log)
    out["sp"], out["sp_gauss"] = sp, separable(sp, g)
    out["sp_median"] = median(sp, p["median"])

    unfolded, t_unf = timed(filter2d_unfold, noisy, g2)
    fft, t_fft = timed(filter_fft, noisy, g2)
    checks = {
        "kernel": tuple(g2.shape),
        "separable_vs_2d": (out["gauss"] - gauss_2d).abs().max().item(),
        "unfold_vs_conv2d": (unfolded - gauss_2d).abs().max().item(),
        "fft_vs_circular": (fft - filter2d(noisy, g2, mode="circular")).abs().max().item(),
        "ms": {"2D conv2d": t_2d, "separable": t_sep, "unfold+einsum": t_unf, "FFT": t_fft},
        "psnr": {
            "salt & pepper": psnr(sp, clean),
            "s&p + Gaussian": psnr(out["sp_gauss"], clean),
            "s&p + median": psnr(out["sp_median"], clean),
        },
    }
    n = 128
    sharpen = -p["amount"] * g2
    sharpen[g2.shape[0] // 2, g2.shape[1] // 2] += 1 + p["amount"]  # δ + α(δ − G) as one kernel
    lap = frequency_response(k_log, n)
    responses = {
        f"box {p['box']}×{p['box']}": frequency_response(k_box, n),
        f"Gaussian σ = {p['sigma']:g}": frequency_response(g2, n),
        "unsharp mask": frequency_response(sharpen, n),
        "LoG (scaled)": lap / lap.abs().max(),  # −k² Ĝ: a band-pass that flips the sign
    }
    return out, {**checks, "responses": responses}
