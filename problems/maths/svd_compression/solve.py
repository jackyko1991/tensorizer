"""Decompose the images, check Eckart–Young, and tabulate quality against storage for every rank k."""

import torch

from problems._common.progress import steps

from .equation import compression_ratio, reconstruct, tail_energy


def direct_errors(a: torch.Tensor, u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, ks: list[int]) -> torch.Tensor:
    """‖A − A_k‖_F² by actually rebuilding A_k for each k in ks (grey image (H, W)): (len(ks),)."""
    out = torch.empty(len(ks), dtype=a.dtype, device=a.device)
    for n in steps(len(ks), "ranks"):
        out[n] = ((a - reconstruct(u, s, v, ks[n])) ** 2).sum()
    return out


def quality_curve(s_rgb: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For the colour image, every rank k = 1..R at once, from the singular values alone (Eckart–Young).

    s_rgb: (3, R). Returns ks (R,), PSNR in dB (R,) and the compression ratio (R,).
    """
    r = s_rgb.shape[-1]
    err = tail_energy(s_rgb).sum(dim=0)[1:]  # (R,): Σ_c Σ_{r>k} σ_{cr}², k = 1..R
    mse = err / (3 * h * w)
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-12))
    ks = torch.arange(1, r + 1)
    ratio = torch.tensor([compression_ratio(h, w, int(k)) for k in ks])
    return ks, psnr, ratio


def frame_ranks(k_max: int, n_frames: int) -> list[int]:
    """Ranks shown in the animation: roughly geometric from 1 to k_max, so the early layers get more frames."""
    ks = torch.logspace(0, torch.log10(torch.tensor(float(k_max))), n_frames).round().int().unique()
    return ks.tolist()
