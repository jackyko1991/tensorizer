"""Monte Carlo of decays → coincidences → sinogram, then MLEM (and FBP for comparison) reconstruction."""

import math

import torch

from problems._common.progress import steps

from . import tensors
from .equation import (
    annihilation_photons,
    back_project,
    forward_project,
    log_likelihood,
    lor_coordinates,
    mlem_step,
    positron_range,
    ring_hits,
)


def simulate_events(
    activity: torch.Tensor, n_decays: int, sigma: float, n_detectors: int, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Follow every decay to a detected coincidence.  Returns per-event tensors (E = n_decays):

    decay, annihilation (E, 2) · hits on the ring (E, 2, 2) · crystal centres (E, 2, 2) · sinogram bin (E,)
    · in_sinogram (E,) bool · energies (E, 2) · directions (E, 2, 2)
    """
    dev = activity.device
    n = activity.shape[0]
    pixel = tensors.WIDTH / n
    # 1. where the tracer decays: pick pixels in proportion to activity, then a random point inside the pixel
    j = torch.multinomial(activity.flatten().cpu(), n_decays, replacement=True, generator=generator).to(dev)
    jitter = (torch.rand((n_decays, 2), generator=generator).to(dev) - 0.5) * pixel
    decay = tensors.pixel_centres(n, dev).view(-1, 2)[j] + jitter
    # 2. positron range, 3. back-to-back photons at a random angle, 4. where they hit the ring
    annihilation = positron_range(decay, sigma, generator)
    phi = 2 * math.pi * torch.rand(n_decays, generator=generator).to(dev)
    directions, energies = annihilation_photons(phi)
    hits = ring_hits(annihilation, directions, tensors.RING_RADIUS)  # (E, 2, 2)
    # 5. the detector ring is made of n_detectors crystals: we only know WHICH crystal fired, so the LOR end is
    #    placed at a random point of that crystal (snapping to the crystal centre would leave sinogram bins
    #    unevenly filled → ring artefacts; real scanners remove those with a measured normalisation)
    width = 2 * math.pi / n_detectors
    crystal = torch.floor(torch.remainder(torch.atan2(hits[..., 1], hits[..., 0]), 2 * math.pi) / width)
    angle = (crystal + torch.rand(crystal.shape, generator=generator).to(dev)) * width
    crystals = tensors.RING_RADIUS * torch.stack([torch.cos(angle), torch.sin(angle)], -1)  # (E, 2, 2)
    # 6. bin the LOR into the sinogram (θ_k, s_l)
    theta, s = lor_coordinates(crystals[:, 0], crystals[:, 1])
    K, L, s_max = tensors.sinogram_bins(n)
    k = torch.clamp(torch.floor(theta / (math.pi / K)).long(), max=K - 1)
    ell = torch.floor((s + s_max) / (2 * s_max / L)).long()
    in_sino = (ell >= 0) & (ell < L)
    return {
        "decay": decay,
        "annihilation": annihilation,
        "directions": directions,
        "energies": energies,
        "hits": hits,
        "crystals": crystals,
        "bin": torch.where(in_sino, k * L + ell, 0),
        "in_sinogram": in_sino,
    }


def detect(
    events: dict, attenuation: torch.Tensor, n_bins: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep each coincidence with probability exp(-∫μ dl) along its LOR; return (kept mask (E,), sinogram (B,))."""
    survive = torch.rand(events["bin"].shape, generator=generator).to(attenuation.device) < attenuation[events["bin"]]
    kept = events["in_sinogram"] & survive
    return kept, torch.bincount(events["bin"][kept], minlength=n_bins).float()


def mlem(
    A: torch.Tensor, y: torch.Tensor, r: torch.Tensor, n_iter: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MLEM from a flat image.  Returns (images (n_iter+1, J), log-likelihood (n_iter+1,),
    total expected counts Σ_b A λ (n_iter+1,))."""
    sens = back_project(A, torch.ones_like(y))  # (J,) s_j = A_{bj} 1_b
    lam = torch.full_like(sens, ((y - r).sum().clamp(min=1) / sens.sum()).item())  # flat start, right total
    images, ll, totals = [lam], [], []
    for _ in steps(n_iter, "MLEM iterations"):
        lam = mlem_step(lam, A, y, r, sens)
        images.append(lam)
    for lam in images:
        y_bar = forward_project(A, lam, r)
        ll.append(log_likelihood(y, y_bar))
        totals.append((y_bar - r).sum())
    return torch.stack(images), torch.stack(ll), torch.stack(totals)


def ramp_filter(sino: torch.Tensor) -> torch.Tensor:
    """Filter every projection (rows of a (K, L) sinogram) with a ramp |ν| × Hann window, via the FFT."""
    L = sino.shape[1]
    nu = torch.fft.fftfreq(2 * L, device=sino.device)  # zero-padded to avoid wrap-around
    H = nu.abs() * 0.5 * (1 + torch.cos(2 * math.pi * nu))  # Hann apodisation tames the noise
    return torch.fft.ifft(torch.fft.fft(sino, n=2 * L, dim=1) * H, dim=1).real[:, :L]


def fbp(A_geo: torch.Tensor, y_corr: torch.Tensor, K: int, L: int) -> torch.Tensor:
    """Filtered back-projection of corrected data y_corr (B,), scaled so its forward projection has the same counts."""
    image = back_project(A_geo, ramp_filter(y_corr.view(K, L)).flatten())
    return image * y_corr.sum() / forward_project(A_geo, image, torch.zeros_like(y_corr)).sum()


def relative_error(images: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """‖λ - λ_true‖ / ‖λ_true‖ for each image in a batch (…, J)."""
    return (images - truth).norm(dim=-1) / truth.norm()
