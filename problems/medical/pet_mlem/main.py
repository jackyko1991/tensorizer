"""Main loop: phantom → simulated coincidences → sinogram → MLEM / FBP reconstruction → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(0)
    n = params["grid"]
    K, L, _ = tensors.sinogram_bins(n)

    activity, mu = tensors.phantom(n, dev)  # (n, n) each
    A_geo = tensors.system_matrix(n, dev)  # (K·L, n·n) detection probabilities without attenuation
    if params["attenuation"] == "on":
        att = torch.exp(-tensors.path_lengths(A_geo, n) @ mu.flatten())  # (B,) survival exp(-∫μ dl) per LOR
    else:
        att = torch.ones(K * L, device=dev)
    A = att[:, None] * A_geo  # attenuation folded into the system model

    n_decays = params["kilo_decays"] * 1000
    events = solve.simulate_events(activity, n_decays, params["positron_range"] / 10, params["n_detectors"], gen)
    kept, trues = solve.detect(events, att, K * L, gen)
    bg = params["background"]
    r = bg / (1 - bg) * trues.sum() * tensors.background_shape(n, dev)  # (B,) expected randoms + scatter
    y = trues + torch.poisson(r.cpu(), generator=gen).to(dev)  # measured sinogram

    images, ll, totals = solve.mlem(A, y, r, params["n_iter"])
    recon_fbp = solve.fbp(A_geo, (y - r) / att, K, L)
    truth = activity.flatten() * n_decays / activity.sum()  # expected decays per pixel
    err = solve.relative_error(images, truth)
    err_fbp = solve.relative_error(recon_fbp, truth).item()

    print(f"{n_decays} decays → {int(kept.sum())} detected coincidences ({100 * kept.float().mean():.1f}%)")
    print(f"sinogram {K} angles × {L} distances, background {100 * (y.sum() - trues.sum()) / y.sum():.0f}% of counts")
    print(f"system matrix A {tuple(A.shape)}, image {n}×{n}")
    best = int(err.argmin())
    print(f"MLEM error: {err[1]:.3f} after 1 it, best {err[best]:.3f} at it {best}, {err[-1]:.3f} at it {len(err) - 1}")
    print(f"FBP error: {err_fbp:.3f}")
    rising = bool((ll.diff() >= -1e-6 * ll.abs()[1:]).all())  # float32 round-off allowed
    print(f"log-likelihood {ll[0]:.4g} → {ll[-1]:.4g}, never decreasing: {rising}")
    print(f"counts Σ Aλ = {totals[-1]:.0f} vs Σ (y - r) = {(y - r).sum():.0f}")
    return plot.animate(events, kept, y, activity, images, recon_fbp, err, err_fbp, n, out_dir / "pet_mlem.gif")


if __name__ == "__main__":
    cli(run, __file__)
