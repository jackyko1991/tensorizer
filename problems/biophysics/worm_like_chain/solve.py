"""Metropolis Monte Carlo for every force and every chain at once: local tangent rotations plus pivot moves.

One sweep = update the even segments, then the odd ones (a segment's energy change depends only on its two
neighbours, so all even segments can move together), then one pivot move per chain that turns the whole tail of
the chain at once, which relaxes the long, floppy shapes at low force much faster than local moves alone.
"""

import torch

from problems._common.progress import steps

from .equation import bending_energy, local_energy_change, local_field, relative_extension, tangent_correlation


def local_moves(t: torch.Tensor, K: float, f: torch.Tensor, step: float, gen: torch.Generator) -> torch.Tensor:
    """Try t_i → normalise(t_i + step·ξ) on the even, then the odd segments of every chain.   t: (B, M, N, 3)."""
    n = t.shape[2]
    for parity in (0, 1):
        noise = torch.randn(t.shape, generator=gen).to(t.device)
        t_new = torch.nn.functional.normalize(t + step * noise, dim=-1)  # a small random turn, still a unit vector
        dE = local_energy_change(local_field(t, K, f), t, t_new)  # (B, M, N)
        u = torch.rand(dE.shape, generator=gen).to(t.device)
        on_parity = (torch.arange(n, device=t.device) % 2 == parity)[None, None, :]
        accept = (u < torch.exp(-dE)) & on_parity
        t = torch.where(accept[..., None], t_new, t)
    return t


def rotation_matrices(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: R = I + sin θ [u]× + (1 - cos θ) [u]×², for unit axes u (..., 3) and angles θ (...).   (..., 3, 3)"""
    ux, uy, uz = axis.unbind(-1)
    zero = torch.zeros_like(ux)
    cross = torch.stack([zero, -uz, uy, uz, zero, -ux, -uy, ux, zero], dim=-1).reshape(*axis.shape, 3)
    eye = torch.eye(3, device=axis.device).expand_as(cross)
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return eye + s * cross + (1 - c) * cross @ cross


def pivot_move(t: torch.Tensor, K: float, f: torch.Tensor, max_angle: float, gen: torch.Generator) -> torch.Tensor:
    """Rotate segments i, i+1, …, N-1 of each chain about a random axis: t'_j = R t_j for j ≥ i."""
    B, M, n, _ = t.shape
    dev = t.device
    axis = torch.nn.functional.normalize(torch.randn(B, M, 3, generator=gen), dim=-1).to(dev)
    angle = (max_angle * (2 * torch.rand(B, M, generator=gen) - 1)).to(dev)
    i = torch.randint(0, n, (B, M, 1), generator=gen).to(dev)
    R = rotation_matrices(axis, angle)  # (B, M, 3, 3)
    turned = torch.einsum("bmkl,bmil->bmik", R, t)  # every segment rotated ...
    t_new = torch.where((torch.arange(n, device=dev) >= i)[..., None], turned, t)  # ... but only the tail kept
    dE = bending_energy(t_new, K, f) - bending_energy(t, K, f)  # (B, M)
    accept = torch.rand(B, M, generator=gen).to(dev) < torch.exp(-dE)
    return torch.where(accept[:, :, None, None], t_new, t)


def simulate(
    t0: torch.Tensor, K: float, f: torch.Tensor, sweeps: int, n_records: int, show: list[int], seed: int = 0
) -> dict:
    """Run the Monte Carlo; the first half of the sweeps equilibrates and the second half is measured.

    show: force indices whose chain m = 0 is stored for the animation.
    """
    gen = torch.Generator().manual_seed(seed)
    step = min(0.5, 1.5 / K**0.5)  # turn size matched to the thermal wobble of one segment, ~1/√(2K)
    every = max(1, sweeps // n_records)
    t = t0.clone()
    z = torch.empty(sweeps + 1, *t.shape[:2], device=t.device)  # (S+1, B, M) extension after every sweep
    z[0] = relative_extension(t)
    snapshots, record_sweeps = [t[show, 0].clone()], [0]
    max_lag = min(60, t.shape[2] - 1)
    corr, n_corr = torch.zeros(max_lag + 1, device=t.device), 0

    for s in steps(sweeps, "MC sweeps"):
        t = local_moves(t, K, f, step, gen)
        t = pivot_move(t, K, f, 0.6, gen)
        z[s + 1] = relative_extension(t)
        if (s + 1) % every == 0 or s == sweeps - 1:
            snapshots.append(t[show, 0].clone())
            record_sweeps.append(s + 1)
        if s >= sweeps // 2 and s % 10 == 0:  # tangent correlations at the lowest force, measured half
            corr += tangent_correlation(t[0], max_lag)
            n_corr += 1

    return {
        "t": t,  # (B, M, N, 3) final tangents
        "z": z,  # (S+1, B, M) relative extension history
        "z_mean": z[sweeps // 2 + 1 :].mean(dim=(0, 2)),  # (B,) measured ⟨z⟩/L at every force
        "z_err": z[sweeps // 2 + 1 :].mean(0).std(-1) / t.shape[1] ** 0.5,  # (B,) error from chain-to-chain spread
        "snapshots": torch.stack(snapshots),  # (K, len(show), N, 3)
        "record_sweeps": torch.tensor(record_sweeps),  # (K,)
        "correlation": corr / max(n_corr, 1),  # (max_lag+1,) ⟨t_i·t_{i+n}⟩ at the lowest force
    }
