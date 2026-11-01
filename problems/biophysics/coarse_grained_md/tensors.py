"""Build the concrete tensors: the native fold, the Gō-model parameters read off it, and the starting chains.

Units: lengths in Å, energies in ε (the depth of one native contact), mass 1, temperature in ε / k_B,
time in τ = Å √(m / ε).
"""

import math

import torch

from .equation import pair_distances

CONTACT_CUTOFF = 8.0  # Å: residues closer than this in the native fold form a native contact
MIN_SEQ_SEP = 3  # only pairs |i - j| ≥ 3 count as contacts (closer ones are held by bonds and angles)


def native_structure(n_helix: int, device: torch.device) -> torch.Tensor:
    """Cα beads of a small helical hairpin, shape (N, 3): two antiparallel right-handed helices and a 3-bead loop."""
    r, rise, turn, sep = 2.3, 1.5, math.radians(100.0), 8.5  # α-helix geometry, Å; 8.5 Å between the helix axes
    t = torch.arange(n_helix, dtype=torch.float32)
    phi = turn * t
    h1 = torch.stack([-sep / 2 + r * torch.cos(phi), r * torch.sin(phi), rise * t], -1)
    top = rise * (n_helix - 1)
    h2 = torch.stack([sep / 2 - r * torch.cos(phi), r * torch.sin(phi), top - rise * t], -1)
    # loop: straight line between the helix ends, lifted into an arch
    s = torch.linspace(0, 1, 5)[1:-1, None]  # 3 interior points
    loop = (1 - s) * h1[-1] + s * h2[0] + torch.tensor([0.0, 0.0, 3.5]) * torch.sin(math.pi * s)
    return torch.cat([h1, loop, h2]).to(device)


def go_model(native: torch.Tensor) -> dict:
    """Gō model: every bond length, bond angle and contact distance is set to its value in the native fold.

    Returns b0 (N-1,) bond lengths, theta0 (N-2,) bond angles, and for every pair p = (i_p, j_p) with
    j - i ≥ 3: i, j (P,) indices, r0 (P,) native distance, native (P,) 1.0 for a native contact else 0.0.
    """
    n = native.shape[0]
    r_native = pair_distances(native)  # (N, N)
    i, j = torch.triu_indices(n, n, offset=MIN_SEQ_SEP, device=native.device)  # each pair once
    r0 = r_native[i, j]
    b = native[1:] - native[:-1]  # (N-1, 3) bond vectors
    cos = (-(b[:-1] * b[1:]).sum(-1)) / (b[:-1].norm(dim=-1) * b[1:].norm(dim=-1))
    return {
        "b0": b.norm(dim=-1),
        "theta0": torch.arccos(cos),
        "i": i,
        "j": j,
        "r0": r0,
        "native": (r0 < CONTACT_CUTOFF).to(r0.dtype),
    }


def temperatures(t_min: float, t_max: float, n: int, device: torch.device) -> torch.Tensor:
    """T_b, shape (B,): evenly spaced temperatures, one column of replicas each."""
    return torch.linspace(t_min, t_max, n, device=device)


def initial_chains(native: torch.Tensor, n_temps: int, n_rep: int, seed: int = 0) -> torch.Tensor:
    """x0 of shape (B·R, N, 3): for every temperature, half the replicas start folded (native) and half
    as a straight, fully unfolded chain. Agreement between the two halves means the run has equilibrated."""
    gen = torch.Generator().manual_seed(seed)
    n = native.shape[0]
    bond = (native[1:] - native[:-1]).norm(dim=-1).mean().item()
    straight = torch.zeros(n, 3, device=native.device)
    straight[:, 0] = bond * torch.arange(n, device=native.device)
    starts = [native if r < n_rep // 2 else straight for r in range(n_rep)]
    x0 = torch.stack(starts)[None].expand(n_temps, -1, -1, -1).reshape(-1, n, 3)
    return x0 + 0.05 * torch.randn(x0.shape, generator=gen).to(native.device)
