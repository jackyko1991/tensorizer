"""Build the concrete tensors: the target fold, the distogram bins, a noisy "network prediction", residue features.

Units: Å. Only the Cα atom of each residue is modelled (one point per amino acid), 3.8 Å apart along the chain.
"""

import math

import torch

CA_CA = 3.8  # Å, distance between consecutive Cα atoms (fixed by the planar peptide bond)
HELIX_RADIUS = 2.3  # Å, radius of the Cα spiral of an α-helix
HELIX_RISE = 1.5  # Å per residue along the helix axis
HELIX_TURN = math.radians(100.0)  # 3.6 residues per turn
HELIX_SEPARATION = 10.0  # Å between the two helix axes (typical for packed helices)
LOOP = 3  # residues in the turn between the helices


def target_structure(n_helix: int, device: torch.device) -> torch.Tensor:
    """Cα trace of a synthetic helix–turn–helix, shape (N, 3) with N = 2 n_helix + LOOP.

    Two right-handed α-helices run antiparallel, side by side, joined by a short loop. Everything is built
    geometrically, so the "known structure" is exact (no coordinates copied from the PDB).
    """
    t = torch.arange(n_helix, dtype=torch.float64)
    phi = HELIX_TURN * t
    # helix 1 goes up the z-axis at x = -s/2; right-handed because the angle grows with z
    h1 = torch.stack(
        [-HELIX_SEPARATION / 2 + HELIX_RADIUS * torch.cos(phi), HELIX_RADIUS * torch.sin(phi), HELIX_RISE * t], -1
    )
    # helix 2: helix 1 half a turn on, rotated by π about the x-axis (still right-handed), moved to x = +s/2
    top = HELIX_RISE * (n_helix - 1)
    h2 = torch.stack(
        [
            HELIX_SEPARATION / 2 - HELIX_RADIUS * torch.cos(phi),
            HELIX_RADIUS * torch.sin(phi),
            top - HELIX_RISE * t,
        ],
        -1,
    )
    loop = _arc(h1[-1], h2[0], LOOP)
    return torch.cat([h1, loop, h2]).float().to(device)


def _arc(p: torch.Tensor, q: torch.Tensor, m: int) -> torch.Tensor:
    """m points on a circular arc from p to q (exclusive), bulging upwards, with all m + 1 chords equal to 3.8 Å."""
    span = torch.linalg.norm(q - p).item()
    # find the arc angle Θ with  sin(Θ/2) / sin(Θ / (2(m+1))) = span / 3.8  by bisection
    lo, hi = 1e-6, 2 * math.pi - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ratio = math.sin(mid / 2) / math.sin(mid / (2 * (m + 1)))
        lo, hi = (mid, hi) if ratio > span / CA_CA else (lo, mid)
    theta = 0.5 * (lo + hi)
    radius = span / (2 * math.sin(theta / 2))
    e = (q - p) / span  # along the chord
    up = torch.tensor([0.0, 0.0, 1.0], dtype=p.dtype)
    up = up - (up @ e) * e
    up = up / torch.linalg.norm(up)  # perpendicular to the chord, pointing up
    centre = 0.5 * (p + q) - radius * math.cos(theta / 2) * up
    a = -theta / 2 + theta * torch.arange(1, m + 1, dtype=p.dtype) / (m + 1)  # angles of the loop points
    return centre + radius * (torch.sin(a)[:, None] * e + torch.cos(a)[:, None] * up)


def bin_centres(device: torch.device, n_bins: int = 64, d_min: float = 2.0, d_max: float = 22.0) -> torch.Tensor:
    """c_k, shape (K,): 64 distance bins between 2 and 22 Å, as in AlphaFold's distogram head."""
    edges = torch.linspace(d_min, d_max, n_bins + 1, device=device)
    return 0.5 * (edges[1:] + edges[:-1])


def noisy_distances(D: torch.Tensor, noise: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand-in for a trained network: d̃_ij = D_ij + σ_ij ε_ij, symmetric, with σ_ij = noise · D_ij / 10 Å.

    Close pairs are predicted sharply and far pairs vaguely, as real distograms are. Returns (d̃, σ), both (N, N).
    """
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(D.shape, generator=gen).to(D.device)
    eps = (eps + eps.T) / math.sqrt(2.0)  # symmetric: d_ij and d_ji are the same prediction
    sigma = noise * D / 10.0 + 0.2  # (N, N); a 0.2 Å floor
    d = (D + sigma * eps).clamp(min=2.0)
    d.fill_diagonal_(0.0)
    return d, sigma


def positional_features(n: int, dim: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal encoding of the residue index i, shape (N, E): the only "sequence" information the toy sees."""
    i = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(100.0) * torch.arange(0, dim, 2, device=device) / dim)[None, :]
    return torch.cat([torch.sin(i * freq), torch.cos(i * freq)], dim=1)
