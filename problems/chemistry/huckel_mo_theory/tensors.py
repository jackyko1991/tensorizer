"""Build the concrete tensors: carbon skeletons, their adjacency matrices, padded into one batch.

Energies are in units of |β| measured from α: ALPHA = 0 and BETA = -1, so an MO at α + xβ has energy -x.
"""

import math

import torch

ALPHA = 0.0
BETA = -1.0
PAD = 1e3  # diagonal entry for padding atoms: their fake MOs sit far above every real one and are never filled


def _zigzag(n: int) -> list[tuple[float, float]]:
    """An all-trans chain with 120° angles and unit bonds."""
    return [(i * math.sqrt(3) / 2, 0.5 * (i % 2)) for i in range(n)]


def _ring(n: int, radius: float, start: float = 90.0) -> list[tuple[float, float]]:
    return [
        (radius * math.cos(math.radians(start + 360 * i / n)), radius * math.sin(math.radians(start + 360 * i / n)))
        for i in range(n)
    ]


def _naphthalene() -> list[tuple[float, float]]:
    s = math.sqrt(3) / 2
    left = [(-s, 1), (-2 * s, 0.5), (-2 * s, -0.5), (-s, -1)]
    right = [(s, 1), (2 * s, 0.5), (2 * s, -0.5), (s, -1)]
    return [(0, 0.5), (0, -0.5)] + left + right  # the two shared atoms first


# name → (atom positions with unit C–C bonds, number of π electrons); the allyl radical has an odd count
MOLECULES = {
    "ethylene": (_zigzag(2), 2),
    "allyl": (_zigzag(3), 3),
    "butadiene": (_zigzag(4), 4),
    "hexatriene": (_zigzag(6), 6),
    "cyclobutadiene": ([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], 4),
    "benzene": (_ring(6, 1.0), 6),
    "naphthalene": (_naphthalene(), 10),
}


def adjacency(xy: torch.Tensor) -> torch.Tensor:
    """A_ij = 1 for atoms one bond length (1) apart, else 0.   xy (N, 2) → (N, N)"""
    d = torch.cdist(xy, xy)
    return ((d > 0.9) & (d < 1.1)).to(xy.dtype)


def batch(device: torch.device) -> tuple[list[str], list[torch.Tensor], torch.Tensor, torch.Tensor, list[int]]:
    """Every molecule's adjacency matrix in one padded batch.

    Returns names, positions [(N_b, 2)], A (B, N, N), a padding diagonal D (B, N, N) and electron counts.
    Molecule b fills the top-left N_b × N_b block; its padding atoms get D_ii = PAD and no bonds.
    """
    names = list(MOLECULES)
    xys = [torch.tensor(MOLECULES[m][0], dtype=torch.float64) for m in names]
    N = max(len(xy) for xy in xys)
    A = torch.zeros(len(names), N, N, dtype=torch.float64)
    D = torch.zeros_like(A)
    for b, xy in enumerate(xys):
        n = len(xy)
        A[b, :n, :n] = adjacency(xy)
        D[b, range(n, N), range(n, N)] = PAD
    return names, xys, A.to(device), D.to(device), [MOLECULES[m][1] for m in names]


def ring_adjacency(N: int) -> torch.Tensor:
    """A cycle of N atoms: A_ij = 1 when j = i ± 1 (mod N).   (N, N)"""
    eye = torch.eye(N, dtype=torch.float64)
    return eye.roll(1, 1) + eye.roll(-1, 1)


def ring_electrons(N: int) -> int:
    """π electrons of the ring studied for each size: the neutral molecule for even N; for odd N the ion
    (cation or anion) with 4n + 2 electrons, like C₃H₃⁺, C₅H₅⁻ and C₇H₇⁺."""
    if N % 2 == 0:
        return N
    return N - 1 if (N - 1) % 4 == 2 else N + 1
