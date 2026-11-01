"""Build the concrete tensors: honeycomb vectors, the Brillouin-zone grid, a zigzag ribbon and a finite flake."""

import math

import torch

S3 = math.sqrt(3)
DELTA = torch.tensor([[0.0, 1.0], [-S3 / 2, -0.5], [S3 / 2, -0.5]], dtype=torch.float64)  # δ_{ni}: A → B bonds
B_NNN = torch.stack([DELTA[1] - DELTA[2], DELTA[2] - DELTA[0], DELTA[0] - DELTA[1]])  # b_{ni}: A → A bonds
# A → B bonds measured from cell to cell (δ_n - δ_1 are lattice vectors). With these phases H(k + G) = H(k), so
# eigenvectors can be compared across the zone boundary, as the FHS links on a periodic k-grid require.
CELL_DELTA = DELTA - DELTA[0]
A1 = torch.tensor([S3, 0.0], dtype=torch.float64)  # Bravais vectors
A2 = torch.tensor([S3 / 2, 1.5], dtype=torch.float64)


def bz_grid(n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """k = (i G_1 + j G_2)/n on an n×n periodic grid covering the Brillouin zone once: (n, n, 2), and the cell area.

    G_1, G_2 are the reciprocal vectors, a_i · G_j = 2π δ_ij.
    """
    G = 2 * math.pi * torch.linalg.inv(torch.stack([A1, A2])).T  # rows G_1, G_2
    s = torch.arange(n, dtype=torch.float64) / n
    k = s[:, None, None] * G[0] + s[None, :, None] * G[1]  # (n, n, 2)
    cell = abs(torch.linalg.det(G).item()) / n**2  # area of one k-plaquette
    return k.to(device), cell


def ribbon_cell(width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One period of a zigzag ribbon, infinite along x and `width` cells tall: positions (2W-2, 2), ξ = ±1 (2W-2,).

    Sites are stacked A, B, A, B, … from bottom to top. The first A and the last B would hang on a single bond
    (a "bearded" edge), so they are dropped, as in the trimmed flake: both edges are then zigzag.
    """
    n2 = torch.arange(width, dtype=torch.float64)
    a_sites = n2[:, None] * A2  # (W, 2)
    pos = torch.stack([a_sites, a_sites + DELTA[0]], dim=1).reshape(-1, 2)[1:-1]
    pos[:, 0] = torch.remainder(pos[:, 0], S3)  # fold into one period along x
    xi = torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(width)[1:-1]
    return pos.to(device), xi.to(device)


def flake(width: float, height: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Honeycomb sites inside a width × height rectangle (zigzag edges top and bottom): positions (N, 2), ξ (N,).

    Sites left with fewer than two neighbours are trimmed, so every edge site has two couplings.
    """
    n = int(2 * (width + height))
    i, j = torch.meshgrid(torch.arange(-n, n + 1.0), torch.arange(-n, n + 1.0), indexing="ij")
    cells = i.reshape(-1, 1).double() * A1 + j.reshape(-1, 1).double() * A2  # (M, 2)
    pos = torch.cat([cells, cells + DELTA[0]])
    xi = torch.cat([torch.ones(len(cells)), -torch.ones(len(cells))]).double()
    eps = 1e-6
    keep = (pos[:, 0] > -eps) & (pos[:, 0] < width + eps) & (pos[:, 1] > -eps) & (pos[:, 1] < height + eps)
    pos, xi = _trim(pos[keep], xi[keep])
    return pos.to(device), xi.to(device)


def carve_defect(pos: torch.Tensor, xi: torch.Tensor, center: torch.Tensor, radius: float):
    """Remove every resonator within `radius` of `center` (a notch cut into the edge), then trim dangling sites."""
    keep = (pos - center).norm(dim=1) > radius
    return _trim(pos[keep], xi[keep])


def _trim(pos: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeatedly drop sites with fewer than two nearest neighbours (distance 1)."""
    while True:
        neighbours = ((torch.cdist(pos, pos) - 1).abs() < 1e-6).sum(1)
        if (neighbours >= 2).all():
            return pos, xi
        pos, xi = pos[neighbours >= 2], xi[neighbours >= 2]


def disorder(n: int, strength: float, seed: int, device: torch.device) -> torch.Tensor:
    """Random detuning of every resonator, uniform in [-W/2, W/2]: (N,). Fixed seed."""
    g = torch.Generator().manual_seed(seed)
    return (strength * (torch.rand(n, generator=g, dtype=torch.float64) - 0.5)).to(device)


def gaussian(pos: torch.Tensor, center: torch.Tensor, width: float) -> torch.Tensor:
    """Normalised Gaussian spot of light on the resonators: (N,) complex."""
    psi = torch.exp(-((pos - center) ** 2).sum(1) / (2 * width**2)).to(torch.complex128)
    return psi / psi.norm()
