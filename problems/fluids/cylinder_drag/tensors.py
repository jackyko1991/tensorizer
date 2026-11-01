"""Build the concrete tensors: the D2Q9 lattice, the cylinder mask, its boundary links and the initial populations."""

import torch

from .equation import equilibrium


def lattice(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """D2Q9: velocities e_ia (9, 2), weights w_i (9,) and the index of the opposite velocity opp[i] (9,).

    i = 0 is rest, 1-4 the axis directions, 5-8 the diagonals.
    """
    e = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
        device=device, dtype=torch.float32,
    )  # fmt: skip
    w = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, device=device)
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device)
    return e, w, opp


def cylinder(nx: int, ny: int, cx: float, cy: float, radius: float, device: torch.device) -> torch.Tensor:
    """Solid mask (Nx, Ny), True inside the cylinder."""
    x = torch.arange(nx, device=device)[:, None]
    y = torch.arange(ny, device=device)[None, :]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def boundary_links(solid: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """links[x, y, i] = 1 where a population moving along e_i just streamed from a fluid node into solid node (x, y).

    These are the links that carry momentum into the body: (x, y) is solid and (x, y) - e_i is fluid.
    """
    links = []
    for ex, ey in e.int().tolist():
        came_from_fluid = ~torch.roll(solid, shifts=(ex, ey), dims=(0, 1))  # fluid at (x, y) - e_i
        links.append(solid & came_from_fluid)
    return torch.stack(links, dim=-1).float()  # (Nx, Ny, 9)


def initial_populations(
    nx: int, ny: int, u_in: float, e: torch.Tensor, w: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """f_xyi at equilibrium with uniform flow u_in, plus a small sideways kick that triggers the shedding sooner."""
    rho = torch.ones(nx, ny, device=device)
    u = torch.zeros(nx, ny, 2, device=device)
    u[..., 0] = u_in
    y = torch.arange(ny, device=device)
    u[..., 1] = (
        0.2 * u_in * torch.cos(2 * torch.pi * y / ny)[None, :]
    )  # even in y about the centre: breaks the mirror symmetry
    return equilibrium(rho, u, e, w)
