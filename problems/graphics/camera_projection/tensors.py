"""Build the concrete tensors: wireframe objects and the floor grid as homogeneous vertices + edge lists."""

import math

import torch


def homogeneous(xyz: torch.Tensor) -> torch.Tensor:
    """Append w = 1 to every point: (N, 3) → (N, 4)."""
    return torch.cat([xyz, torch.ones(len(xyz), 1, device=xyz.device)], dim=1)


def cube(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit cube centred at the origin: vertices (8, 4), edges (12, 2) as vertex index pairs."""
    corners = torch.tensor([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], device=device)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]  # differ in one axis
    return homogeneous(corners), torch.tensor(edges, device=device)


def house(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The cube with a gable roof: two ridge points above the top face.   vertices (10, 4), edges (17, 2)."""
    v, e = cube(device)
    ridge = homogeneous(torch.tensor([[-0.5, 1.0, 0.0], [0.5, 1.0, 0.0]], device=device))
    # top corners: 2 (−x,+y,−z), 3 (−x,+y,+z), 6 (+x,+y,−z), 7 (+x,+y,+z); ridge points are 8 (−x) and 9 (+x)
    roof = torch.tensor([[8, 9], [8, 2], [8, 3], [9, 6], [9, 7]], device=device)
    return torch.cat([v, ridge]), torch.cat([e, roof])


def torus(device: torch.device, big: float = 0.5, small: float = 0.2, n_u: int = 20, n_v: int = 8):
    """Torus around the y axis: vertices (n_u·n_v, 4) and edges along both rings (2·n_u·n_v, 2)."""
    u = torch.arange(n_u, device=device) * 2 * math.pi / n_u
    v = torch.arange(n_v, device=device) * 2 * math.pi / n_v
    uu, vv = torch.meshgrid(u, v, indexing="ij")  # (n_u, n_v)
    rho = big + small * torch.cos(vv)
    xyz = torch.stack([rho * torch.cos(uu), small * torch.sin(vv), rho * torch.sin(uu)], dim=-1).reshape(-1, 3)
    idx = torch.arange(n_u * n_v, device=device).reshape(n_u, n_v)
    around = torch.stack([idx, idx.roll(-1, dims=1)], dim=-1).reshape(-1, 2)  # small circles
    along = torch.stack([idx, idx.roll(-1, dims=0)], dim=-1).reshape(-1, 2)  # big circles
    return homogeneous(xyz), torch.cat([around, along])


def shape(name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return {"cube": cube, "house": house, "torus": torus}[name](device)


def floor_grid(extent: float, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Lines x = const and z = const on the floor y = 0: vertices (4n, 4), edges (2n, 2).
    The first n edges run along z, the last n along x, so each family is a set of parallel lines."""
    s = torch.linspace(-extent, extent, n, device=device)
    e = torch.full_like(s, extent)
    zero = torch.zeros_like(s)
    along_z = torch.cat([torch.stack([s, zero, -e], 1), torch.stack([s, zero, e], 1)])  # (2n, 3)
    along_x = torch.cat([torch.stack([-e, zero, s], 1), torch.stack([e, zero, s], 1)])
    k = torch.arange(n, device=device)
    edges = torch.cat([torch.stack([k, k + n], 1), torch.stack([k + 2 * n, k + 3 * n], 1)])
    return homogeneous(torch.cat([along_z, along_x])), edges


def orbit(n_frames: int, distance: float, height: float, device: torch.device) -> torch.Tensor:
    """Camera positions on a circle around the y axis, starting 30° off the z axis: (F, 3)."""
    a = math.radians(30) + torch.linspace(0, 2 * math.pi, n_frames + 1, device=device)[:-1]
    return torch.stack([distance * torch.sin(a), torch.full_like(a, height), distance * torch.cos(a)], dim=-1)
