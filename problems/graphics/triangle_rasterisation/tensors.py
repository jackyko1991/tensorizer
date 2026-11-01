"""Build the concrete tensors: a procedural mesh (torus, icosphere or terrain) on a checkered floor, the camera and the
pixel grid.

Every triangle carries its corner texture coordinates uv_{tkc} (k = corner, c = u or v), so seams need no duplicated
vertices. Faces are wound counter-clockwise seen from outside (right-hand rule gives the outward normal).
"""

import math

import torch


def grid_faces(nu: int, nv: int, wrap_u: bool, wrap_v: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Two triangles per cell of an nu × nv parameter grid: faces (T, 3) and corner uv (T, 3, 2), T = 2·nu·nv."""
    cols_v = nv if wrap_v else nv + 1  # vertices per grid row

    def vid(i, j):
        return (i % nu if wrap_u else i) * cols_v + (j % nv if wrap_v else j)

    faces, uv = [], []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)  # this order makes ∂v × ∂u the normal
            for tri in ((a, b, c), (a, c, d)):
                faces.append([vid(*p) for p in tri])
                uv.append([[p[0] / nu, p[1] / nv] for p in tri])
    return torch.tensor(faces), torch.tensor(uv)


def torus(nu: int = 20, nv: int = 10, big_r: float = 0.9, small_r: float = 0.38):
    """Torus around the y axis: vertices (nu·nv, 3), faces (T, 3), corner uv (T, 3, 2)."""
    u = torch.arange(nu) * 2 * math.pi / nu
    v = torch.arange(nv) * 2 * math.pi / nv
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    ring = big_r + small_r * torch.cos(vv)
    verts = torch.stack([ring * torch.cos(uu), small_r * torch.sin(vv), ring * torch.sin(uu)], -1).reshape(-1, 3)
    faces, uv = grid_faces(nu, nv, True, True)
    return verts, faces, uv * torch.tensor([3.0, 1.0])  # the long way round gets 3× more checker tiles


def icosphere(subdivisions: int = 2, radius: float = 1.0):
    """Icosahedron split `subdivisions` times (each triangle → 4) and pushed onto a sphere."""
    t = (1 + math.sqrt(5)) / 2
    verts = torch.tensor(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t]]
        + [[t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6]]
        + [[7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10]]
        + [[8, 6, 7], [9, 8, 1]]
    )
    for _ in range(subdivisions):
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]  # (T, 3) each
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        corners = torch.stack([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], 1).reshape(-1, 3)  # 4 new triangles each
        # Merge the copies of shared vertices: identical rounded coordinates → one index.
        verts, faces = torch.unique((corners * 1e5).round(), dim=0, return_inverse=True)
        verts, faces = verts / 1e5, faces.reshape(-1, 3)
    verts = radius * verts / verts.norm(dim=1, keepdim=True)
    # Spherical texture coordinates per corner; a triangle straddling the u = 0/1 seam gets its small u's lifted by 1.
    p = verts[faces]  # (T, 3, 3)
    u = 0.5 + torch.atan2(p[..., 2], p[..., 0]) / (2 * math.pi)
    v = 0.5 + torch.asin(p[..., 1].clamp(-1, 1)) / math.pi
    seam = (u.max(1, keepdim=True).values - u.min(1, keepdim=True).values) > 0.5
    u = torch.where(seam & (u < 0.5), u + 1, u)
    return verts, faces, torch.stack([4 * u, 2 * v], -1)


def terrain(n: int = 14, size: float = 2.4):
    """Low-poly height field y = h(x, z) on an n × n grid (rolling hills)."""
    s = torch.linspace(-size / 2, size / 2, n + 1)
    xx, zz = torch.meshgrid(s, s, indexing="ij")
    yy = 0.22 * torch.sin(2.3 * xx) * torch.cos(1.9 * zz) + 0.12 * torch.cos(3.1 * xx + 1.7 * zz) - 0.2
    verts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    faces, uv = grid_faces(n, n, False, False)
    return verts, faces, uv * 2


MESHES = {"torus": torus, "icosphere": icosphere, "terrain": terrain}


def scene(mesh: str, device: torch.device) -> dict:
    """The mesh plus a 6 × 6 floor made of only two big triangles (where affine texturing goes badly wrong).

    Returns verts (N, 3), faces (T, 3), uv (T, 3, 2), is_mesh_vertex (N,) and is_mesh_face (T,)."""
    mv, mf, muv = MESHES[mesh]()
    fv = torch.tensor([[-3.0, -1.1, -3.0], [-3.0, -1.1, 3.0], [3.0, -1.1, 3.0], [3.0, -1.1, -3.0]])
    ff, fuv = grid_faces(1, 1, False, False)
    # grid_faces' vertex order (i, j) → i·2 + j matches fv as (x index, z index): 0 (−,−), 1 (−,+), 2 (+,−), 3 (+,+).
    fv = fv[[0, 1, 3, 2]]
    verts = torch.cat([mv, fv])
    faces = torch.cat([mf, ff + len(mv)])
    uv = torch.cat([muv, fuv])
    return {
        "verts": verts.to(device),
        "faces": faces.to(device),
        "uv": uv.to(device),
        "is_mesh_vertex": (torch.arange(len(verts)) < len(mv)).to(device),
        "is_mesh_face": (torch.arange(len(faces)) < len(mf)).to(device),
    }


def camera(res: int, device: torch.device) -> dict:
    """Pinhole camera looking at the origin from above and in front; rotation rows = (right, up, forward)."""
    eye = torch.tensor([0.0, 1.5, 4.0])
    forward = torch.tensor([0.0, -0.25, 0.0]) - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    return {
        "eye": eye.to(device),
        "R": torch.stack([right, up, forward]).to(device),  # world → camera, (3, 3)
        "f": 1.15 * res,  # focal length in pixels (≈ 47° field of view)
        "c": res / 2,  # principal point (centre of the image)
        "res": res,
    }


def pixels(res: int, device: torch.device) -> torch.Tensor:
    """Pixel centres q_{pi} of shape (res·res, 2), row-major from the top row; screen y points up."""
    s = torch.arange(res, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(res - s, s, indexing="ij")
    return torch.stack([xx, yy], -1).reshape(-1, 2)


def light(device: torch.device) -> torch.Tensor:
    """Direction towards a distant light (unit vector), (3,)."""
    d = torch.tensor([-0.5, 1.0, 0.7], device=device)
    return d / d.norm()
