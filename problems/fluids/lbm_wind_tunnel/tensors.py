"""Build the concrete tensors: the D3Q19 lattice and its moment basis, the STL triangles, the voxelised body and the
links where populations bounce off it."""

import struct
from pathlib import Path

import numpy as np
import torch


def lattice(device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """D3Q19: velocities e_ia (19, 3), weights w_i (19,) and the opposite direction opp[i] (19,).

    i = 0 is rest, 1-6 the six faces (weight 1/18), 7-18 the twelve edges (weight 1/36).
    """
    faces = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    edges = [[a, b, 0] for a in (1, -1) for b in (1, -1)] + [[a, 0, b] for a in (1, -1) for b in (1, -1)]
    edges += [[0, a, b] for a in (1, -1) for b in (1, -1)]
    e = torch.tensor([[0, 0, 0]] + faces + edges, device=device, dtype=dtype)
    w = torch.tensor([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, device=device, dtype=dtype)
    opp = (e[:, None, :] == -e[None, :, :]).all(-1).int().argmax(1)  # opp[i]: e_opp = -e_i
    return e, w, opp


def moment_basis(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MRT moment matrix M (19, 19) and a mask of odd moments (19,).

    Rows are the polynomials of d'Humières et al. (2002) — density, momentum, energy, stresses, heat fluxes and
    higher moments — evaluated on the lattice velocities and made orthogonal by Gram–Schmidt.
    """
    x, y, z = e.T
    r2 = x * x + y * y + z * z
    polys = [
        (torch.ones_like(x), 0), (x, 1), (y, 1), (z, 1),  # density, momentum (conserved)
        (r2, 2), (2 * x * x - y * y - z * z, 2), (y * y - z * z, 2), (x * y, 2), (y * z, 2), (x * z, 2),  # stresses
        (x * r2, 3), (y * r2, 3), (z * r2, 3), (x * (y * y - z * z), 3), (y * (z * z - x * x), 3),
        (z * (x * x - y * y), 3), (r2 * r2, 4), ((2 * x * x - y * y - z * z) * r2, 4), ((y * y - z * z) * r2, 4),
    ]  # fmt: skip
    rows = []
    for p, _ in polys:  # Gram–Schmidt (plain sum over i as inner product)
        for q in rows:
            p = p - (p @ q) / (q @ q) * q
        rows.append(p)
    odd = torch.tensor([deg % 2 == 1 for _, deg in polys], device=e.device)
    return torch.stack(rows), odd


# ---------------------------------------------------------------- geometry: STL → triangles → voxels


def read_stl(path: Path, device=None, dtype=torch.float32) -> torch.Tensor:
    """Triangles (T, 3, 3) [triangle, vertex, xyz] from a binary or ASCII STL file."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * n:  # binary: 80-byte header, count, then 50 bytes per triangle
        rec = np.frombuffer(
            raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84
        )
        tri = rec["v"].astype(np.float64)
    else:  # ASCII: every "vertex x y z" line
        words = raw.decode("ascii", "replace").split()
        idx = [k for k, word in enumerate(words) if word == "vertex"]
        tri = np.array([[float(words[k + 1]), float(words[k + 2]), float(words[k + 3])] for k in idx]).reshape(-1, 3, 3)
    return torch.tensor(tri, device=device, dtype=dtype)


def mesh_volume(tri: torch.Tensor) -> torch.Tensor:
    """V = Σ_t v0 · (v1 × v2) / 6 (divergence theorem; positive for outward-facing triangles)."""
    return torch.einsum("ta,ta->", tri[:, 0], torch.linalg.cross(tri[:, 1], tri[:, 2])) / 6


def edge_use_counts(tri: torch.Tensor, decimals: int = 4) -> torch.Tensor:
    """How many triangles share each edge (2 everywhere ⇔ closed surface)."""
    verts, ids = torch.unique(torch.round(tri.reshape(-1, 3) * 10**decimals), dim=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    edges = torch.cat([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])  # (3T, 2)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0, return_counts=True)[1]


def voxelise(tri: torch.Tensor, shape: tuple[int, int, int], max_pairs: int = 4_000_000) -> torch.Tensor:
    """Solid mask (Nx, Ny, Nz) by ray casting along +x from every node, all rays and all triangles at once (chunked
    over rays to bound memory).

    For the ray through (y, z) and triangle (a, b, c): barycentric weights in the (y, z) projection decide whether it
    hits, and x_hit = a_x + λ_b (b_x − a_x) + λ_c (c_x − a_x). Each hit counts +1 if the outward normal points along
    +x (the ray leaves a solid) and −1 otherwise; the sum over the hits ahead of a node is its winding number. For one
    closed surface this is the ray-parity test (inside ⇔ odd number of crossings ⇔ winding 1); for parts that overlap
    (wings through a fuselage) it stays right where plain parity would cancel. Hits ahead come from a searchsorted.
    """
    nx, ny, nz = shape
    dev, dt = tri.device, tri.dtype
    yy, zz = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=dt), torch.arange(nz, device=dev, dtype=dt), indexing="ij"
    )
    rays = torch.stack([yy.reshape(-1) + 1.234e-5, zz.reshape(-1) + 2.345e-5], 1)  # (R, 2), off-grid to avoid edges
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]  # (T, 3)
    ab, ac = (b - a)[:, 1:], (c - a)[:, 1:]  # (T, 2) in the (y, z) plane

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    det = cross(ab, ac)  # (T,) = x-component of the (unnormalised) triangle normal
    xs = torch.arange(nx, device=dev, dtype=dt)
    inside = []
    chunk = max(1, max_pairs // max(1, len(tri)))
    for start in range(0, len(rays), chunk):  # chunks of rays, not single nodes
        ap = rays[start : start + chunk, None, :] - a[None, :, 1:]  # (r, T, 2)
        lb = cross(ap, ac) / det
        lc = cross(ab, ap) / det
        hit = (lb >= 0) & (lc >= 0) & (lb + lc <= 1) & (det != 0)
        x_hit = a[:, 0] + lb * (b - a)[:, 0] + lc * (c - a)[:, 0]  # (r, T)
        x_hit, order = torch.where(hit, x_hit, torch.full_like(x_hit, float("inf"))).sort(dim=1)
        sign = torch.gather(torch.where(hit, torch.sign(det).expand_as(hit), torch.zeros_like(x_hit)), 1, order)
        before = torch.nn.functional.pad(torch.cumsum(sign, 1), (1, 0))  # (r, T+1): signed hits up to each index
        behind = torch.searchsorted(x_hit, xs.expand(len(x_hit), nx).contiguous())  # hits with x_hit < x
        winding = before[:, -1:] - torch.gather(before, 1, behind)  # signed hits ahead of each node
        inside.append(winding > 0.5)  # (r, Nx)
    return torch.cat(inside).T.reshape(nx, ny, nz)


def place_body(tri: torch.Tensor, shape: tuple[int, int, int], length: float, nose_x: float, clearance: float | None,
               pitch_deg: float = 0.0) -> tuple[torch.Tensor, float]:  # fmt: skip
    """Scale an STL (any unit) so its x-extent is `length` lattice spacings, pitch it nose-up by pitch_deg about its
    centre, put the nose at x = nose_x, centre it in y, and lift it `clearance` (STL units) above the ground plane
    (the half-way wall at z = −½), or centre it in z if clearance is None (free flight). Returns lattice-coordinate
    triangles and the scale (lattice units per STL unit)."""
    v = tri.reshape(-1, 3)
    scale = length / float(v[:, 0].max() - v[:, 0].min())
    v = (v - v.mean(0)) * scale
    th = torch.deg2rad(torch.tensor(pitch_deg, dtype=v.dtype, device=v.device))
    rot = torch.stack([torch.stack([torch.cos(th), torch.zeros_like(th), torch.sin(th)]),
                       torch.tensor([0.0, 1.0, 0.0], dtype=v.dtype, device=v.device),
                       torch.stack([-torch.sin(th), torch.zeros_like(th), torch.cos(th)])])  # fmt: skip
    v = torch.einsum("ab,nb->na", rot, v)  # nose up for positive pitch (x forward, z up)
    lo = v.min(0).values
    shift = torch.tensor([nose_x, 0.0, -0.5 + (clearance or 0.0) * scale], dtype=v.dtype, device=v.device) - lo
    shift[1] = (shape[1] - 1) / 2 - (v[:, 1].max() + v[:, 1].min()) / 2
    if clearance is None:
        shift[2] = (shape[2] - 1) / 2 - (v[:, 2].max() + v[:, 2].min()) / 2
    return (v + shift).reshape(-1, 3, 3), scale


def sphere_mesh(radius: float, centre, n_theta: int = 24, n_phi: int = 48, dtype=torch.float32) -> torch.Tensor:
    """UV sphere (T, 3, 3), outward-facing triangles."""
    th = torch.linspace(0, torch.pi, n_theta + 1, dtype=dtype)
    ph = torch.linspace(0, 2 * torch.pi, n_phi + 1, dtype=dtype)
    t, p = torch.meshgrid(th, ph, indexing="ij")
    pts = torch.stack([torch.sin(t) * torch.cos(p), torch.sin(t) * torch.sin(p), torch.cos(t)], -1) * radius
    pts = pts + torch.tensor(centre, dtype=dtype)
    a, b, c, d = pts[:-1, :-1], pts[1:, :-1], pts[1:, 1:], pts[:-1, 1:]  # quad corners (n_theta, n_phi, 3)
    tri = torch.cat([torch.stack([a, b, c], 2).reshape(-1, 3, 3), torch.stack([a, c, d], 2).reshape(-1, 3, 3)])
    area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1)
    return tri[area > 1e-12 * radius**2]  # drop the degenerate triangles at the poles


def box_mesh(lo, hi, dtype=torch.float32) -> torch.Tensor:
    """Axis-aligned box (12, 3, 3), outward-facing triangles."""
    lo, hi = torch.tensor(lo, dtype=dtype), torch.tensor(hi, dtype=dtype)
    corner = torch.stack([torch.where(torch.tensor([(k >> j) & 1 for j in range(3)]).bool(), hi, lo) for k in range(8)])
    quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]  # outward order
    return torch.stack([corner[[q[0], q[k], q[k + 1]]] for q in quads for k in (1, 2)])


# ---------------------------------------------------------------- lattice masks and initial state


def body_links(solid: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """links[i, x, y, z] = True where node x is fluid and x + e_i is inside the body (a population leaving along e_i
    hits the body and bounces back).   (19, Nx, Ny, Nz)"""
    hits = [torch.roll(solid, shifts=(-ex, -ey, -ez), dims=(0, 1, 2)) for ex, ey, ez in e.int().tolist()]
    return ~solid & torch.stack(hits)


def frontal_area(solid: torch.Tensor) -> float:
    """Projected area seen by the flow: number of (y, z) columns that contain a solid node."""
    return float(solid.any(dim=0).sum())


def planform_area(solid: torch.Tensor) -> float:
    """Area seen from above (wings): number of (x, y) columns that contain a solid node."""
    return float(solid.any(dim=2).sum())


def initial_populations(solid: torch.Tensor, u_in: float, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Equilibrium (19, Nx, Ny, Nz) at ρ = 1 with the free-stream velocity outside the body, at rest inside."""
    from .equation import equilibrium

    u = torch.zeros(3, *solid.shape, device=e.device, dtype=e.dtype)
    u[0] = u_in
    u[:, solid] = 0
    return equilibrium(torch.ones(solid.shape, device=e.device, dtype=e.dtype), u, e, w)


def read_reference_csv(path: Path) -> dict[str, torch.Tensor]:
    """Columns of a CSV with '#' comment lines and a header row, as float64 tensors."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    names = lines[0].split(",")
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    return {name: torch.tensor(data[:, k], dtype=torch.float64) for k, name in enumerate(names)}


def surface_nets(field: torch.Tensor, level: float, periodic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Iso-surface {field = level} by naive surface nets, for all cells at once.

    field (Nx, Ny, Nz) on integer grid points. Every cell (cube of 8 neighbouring points) whose corners straddle the
    level gets one vertex: the mean of the points where the level crosses its 12 edges (linear interpolation along
    each edge). Every grid edge that crosses the level gets a quad joining the 4 cells around it, split into two
    triangles oriented away from the inside (field > level). periodic=True wraps all three directions (a torus).
    Returns vertices (V, 3) in grid units and triangles (F, 3) of vertex indices."""
    dims = torch.tensor(field.shape, device=field.device)
    cells = tuple(int(n) if periodic else int(n) - 1 for n in field.shape)

    def at(offset):  # field at cell corner (i, j, k) + offset, for every cell
        f = torch.roll(field, shifts=tuple(-int(o) for o in offset), dims=(0, 1, 2)) if any(offset) else field
        return f[: cells[0], : cells[1], : cells[2]]

    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    values = {c: at(c) - level for c in corners}
    edges = [(p, q) for p in corners for q in corners if p < q and sum(abs(x - y) for x, y in zip(p, q)) == 1]
    total = torch.zeros(*cells, 3, dtype=field.dtype, device=field.device)
    count = torch.zeros(cells, dtype=field.dtype, device=field.device)
    for p, q in edges:  # the 12 edges of a cell (a loop over edge types, not over cells)
        a, b = values[p], values[q]
        cross = (a > 0) != (b > 0)
        t = torch.where(cross, a / (a - b).masked_fill(~cross, 1.0), torch.zeros_like(a))
        point = torch.tensor(p, dtype=field.dtype, device=field.device) + t[..., None] * torch.tensor(
            [qq - pp for pp, qq in zip(p, q)], dtype=field.dtype, device=field.device)  # fmt: skip
        total += point * cross[..., None]
        count += cross
    active = count > 0
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=field.device) for n in cells], indexing="ij"), -1)
    verts = (grid + total / count.clamp(min=1)[..., None])[active]  # (V, 3)
    index = torch.full(cells, -1, dtype=torch.long, device=field.device)
    index[active] = torch.arange(int(active.sum()), device=field.device)

    inside = field > level
    faces = []
    for axis in range(3):  # grid edges along each axis
        b_ax, c_ax = [d for d in range(3) if d != axis]
        step = [0, 0, 0]
        step[axis] = 1
        nxt = torch.roll(inside, shifts=-1, dims=axis)
        crossing = inside != nxt
        if not periodic:  # the edge and all four cells around it must lie inside the box
            valid = torch.zeros_like(crossing)
            sl = [slice(None)] * 3
            sl[axis] = slice(0, int(dims[axis]) - 1)
            sl[b_ax] = slice(1, int(dims[b_ax]) - 1)
            sl[c_ax] = slice(1, int(dims[c_ax]) - 1)
            valid[tuple(sl)] = True
            crossing &= valid
        p = crossing.nonzero()  # (E, 3) edge start points
        flip = ~inside[p[:, 0], p[:, 1], p[:, 2]] ^ (axis == 1)  # (axis, b, c) is left-handed for the y axis

        around = []  # vertex index of the cells (p − db e_b − dc e_c) sharing this edge
        for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
            q = p.clone()
            q[:, b_ax] -= db
            q[:, c_ax] -= dc
            q %= torch.tensor(cells, device=field.device)
            around.append(index[q[:, 0], q[:, 1], q[:, 2]])
        v00, v10, v11, v01 = around
        quad = torch.stack([v00, v10, v11, v01], 1)
        quad = torch.where(flip[:, None], quad.flip(1), quad)
        faces.append(torch.cat([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
    faces = torch.cat(faces)
    return verts, faces[(faces >= 0).all(1)]
