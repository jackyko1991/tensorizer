"""Build the concrete tensors: read the STL and CSV data, voxelise the lumen, signed distance to the wall.

data/aneurysm.stl          watertight surface in mm; the 2-byte STL attribute of each triangle is 0 wall, 1 inlet cap,
                           2, 3 outlet caps (synthetic geometry, made offline with VMTK: see meta.yaml notes)
data/inflow_waveform.csv   Q(t) in ml/s over one cardiac cycle (analytic ICA-like waveform, Ford et al. 2005 mean)
data/vmtk_reference_centerlines.csv   VMTK centrelines of the same surface (test reference only)
data/openfoam_reference_womersley.csv OpenFOAM pimpleFoam pipe run (test reference only)
"""

import math
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
RHO, NU = 1060.0, 3.5e-6  # blood density (kg/m³) and kinematic viscosity (m²/s), Newtonian
MM = 1e-3


# ---------------------------------------------------------------- file readers (no VTK, no external tools)
def read_stl(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary STL → triangles (F, 3, 3) in metres and the 2-byte attribute per triangle (F,)."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    rec = np.frombuffer(raw[84 : 84 + 50 * n], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tris = torch.tensor(rec["v"].astype(np.float64)) * MM
    return tris, torch.tensor(rec["a"].astype(np.int64))


def read_csv(path: Path) -> np.ndarray:
    """A numeric CSV with '#' comment lines and one header line."""
    lines = [s for s in Path(path).read_text().splitlines() if s and not s.startswith("#")]
    return np.array([[float(x) for x in s.split(",")] for s in lines[1:]])


def inflow_waveform() -> tuple[torch.Tensor, torch.Tensor]:
    """t (K,) in s and Q (K,) in m³/s over one period."""
    a = read_csv(DATA / "inflow_waveform.csv")
    return torch.tensor(a[:, 0]), torch.tensor(a[:, 1]) * 1e-6


GEOMETRIES = {  # surface STL and its VMTK reference centrelines
    "lausanne": ("aneurysm_lausanne.stl", "vmtk_reference_centerlines_lausanne.csv"),
    "synthetic": ("aneurysm.stl", "vmtk_reference_centerlines.csv"),
}
R_ICA = 2.0e-3  # internal carotid radius that the 4.58 ml/s mean flow belongs to (m)


def surface(name: str = "lausanne") -> tuple[torch.Tensor, torch.Tensor]:
    """Triangles (F, 3, 3) in m and cap ids (0 wall, 1 inlet, 2… outlets) of a bundled geometry."""
    return read_stl(DATA / GEOMETRIES[name][0])


def vmtk_reference(name: str = "lausanne") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VMTK centreline points (M, 3) in m, maximum-inscribed-sphere radius (M,) in m, outlet index (M,)."""
    a = torch.tensor(read_csv(DATA / GEOMETRIES[name][1]))
    return a[:, :3] * MM, a[:, 3] * MM, a[:, 4].long()


def openfoam_reference() -> dict:
    a = torch.tensor(read_csv(DATA / "openfoam_reference_womersley.csv"))
    return {"t": a[:, 0], "u_axis": a[:, 1], "u_half": a[:, 2], "wss": a[:, 3], "Q": a[:, 4]}


# ---------------------------------------------------------------- geometry helpers
def mesh_volume(tris: torch.Tensor) -> float:
    """Signed volume ⅙ Σ a·(b×c) of a closed, outward-oriented triangle mesh (divergence theorem)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return (torch.einsum("fi,fi->f", a, torch.linalg.cross(b, c)).sum() / 6).item()


def cylinder_mesh(R: float, L: float, n_theta: int = 64, n_z: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed triangulated cylinder along z from 0 to L (outward normals); ids 0 wall, 1 cap at z=0, 2 cap at z=L."""
    th = torch.linspace(0, 2 * math.pi, n_theta + 1, dtype=torch.float64)[:-1]
    z = torch.linspace(0, L, n_z + 1, dtype=torch.float64)
    ring = torch.stack([R * torch.cos(th), R * torch.sin(th)], -1)  # (Θ, 2)
    P = torch.cat([ring[None].expand(n_z + 1, -1, -1), z[:, None, None].expand(-1, n_theta, 1)], -1)  # (Z+1, Θ, 3)
    i, j = torch.meshgrid(torch.arange(n_z), torch.arange(n_theta), indexing="ij")
    j1 = (j + 1) % n_theta
    a, b, c, d = P[i, j], P[i, j1], P[i + 1, j1], P[i + 1, j]
    wall = torch.cat([torch.stack([a, b, c], -2), torch.stack([a, c, d], -2)]).reshape(-1, 3, 3)
    centre = lambda zz: torch.tensor([0.0, 0.0, zz], dtype=torch.float64)  # noqa: E731
    bot = torch.stack([centre(0.0).expand(n_theta, 3), P[0, (torch.arange(n_theta) + 1) % n_theta], P[0]], 1)
    top = torch.stack([centre(L).expand(n_theta, 3), P[-1], P[-1, (torch.arange(n_theta) + 1) % n_theta]], 1)
    tris = torch.cat([wall, bot, top])
    ids = torch.cat([torch.zeros(len(wall)), torch.ones(n_theta), 2 * torch.ones(n_theta)]).long()
    return tris, ids


def voxelize(tris: torch.Tensor, lo: torch.Tensor, h: float, shape: tuple) -> torch.Tensor:
    """Inside test by ray parity: a voxel centre is inside if a ray along +z crosses the surface an odd number of times.

    For each triangle and each (x, y) ray column under it, the crossing height z comes from barycentric coordinates;
    crossings are histogrammed into the z cells and a cumulative sum mod 2 fills the lumen. → (nx, ny, nz) bool.
    """
    nx, ny, nz = shape
    P = (tris - lo) / h  # (F, 3, 3) in voxel units; voxel centre i sits at i
    mn, mx = P[..., :2].min(1).values.ceil().long(), P[..., :2].max(1).values.floor().long()  # (F, 2) ray columns
    k = int((mx - mn).max().item()) + 1
    off = torch.stack(torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij"), -1).reshape(-1, 2)  # (K, 2)
    cand = mn[:, None, :] + off[None]  # (F, K, 2) candidate columns
    f, c = (cand <= mx[:, None, :]).all(-1).nonzero(as_tuple=True)
    q = cand[f, c].double() + torch.tensor([1e-7, 1.4142e-7], dtype=torch.float64)  # tiny shift: never hit an edge
    a, b, cc = P[f, 0], P[f, 1], P[f, 2]  # (M, 3) each
    e1, e2, e3 = b[:, :2] - a[:, :2], cc[:, :2] - a[:, :2], q - a[:, :2]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    s = (e3[:, 0] * e2[:, 1] - e3[:, 1] * e2[:, 0]) / det
    t = (e1[:, 0] * e3[:, 1] - e1[:, 1] * e3[:, 0]) / det
    hit = (s >= 0) & (t >= 0) & (s + t <= 1) & (det.abs() > 1e-14)
    z = (a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (cc[:, 2] - a[:, 2]))[hit]
    col = q[hit].round().long()
    iz = (torch.floor(z).long() + 1).clamp(0, nz)  # the crossing lies between centres floor(z) and floor(z) + 1
    keep = (col[:, 0] >= 0) & (col[:, 0] < nx) & (col[:, 1] >= 0) & (col[:, 1] < ny)
    count = torch.zeros(nx, ny, nz + 1, dtype=torch.long)
    count.index_put_((col[keep, 0], col[keep, 1], iz[keep]), torch.ones_like(iz[keep]), accumulate=True)
    return (count.cumsum(2)[..., :nz] % 2) == 1


def vertices_and_normals(tris: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique vertices (V, 3) and area-weighted outward unit normals (V, 3) of a triangle soup."""
    V, inv = torch.unique((tris.reshape(-1, 3) / 1e-9).round(), dim=0, return_inverse=True)
    V = V * 1e-9
    fn = torch.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # (F, 3), |fn| = 2 area
    N = torch.zeros_like(V).index_add(0, inv, fn.repeat_interleave(3, 0))
    return V, N / N.norm(dim=1, keepdim=True).clamp(min=1e-30)


def wall_distance(x: torch.Tensor, V: torch.Tensor, N: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Distance from points x (M, 3) to the wall: |n_v·(x − v)| for the nearest wall vertex v (tangent plane)."""
    out = []
    for i in range(0, len(x), chunk):
        xi = x[i : i + chunk]
        k = torch.cdist(xi.float(), V.float()).argmin(1)  # (m,) nearest vertex
        out.append(torch.einsum("mi,mi->m", xi - V[k], N[k]).abs())
    return torch.cat(out) if out else x.new_zeros(0)


def caps(tris: torch.Tensor, ids: torch.Tensor) -> list[dict]:
    """For each cap id ≥ 1: centre, outward unit normal and equivalent radius √(A/π)."""
    out = []
    for k in range(1, int(ids.max()) + 1):
        t = tris[ids == k]
        fn = torch.linalg.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) / 2  # (F, 3) area vectors
        area = fn.norm(dim=1)
        centre = (t.mean(1) * area[:, None]).sum(0) / area.sum()
        n = fn.sum(0)
        out.append({"centre": centre, "normal": n / n.norm(), "radius": math.sqrt(area.sum().item() / math.pi)})
    return out


def lumen_grid(tris: torch.Tensor, ids: torch.Tensor, h: float, pad: int = 3) -> dict:
    """Voxelise the lumen and compute the signed wall distance φ (> 0 in the fluid) on the cell centres.

    φ is exact enough near the wall (tangent plane of the nearest wall vertex, cap triangles excluded) and is the
    maximum-inscribed-sphere radius R(x) deep inside, which the centreline step uses.
    """
    P = tris.reshape(-1, 3)
    lo = P.min(0).values - pad * h
    shape = tuple(int(s) for s in ((P.max(0).values + pad * h - lo) / h).ceil().long().tolist())
    inside = voxelize(tris, lo, h, shape)
    V, N = vertices_and_normals(tris[ids == 0])
    band = torch.nn.functional.max_pool3d(inside.float()[None, None], 5, 1, 2)[0, 0] > 0  # fluid + 2 solid layers
    idx = band.nonzero()
    phi = torch.full(shape, -3 * h, dtype=torch.float64)
    d = wall_distance(lo + h * idx.double(), V, N)
    phi[tuple(idx.T)] = torch.where(inside[tuple(idx.T)], d, -d)
    return {"lo": lo, "h": h, "shape": shape, "inside": inside, "phi": phi, "V": V, "N": N, "caps": caps(tris, ids)}


def pipe_grid(R: float, h: float, nz: int = 2) -> dict:
    """Straight periodic pipe along z with the exact signed distance φ = R − r (used for the Poiseuille/Womersley
    checks); the wall sample points are a ring of vertices at mid-height with outward normals."""
    n = int(math.ceil(R / h)) + 3
    ax = h * (torch.arange(2 * n + 1, dtype=torch.float64) - n)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    phi = (R - torch.sqrt(X**2 + Y**2))[:, :, None].expand(-1, -1, nz).clone()
    th = torch.linspace(0, 2 * math.pi, 65, dtype=torch.float64)[:-1]
    N = torch.stack([torch.cos(th), torch.sin(th), torch.zeros_like(th)], -1)
    V = R * N + torch.tensor([0.0, 0.0, h * (nz - 1) / 2], dtype=torch.float64)
    lo = torch.tensor([-n * h, -n * h, 0.0], dtype=torch.float64)
    return {"lo": lo, "h": h, "shape": tuple(phi.shape), "inside": phi > 0, "phi": phi, "V": V, "N": N, "caps": []}


def tube_mask(polylines: list, radii: list, h: float, pad: float = 3.0) -> dict:
    """Voxel mask and exact inscribed radius R = −sdf of a union of tubes around polylines (centreline checks)."""
    pts = torch.cat([torch.tensor(p, dtype=torch.float64) for p in polylines])
    rmax = max(radii)
    lo = pts.min(0).values - rmax - pad * h
    shape = tuple(int(s) for s in ((pts.max(0).values + rmax + pad * h - lo) / h).ceil().long().tolist())
    axes = [lo[i] + h * torch.arange(shape[i], dtype=torch.float64) for i in range(3)]
    X = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    sdf = torch.full((len(X),), float("inf"), dtype=torch.float64)
    for poly, r in zip(polylines, radii):
        p = torch.tensor(poly, dtype=torch.float64)
        a, b = p[:-1], p[1:]  # (S, 3) segments
        ab = b - a
        t = (torch.einsum("msi,si->ms", X[:, None] - a, ab) / (ab * ab).sum(1)).clamp(0, 1)
        d = (X[:, None] - a - t[..., None] * ab).norm(dim=-1).min(1).values
        sdf = torch.minimum(sdf, d - r)
    phi = (-sdf).reshape(shape)
    return {"lo": lo, "h": h, "shape": shape, "inside": phi > 0, "phi": phi, "caps": []}


def openfoam_aneurysm_reference() -> dict | None:
    """OpenFOAM pimpleFoam on our own voxel mesh of the Lausanne geometry at h = 0.4 mm (see the CSV headers):
    |u| at slice points (cell centres) at 25 times in the last cycle, and |WSS| at the wall vertices (peak systole,
    TAWSS). Our run is sampled at the same points (trilinear) and vertices, so any voxel size can be compared."""
    f_sl, f_w = DATA / "openfoam_reference_aneurysm_slice.csv", DATA / "openfoam_reference_aneurysm_wss.csv"
    if not (f_sl.is_file() and f_w.is_file()):
        return None
    head = [s for s in f_sl.read_text().splitlines() if s and not s.startswith("#")][0].split(",")
    a, w = torch.tensor(read_csv(f_sl)), torch.tensor(read_csv(f_w))
    t = torch.tensor([float(c[3:]) for c in head[6:]], dtype=torch.float64)
    peak = float([s for s in f_w.read_text().splitlines() if "peak systole" in s][0].split("t = ")[1].split(" s")[0])
    return {"cells": a[:, :3].long(), "points": a[:, 3:6] * MM, "t": t, "u": a[:, 6:], "V": w[:, :3] * MM,
            "wss_peak": w[:, 3],
            "tawss": w[:, 4], "t_peak": peak}  # fmt: skip
