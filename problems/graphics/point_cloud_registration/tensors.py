"""Build the concrete tensors: the Spot mesh, dense surface samples, two partial scans seen from two viewpoints
(z-buffer hidden-point removal), sensor noise, outliers and the unknown rigid motion of the second scan.

data/spot_vertices.csv and data/spot_faces.csv hold Keenan Crane's "Spot" cow (2930 vertices, 5856 triangles,
spot_triangulated.obj), released by its author into the public domain (CC0 1.0):
https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/  ·  Crane, Pinkall & Schröder 2013, ACM TOG 32(4).
"""

import math
from pathlib import Path

import torch

from . import equation

DATA = Path(__file__).parent / "data"
D = torch.float64
EYE_DISTANCE = 2.5  # scanner distance from the object centre (object size = bounding-box diagonal = 1)

DEFAULTS = {
    "n_points": 30000,
    "view_angle": 70.0,
    "noise": 0.003,
    "outliers": 0.03,
    "voxel": 0.02,
    "hypotheses": 4000,
    "icp_iterations": 25,
    "starts": 6,
}


def load_spot() -> tuple[torch.Tensor, torch.Tensor]:
    """Spot's vertices V (Nv, 3), centred and scaled so the bounding-box diagonal is 1, and faces F (Nf, 3)."""
    V = torch.tensor(_read_csv(DATA / "spot_vertices.csv", float), dtype=D)
    F = torch.tensor(_read_csv(DATA / "spot_faces.csv", int), dtype=torch.long)
    lo, hi = V.min(0).values, V.max(0).values
    return (V - (lo + hi) / 2) / (hi - lo).norm(), F


def _read_csv(path: Path, kind) -> list[list]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]  # skip the header
    return [[kind(v) for v in line.split(",")] for line in lines if line]


def sample_surface(V: torch.Tensor, F: torch.Tensor, n: int, gen: torch.Generator):
    """n random points uniformly on the mesh surface (triangle chosen ∝ area, uniform barycentric coordinates),
    with the triangle's outward normal.   → points (n, 3), normals (n, 3)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]  # (Nf, 3) corners
    cross = torch.linalg.cross(b - a, c - a)
    area = cross.norm(dim=1)
    tri = torch.multinomial(area, n, replacement=True, generator=gen)  # (n,)
    r1, r2 = torch.rand(n, 1, generator=gen, dtype=D), torch.rand(n, 1, generator=gen, dtype=D)
    s = r1.sqrt()  # (1 - √r1, √r1 (1 - r2), √r1 r2) is uniform on a triangle
    X = (1 - s) * a[tri] + s * (1 - r2) * b[tri] + s * r2 * c[tri]
    return X, cross[tri] / area[tri, None]


def camera_axes(eye: torch.Tensor):
    """Right, up and forward unit vectors of a camera at eye looking at the origin (world y up)."""
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype))
    right = right / right.norm()
    return right, torch.linalg.cross(right, fwd), fwd


def visible(X: torch.Tensor, N: torch.Tensor, eye: torch.Tensor, res: int = 160, tol: float = 0.02) -> torch.Tensor:
    """Which points a scanner at eye sees: back-face test N·(eye - X) > 0, then a z-buffer: project into a res × res
    perspective image, keep per pixel the smallest depth (scatter_reduce 'amin'), and keep points within tol of it.
    → bool (n,)."""
    right, up, fwd = camera_axes(eye)
    rel = X - eye
    z = rel @ fwd  # (n,) depth along the viewing direction
    u, v = (rel @ right) / z, (rel @ up) / z  # perspective image coordinates
    half = torch.maximum(u.abs().max(), v.abs().max()) * 1.0001
    px = ((u / half + 1) / 2 * res).long().clamp(0, res - 1)
    py = ((v / half + 1) / 2 * res).long().clamp(0, res - 1)
    pix = py * res + px  # (n,) flat pixel index
    front = ((eye - X) * N).sum(1) > 0
    depth = torch.where(front, z, torch.inf)
    zbuf = torch.full((res * res,), torch.inf, dtype=X.dtype).scatter_reduce(0, pix, depth, "amin")
    return front & (z <= zbuf[pix] + tol)


def viewpoints(view_angle_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Two scanner positions at azimuths ±view_angle/2 around the vertical axis, 20° above the horizon."""
    el = math.radians(20.0)
    eyes = []
    for az in (-view_angle_deg / 2, view_angle_deg / 2):
        a = math.radians(az + 30.0)  # +30°: look at the cow from its front-left side
        eyes.append(EYE_DISTANCE * torch.tensor([math.sin(a) * math.cos(el), math.sin(el), math.cos(a) * math.cos(el)]))
    return eyes[0].to(D), eyes[1].to(D)


def random_rigid(gen: torch.Generator, max_angle_deg: float, max_trans: float):
    """A random rotation by an angle in [max/2, max] about a random axis, and a random translation of length
    max_trans.   → R (3, 3), t (3,)."""
    axis = torch.randn(3, generator=gen, dtype=D)
    angle = math.radians(max_angle_deg) * (0.5 + 0.5 * torch.rand(1, generator=gen, dtype=D))
    t = torch.randn(3, generator=gen, dtype=D)
    return equation.rodrigues(axis / axis.norm() * angle), max_trans * t / t.norm()


def scan(V, F, eye, n_points, noise, outlier_frac, gen) -> tuple[torch.Tensor, int]:
    """One partial scan: dense surface samples, the visible ones, Gaussian noise σ, plus outlier_frac uniform
    outliers in a box twice the object size.   → points (n, 3), number of surface points (they come first)."""
    X, N = sample_surface(V, F, n_points, gen)
    X = X[visible(X, N, eye)]
    X = X + noise * torch.randn(X.shape, generator=gen, dtype=D)
    n_out = int(outlier_frac * len(X))
    out = (torch.rand(n_out, 3, generator=gen, dtype=D) - 0.5) * 2 * (V.max(0).values - V.min(0).values)
    return torch.cat([X, out]), len(X)


def make_scans(params: dict, seed: int) -> dict:
    """Scan A (the target, world frame) and scan B (the source), B moved by an unknown motion (R_gt, t_gt)."""
    gen = torch.Generator().manual_seed(seed)
    V, F = load_spot()
    eye_a, eye_b = viewpoints(params["view_angle"])
    A, n_a = scan(V, F, eye_a, params["n_points"], params["noise"], params["outliers"], gen)
    B_world, n_b = scan(V, F, eye_b, params["n_points"], params["noise"], params["outliers"], gen)
    R_gt, t_gt = random_rigid(gen, 60.0, 0.5)
    B = equation.transform(R_gt, t_gt, B_world)
    eye_b_own = equation.transform(R_gt, t_gt, eye_b[None])[0]  # the scanner knows where it was in its own frame
    return {"A": A, "B": B, "n_a": n_a, "n_b": n_b, "eye_a": eye_a, "eye_b": eye_b_own, "R_gt": R_gt, "t_gt": t_gt}
