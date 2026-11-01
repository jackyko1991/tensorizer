r"""Rasterisation → tensor translation.

A triangle t with screen corners v_{tki} (k = corner 0, 1, 2; i = screen x, y) covers pixel q_{pi} when all three
barycentric weights are non-negative. With the 2D Levi-Civita symbol ε_{ij} (ε_{01} = 1, ε_{10} = −1),

    λ_{ptk} = ε_{ij} (v_{t,k+1} − q_p)_i (v_{t,k+2} − q_p)_j / (2 A_t),
    2 A_t = ε_{ij} (v_{t1} − v_{t0})_i (v_{t2} − v_{t0})_j

(i, j summed): each λ is the area of the small triangle opposite corner k over the whole area, so λ_{pt0} + λ_{pt1}
+ λ_{pt2} = 1. A_t > 0 means the corners appear counter-clockwise, i.e. the triangle faces the camera; A_t ≤ 0 is a
back face and is culled.

Depth is not linear on the screen, but 1/z is: 1/z_{pt} = λ_{ptk} / z_{tk} (k summed). The z-buffer keeps, for every
pixel, the triangle with the smallest z — a min-reduction over t. Any attribute a (texture coordinate, normal,
colour) is interpolated perspective-correctly by weighting with λ/z,

    a_p = (λ_{pk} a_{pk} / z_{pk}) / (λ_{pk} / z_{pk}),

while the cheap affine version a_p = λ_{pk} a_{pk} (used by early game consoles) bends textures on big, tilted
triangles.
"""

import math

import torch

EPS2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # ε_{ij}, 2D Levi-Civita symbol


def rotation(theta: float, tilt: float, device: torch.device) -> torch.Tensor:
    """R_{ij} = (spin by θ about y) · (tilt about x): (3, 3)."""
    c, s, ct, st = math.cos(theta), math.sin(theta), math.cos(tilt), math.sin(tilt)
    spin = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    tilt_m = torch.tensor([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return (spin @ tilt_m).to(device)


def rotate(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'_{ni} = R_{ij} x_{nj}   (j summed).   (N, 3) → (N, 3)"""
    return torch.einsum("ij,nj->ni", R, x)


def project(x: torch.Tensor, cam: dict) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Camera coordinates c_{ni} = R_{ij}(x_{nj} − e_j), then the pinhole: screen = c₀ + f (c_x, c_y) / c_z.

    Returns screen positions (N, 2) and depths z_n = c_z (N,)."""
    c = torch.einsum("ij,nj->ni", cam["R"], x - cam["eye"])
    z = c[:, 2]
    return cam["c"] + cam["f"] * c[:, :2] / z[:, None], z


def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""ε_{ij} a_i b_j   (i, j summed) for any leading shape: (..., 2), (..., 2) → (...)"""
    return torch.einsum("ij,...i,...j->...", EPS2.to(a), a, b)


def signed_area(tri: torch.Tensor) -> torch.Tensor:
    r"""A_t = ½ ε_{ij} (v_{t1} − v_{t0})_i (v_{t2} − v_{t0})_j: positive when the corners are counter-clockwise.

    tri (T, 3, 2) → (T,)"""
    return 0.5 * cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def barycentric(q: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    r"""λ_{ptk} = ε_{ij}(v_{t,k+1} − q_p)_i (v_{t,k+2} − q_p)_j / 2A_t for every pixel and triangle at once.

    q (P, 2), tri (T, 3, 2) → (P, T, 3). The three edge terms add up to 2A_t, which normalises them."""
    d = tri[None] - q[:, None, None]  # (P, T, 3, 2): corners relative to the pixel
    lam = cross2(d.roll(-1, dims=2), d.roll(-2, dims=2))  # (P, T, 3): twice the sub-triangle areas
    return lam / lam.sum(-1, keepdim=True)


def inverse_depth(lam: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    r"""1/z_{pt} = λ_{ptk} / z_{tk}   (k summed): 1/z, not z, is linear on the screen.   (P, T, 3), (T, 3) → (P, T)"""
    return torch.einsum("ptk,tk->pt", lam, 1 / z)


def perspective_interp(lam: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""a_{pc} = (λ_{pk} a_{pkc} / z_{pk}) / (λ_{pk} / z_{pk})   (k summed).   (P, 3), (P, 3), (P, 3, C) → (P, C)"""
    w = lam / z
    return torch.einsum("pk,pkc->pc", w, a) / w.sum(-1, keepdim=True)


def affine_interp(lam: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""a_{pc} = λ_{pk} a_{pkc}   (k summed): linear in screen space, wrong under perspective.   → (P, C)"""
    return torch.einsum("pk,pkc->pc", lam, a)


def blinn_phong(n: torch.Tensor, to_light: torch.Tensor, to_eye: torch.Tensor) -> torch.Tensor:
    r"""I = k_a + k_d max(0, n_i l_i) + k_s max(0, n_i h_i)^s with the half vector h = (l + v)/|l + v|.

    n, to_eye (..., 3) unit vectors, to_light (3,) → (...)"""
    h = to_light + to_eye
    h = h / h.norm(dim=-1, keepdim=True)
    diffuse = (n * to_light).sum(-1).clamp(min=0)
    specular = (n * h).sum(-1).clamp(min=0) ** 40
    return 0.15 + 0.7 * diffuse + 0.5 * specular * (diffuse > 0)


def checker(uv: torch.Tensor, tiles: int = 8) -> torch.Tensor:
    r"""Checkerboard texture: 1 where ⌊n u⌋ + ⌊n v⌋ is even, else 0.   (P, 2) → (P,)"""
    cells = torch.floor(tiles * uv).sum(-1)
    return (cells % 2 == 0).float()
