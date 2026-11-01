"""Build the concrete tensors: a procedural poster on the plane z = 0, two cameras that see it, the two rendered
views and noisy feature matches with outliers.

Pixel convention: pixel (column i, row j) has its centre at image coordinates (u, v) = (i, j).
"""

import math

import torch

POSTER = (2.4, 1.5)  # half width and half height of the poster in world units
BACKGROUND = 0.94


def texture(xy: torch.Tensor) -> torch.Tensor:
    """Colour of the poster at plane points (..., 2) → (..., 3): smooth colour waves, a grid and some discs."""
    x, y = xy[..., 0], xy[..., 1]
    rgb = torch.stack(
        [
            0.55 + 0.35 * torch.sin(2.1 * x + 0.5) * torch.cos(1.7 * y),
            0.55 + 0.35 * torch.sin(1.3 * y - 0.8 * x + 1.0),
            0.60 + 0.30 * torch.cos(1.1 * x * y + 0.3 * x),
        ],
        -1,
    )
    lines = ((torch.frac(2 * x + 100) < 0.05) | (torch.frac(2 * y + 100) < 0.05))[..., None]
    rgb = torch.where(lines, 0.25 * rgb, rgb)
    gen = torch.Generator().manual_seed(7)
    centres = (torch.rand(14, 2, generator=gen) * 2 - 1) * torch.tensor([2.1, 1.25])
    radii = 0.12 + 0.18 * torch.rand(14, generator=gen)
    colours = torch.rand(14, 3, generator=gen) * 0.8
    dist = (xy[..., None, :] - centres.to(xy)).norm(dim=-1)  # (..., 14)
    inside = dist < radii.to(xy)
    disc = torch.where(inside.any(-1, keepdim=True), colours.to(xy)[inside.float().argmax(-1)], rgb)
    on_poster = ((x.abs() <= POSTER[0]) & (y.abs() <= POSTER[1]))[..., None]
    return torch.where(on_poster, disc, torch.full_like(rgb, BACKGROUND))


def look_at(eye, target, roll: float) -> torch.Tensor:
    """World → camera rotation, rows = (right, down, forward) so image v grows downwards; roll about forward."""
    eye, target = torch.tensor(eye, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    forward = (target - eye) / (target - eye).norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    c, s = math.cos(roll), math.sin(roll)
    return torch.stack([c * right + s * down, -s * right + c * down, forward])


def plane_homography(eye, target, roll: float, focal: float, size: tuple[int, int]) -> torch.Tensor:
    """H = K [r₁ r₂ t] maps plane points (X, Y, 1) on z = 0 to image points (u, v, 1) up to scale: (3, 3)."""
    R = look_at(eye, target, roll)
    t = -R @ torch.tensor(eye, dtype=torch.float32)
    K = torch.tensor([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1.0]])
    return K @ torch.stack([R[:, 0], R[:, 1], t], dim=1)


CAMERAS = [
    {"eye": (-0.75, -0.25, 3.0), "target": (-0.65, 0.0, 0.0), "roll": 0.0},
    {"eye": (0.9, 0.35, 2.7), "target": (0.7, 0.1, 0.0), "roll": 0.18},
]


def views(size: tuple[int, int], focal: float, device: torch.device):
    """Plane-to-image homographies (2, 3, 3), the rendered views (2, H, W, 3) and the true view-1 → view-2 H."""
    w, h = size
    Hs = torch.stack([plane_homography(c["eye"], c["target"], c["roll"], focal, size) for c in CAMERAS])
    # Render at 2× and average 2 × 2 blocks (anti-aliasing); sub-pixel centres sit at ±1/4 pixel.
    s_u = torch.arange(2 * w) / 2 - 0.25
    s_v = torch.arange(2 * h) / 2 - 0.25
    vv, uu = torch.meshgrid(s_v, s_u, indexing="ij")
    pix = torch.stack([uu, vv, torch.ones_like(uu)], -1)  # (2H, 2W, 3) homogeneous pixel coordinates
    imgs = []
    for H in Hs:
        plane = torch.einsum("ij,hwj->hwi", torch.linalg.inv(H), pix)  # back-project every pixel onto the plane
        img = texture(plane[..., :2] / plane[..., 2:])
        imgs.append(img.reshape(h, 2, w, 2, 3).mean(dim=(1, 3)))
    H12 = Hs[1] @ torch.linalg.inv(Hs[0])
    return Hs.to(device), torch.stack(imgs).to(device), (H12 / H12[2, 2]).to(device)


def matches(Hs: torch.Tensor, size: tuple[int, int], n: int, noise: float, outlier_frac: float, seed: int):
    """Synthetic feature matches: n poster points seen by both cameras, Gaussian pixel noise, and a fraction of the
    view-2 points replaced by random positions (wrong matches). Returns x1, x2 (M, 2) and is_outlier (M,)."""
    gen = torch.Generator().manual_seed(seed)
    dev = Hs.device
    pts = (torch.rand(4 * n, 2, generator=gen) * 2 - 1) * torch.tensor(POSTER) * 0.97
    homog = torch.cat([pts, torch.ones(len(pts), 1)], 1).to(dev)
    proj = torch.einsum("cij,nj->cni", Hs, homog)
    uv = proj[..., :2] / proj[..., 2:]  # (2, 4n, 2)
    bound = torch.tensor([size[0] - 1, size[1] - 1], device=dev, dtype=torch.float32)
    seen = ((uv >= 2) & (uv <= bound - 2)).all(-1).all(0)
    x1, x2 = uv[0, seen][:n], uv[1, seen][:n]
    m = len(x1)
    x1 = x1 + noise * torch.randn(m, 2, generator=gen).to(dev)
    x2 = x2 + noise * torch.randn(m, 2, generator=gen).to(dev)
    is_outlier = (torch.rand(m, generator=gen) < outlier_frac).to(dev)
    random_pts = (torch.rand(m, 2, generator=gen).to(dev) * bound).to(x2)
    x2 = torch.where(is_outlier[:, None], random_pts, x2)
    return x1, x2, is_outlier
