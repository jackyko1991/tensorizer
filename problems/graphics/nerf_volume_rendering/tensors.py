"""Build the concrete tensors: the analytic scene, the cameras and their rays, the true images and the MLP weights."""

import math

import torch

BACKGROUND = (1.0, 1.0, 1.0)  # white
LIGHT = (0.45, -0.35, 0.82)  # direction towards the light (world frame, z up)
AMBIENT = 0.35
RADIUS = 3.6  # camera distance from the origin
FOV = 30.0  # full field of view in degrees
NEAR, FAR = RADIUS - 1.6, RADIUS + 1.6  # every ray is sampled on t ∈ [NEAR, FAR]
SCALE = 1.5  # positions are divided by this before encoding, so the scene fits in [-1, 1]³


def scene(device: torch.device) -> dict:
    """A red sphere next to a checkered cube (rotated about z) and a small orange ball on top of the cube.

    The checkerboard is the high-frequency detail that a plain MLP struggles to learn.
    """
    return {
        "centres": torch.tensor([[-0.45, -0.25, -0.1], [0.4, 0.3, 0.42]], device=device),
        "radii": torch.tensor([0.5, 0.2], device=device),
        "sphere_rgb": torch.tensor([[0.86, 0.15, 0.15], [0.96, 0.55, 0.04]], device=device),
        "box_centre": torch.tensor([0.4, 0.3, -0.15], device=device),
        "box_half": torch.tensor([0.37, 0.37, 0.37], device=device),
        "box_yaw": 0.6,  # rad
        "box_rgb": torch.tensor([0.09, 0.64, 0.29], device=device),  # checkerboard colours
        "box_rgb2": torch.tensor([0.93, 0.93, 0.85], device=device),
    }


def cameras(azimuths: list[float], elevations: list[float], res: int, device: torch.device) -> dict:
    """V pinhole cameras on a sphere of radius RADIUS looking at the origin; returns the ray of every pixel:

    origins (V, P, 3), unit directions (V, P, 3), with P = res² pixels in row-major order.
    """
    az = torch.tensor(azimuths, device=device)
    el = torch.tensor(elevations, device=device)
    centre = RADIUS * torch.stack([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], dim=-1)  # (V, 3)
    forward = -centre / centre.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(forward)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm(dim=-1, keepdim=True)
    down = torch.linalg.cross(forward, right)
    f = 0.5 * res / math.tan(math.radians(FOV / 2))
    ij = (torch.arange(res, device=device, dtype=torch.float32) + 0.5 - res / 2) / f
    v, u = torch.meshgrid(ij, ij, indexing="ij")
    u, v = u.reshape(-1), v.reshape(-1)  # (P,)
    d = right[:, None] * u[None, :, None] + down[:, None] * v[None, :, None] + forward[:, None]  # (V, P, 3)
    d = d / d.norm(dim=-1, keepdim=True)
    return {"origins": centre[:, None, :].expand_as(d).contiguous(), "dirs": d, "res": res}


def training_cameras(n_views: int, res: int, device: torch.device) -> dict:
    """n_views cameras evenly spaced in azimuth, alternating between 5° and 45° elevation."""
    az = [2 * math.pi * k / n_views for k in range(n_views)]
    el = [math.radians(5.0 if k % 2 == 0 else 45.0) for k in range(n_views)]
    return cameras(az, el, res, device)


def novel_camera(n_views: int, res: int, device: torch.device) -> dict:
    """A held-out view between two training cameras, at 25° elevation."""
    return cameras([math.pi / n_views + 0.9], [math.radians(25.0)], res, device)


def render_scene(sc: dict, o: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth by analytic ray casting (spheres + cube, Lambertian shading, white background).

    o, d: (..., 3) → colour (..., 3) and hit distance t (..., ), inf where the ray misses.
    """
    t_best = torch.full(o.shape[:-1], float("inf"), device=o.device)
    normal = torch.zeros_like(o)
    rgb = torch.zeros_like(o)
    for c, r, col in zip(sc["centres"], sc["radii"], sc["sphere_rgb"]):  # |o + t d - c|² = r²
        oc = o - c
        b = (oc * d).sum(-1)
        disc = b**2 - (oc * oc).sum(-1) + r**2
        t = -b - disc.clamp(min=0).sqrt()
        hit = (disc > 0) & (t > 0) & (t < t_best)
        t_best = torch.where(hit, t, t_best)
        normal = torch.where(hit[..., None], (o + t[..., None] * d - c) / r, normal)
        rgb = torch.where(hit[..., None], col, rgb)

    cy, sy = math.cos(sc["box_yaw"]), math.sin(sc["box_yaw"])
    Rb = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], device=o.device)  # box frame → world
    ob = (o - sc["box_centre"]) @ Rb  # Rᵀ(o - c), row vectors
    db = d @ Rb
    h = sc["box_half"]
    t1, t2 = (-h - ob) / db, (h - ob) / db  # slab test
    t_near = torch.minimum(t1, t2).amax(-1)
    t_far = torch.maximum(t1, t2).amin(-1)
    hit = (t_far > t_near) & (t_near > 0) & (t_near < t_best)
    p = (ob + t_near[..., None] * db) / h
    face = p.abs().argmax(-1, keepdim=True)
    nb = torch.zeros_like(p).scatter(-1, face, p.gather(-1, face).sign())
    t_best = torch.where(hit, t_near, t_best)
    normal = torch.where(hit[..., None], nb @ Rb.T, normal)
    checks = (torch.floor(p + 1).sum(-1) % 2)[..., None]  # 2 × 2 checkerboard on every face
    box_rgb = checks * sc["box_rgb"] + (1 - checks) * sc["box_rgb2"]
    rgb = torch.where(hit[..., None], box_rgb, rgb)

    light = torch.tensor(LIGHT, device=o.device)
    shade = AMBIENT + (1 - AMBIENT) * (normal @ (light / light.norm())).clamp(min=0)
    colour = torch.where(torch.isfinite(t_best)[..., None], rgb * shade[..., None], torch.tensor(BACKGROUND).to(o))
    return colour, t_best


def init_mlp(in_dim: int, width: int, depth: int, seed: int, device: torch.device) -> list[dict]:
    """Weights of an MLP in_dim → width → … → width → 4 (σ, r, g, b): `depth` hidden layers, He-uniform init."""
    gen = torch.Generator().manual_seed(seed)
    sizes = [in_dim] + [width] * depth + [4]
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        bound = math.sqrt(6.0 / n_in)
        A = (torch.rand(n_out, n_in, generator=gen) * 2 - 1) * bound
        layers.append({"A": A.to(device).requires_grad_(True), "b": torch.zeros(n_out, device=device).requires_grad_()})
    return layers


def parameters(layers: list[dict]) -> list[torch.Tensor]:
    return [t for layer in layers for t in layer.values()]
