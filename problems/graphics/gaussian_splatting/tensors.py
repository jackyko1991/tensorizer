"""Build the concrete tensors: the synthetic scene, the cameras, the ground-truth images and the initial Gaussians."""

import math

import torch

BACKGROUND = (1.0, 1.0, 1.0)  # white
LIGHT = (0.45, -0.35, 0.82)  # direction towards the light (world frame, z up)
AMBIENT = 0.35
RADIUS = 3.8  # camera distance from the origin
FOV = 34.0  # full field of view in degrees


def scene(device: torch.device) -> dict:
    """Three coloured spheres and a cube rotated about z: the 'real world' that the Gaussians must reproduce."""
    return {
        "centres": torch.tensor([[-0.55, -0.05, -0.15], [0.55, 0.45, -0.25], [0.15, -0.55, 0.35]], device=device),
        "radii": torch.tensor([0.45, 0.35, 0.28], device=device),
        "sphere_rgb": torch.tensor([[0.86, 0.15, 0.15], [0.15, 0.39, 0.92], [0.96, 0.62, 0.04]], device=device),
        "box_centre": torch.tensor([0.35, 0.05, -0.2], device=device),
        "box_half": torch.tensor([0.3, 0.3, 0.3], device=device),
        "box_yaw": 0.5,  # rad
        "box_rgb": torch.tensor([0.09, 0.64, 0.29], device=device),
    }


def look_at(centre: torch.Tensor) -> torch.Tensor:
    """World→camera rotation W (3, 3) for a camera at `centre` looking at the origin.

    Rows of W are the camera axes in world coordinates: x right, y down (image rows), z forward (depth).
    """
    forward = -centre / centre.norm()
    up = torch.tensor([0.0, 0.0, 1.0], device=centre.device)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    return torch.stack([right, down, forward])


def cameras(azimuths: list[float], elevations: list[float], res: int, device: torch.device) -> dict:
    """V pinhole cameras on a sphere of radius RADIUS, all looking at the origin.

    W (V, 3, 3) and t (V, 3) map a world point to camera coordinates x_c = W x + t; f is the focal length in
    pixels and pix (P, 2) the pixel centres (u, v) of a res × res image, P = res².
    """
    az = torch.tensor(azimuths, device=device)
    el = torch.tensor(elevations, device=device)
    centre = RADIUS * torch.stack([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], dim=-1)  # (V, 3)
    W = torch.stack([look_at(c) for c in centre])  # (V, 3, 3)
    t = -torch.einsum("vij,vj->vi", W, centre)  # (V, 3) so that the camera centre maps to 0
    f = 0.5 * res / math.tan(math.radians(FOV / 2))
    ij = torch.arange(res, device=device, dtype=torch.float32) + 0.5
    v, u = torch.meshgrid(ij, ij, indexing="ij")
    pix = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)  # (P, 2) row-major pixels
    return {"W": W, "t": t, "centre": centre, "f": f, "c": res / 2, "res": res, "pix": pix}


def training_cameras(n_views: int, res: int, device: torch.device) -> dict:
    """n_views cameras evenly spaced in azimuth, alternating between a low (10°) and a high (40°) elevation."""
    az = [2 * math.pi * k / n_views for k in range(n_views)]
    el = [math.radians(10.0 if k % 2 == 0 else 40.0) for k in range(n_views)]
    return cameras(az, el, res, device)


def novel_camera(n_views: int, res: int, device: torch.device) -> dict:
    """A held-out view between the first two training cameras, at an elevation none of them used (25°)."""
    return cameras([math.pi / n_views + 0.1], [math.radians(25.0)], res, device)


def _rays(cams: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Origins (V, P, 3) and unit directions (V, P, 3) of the ray through every pixel of every camera."""
    d_cam = torch.cat([(cams["pix"] - cams["c"]) / cams["f"], torch.ones_like(cams["pix"][:, :1])], dim=-1)  # (P, 3)
    d = torch.einsum("vji,pj->vpi", cams["W"], d_cam)  # W^T d_cam: back to world axes
    d = d / d.norm(dim=-1, keepdim=True)
    return cams["centre"][:, None, :].expand_as(d), d


def render_scene(sc: dict, cams: dict) -> torch.Tensor:
    """Ground truth by ray casting: nearest hit of spheres and cube, Lambertian shading, white background.

    Returns images (V, P, 3) in [0, 1]. This analytic renderer is only the 'camera'; the Gaussians never see it.
    """
    o, d = _rays(cams)
    V, P, _ = o.shape
    t_best = torch.full((V, P), float("inf"), device=o.device)
    normal = torch.zeros(V, P, 3, device=o.device)
    rgb = torch.zeros(V, P, 3, device=o.device)

    for c, r, col in zip(sc["centres"], sc["radii"], sc["sphere_rgb"]):  # |o + t d - c|² = r²
        oc = o - c
        b = (oc * d).sum(-1)
        disc = b**2 - (oc * oc).sum(-1) + r**2
        t = -b - disc.clamp(min=0).sqrt()
        hit = (disc > 0) & (t > 0) & (t < t_best)
        t_best = torch.where(hit, t, t_best)
        n = (o + t[..., None] * d - c) / r
        normal = torch.where(hit[..., None], n, normal)
        rgb = torch.where(hit[..., None], col, rgb)

    # cube: slab test in its own frame (rotated by yaw about z)
    cy, sy = math.cos(sc["box_yaw"]), math.sin(sc["box_yaw"])
    Rb = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], device=o.device)  # box → world
    ob = torch.einsum("ji,vpj->vpi", Rb, o - sc["box_centre"])  # R^T (o - c)
    db = torch.einsum("ji,vpj->vpi", Rb, d)
    h = sc["box_half"]
    t1, t2 = (-h - ob) / db, (h - ob) / db
    t_near = torch.minimum(t1, t2).amax(-1)
    t_far = torch.maximum(t1, t2).amin(-1)
    hit = (t_far > t_near) & (t_near > 0) & (t_near < t_best)
    p = (ob + t_near[..., None] * db) / h  # on the unit cube surface
    face = p.abs().argmax(-1, keepdim=True)
    nb = torch.zeros_like(p).scatter(-1, face, p.gather(-1, face).sign())
    t_best = torch.where(hit, t_near, t_best)
    normal = torch.where(hit[..., None], torch.einsum("ij,vpj->vpi", Rb, nb), normal)
    rgb = torch.where(hit[..., None], sc["box_rgb"], rgb)

    light = torch.tensor(LIGHT, device=o.device)
    light = light / light.norm()
    shade = AMBIENT + (1 - AMBIENT) * (normal @ light).clamp(min=0)
    bg = torch.tensor(BACKGROUND, device=o.device)
    return torch.where(torch.isfinite(t_best)[..., None], rgb * shade[..., None], bg)


def init_gaussians(m: int, seed: int, device: torch.device) -> dict:
    """M Gaussians with random centres in the cube [-1, 1]³ (no SfM point cloud here), all trainable:

    mean (M, 3), log_scale (M, 3), quat (M, 4) unnormalised rotation, colour_logit (M, 3), opacity_logit (M,)
    """
    gen = torch.Generator().manual_seed(seed)
    g = {
        "mean": torch.rand(m, 3, generator=gen) * 2 - 1,
        "log_scale": torch.full((m, 3), math.log(0.1)),
        "quat": torch.randn(m, 4, generator=gen),
        "colour_logit": torch.zeros(m, 3),  # sigmoid(0) = 0.5: grey
        "opacity_logit": torch.full((m,), math.log(0.3 / 0.7)),  # sigmoid → 0.3
    }
    return {k: v.to(device).requires_grad_(True) for k, v in g.items()}
