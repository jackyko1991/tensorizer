"""Trace every pixel's ray through the scene: nearest hit, shading, shadows and mirror bounces."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import EPS, blinn_phong, lambert, lit, normalize, plane_hits, reflect, sphere_hits

AMBIENT = 0.08  # light that arrives from everywhere, so shadows are not pitch black
SPECULAR = 0.6  # strength of the Blinn–Phong highlight
GROUND_MIRROR = 0.15  # the floor reflects a little too


def sky(d: torch.Tensor) -> torch.Tensor:
    """Background colour for rays that hit nothing: white at the horizon, blue overhead.   (P, 3) → (P, 3)"""
    a = d[:, 1:2].clamp(0, 1)
    return (1 - a) * torch.tensor([0.95, 0.96, 0.98], device=d.device) + a * torch.tensor(
        [0.45, 0.65, 0.95], device=d.device
    )


def checker(p: torch.Tensor) -> torch.Tensor:
    """Checkerboard colour of the ground at points p: 1 × 1 squares, white and grey.   (P, 3) → (P, 3)"""
    parity = (torch.floor(p[:, 0]) + torch.floor(p[:, 2])).remainder(2)[:, None]  # (P, 1), 0 or 1
    return 0.9 - 0.45 * parity.expand(-1, 3)


def first_hit(o: torch.Tensor, d: torch.Tensor, sc: dict[str, torch.Tensor]):
    """Nearest surface of every ray: distance t (P,), index k (P,) with k = S for the ground, hit mask (P,)."""
    t_all = torch.cat([sphere_hits(o, d, sc["centres"], sc["radii"]), plane_hits(o, d)[:, None]], dim=1)  # (P, S+1)
    t, k = t_all.min(dim=1)  # argmin over the S spheres and the plane
    return t, k, torch.isfinite(t)


def surface(p: torch.Tensor, k: torch.Tensor, sc: dict[str, torch.Tensor]):
    """Normal n (P, 3), colour (P, 3) and mirror fraction (P,) at hit points p on surface k."""
    n_spheres = len(sc["radii"])
    on_ground = k == n_spheres
    s = k.clamp(max=n_spheres - 1)  # a valid sphere index for every ray (ground rays are overwritten below)
    n = (p - sc["centres"][s]) / sc["radii"][s][:, None]
    n = torch.where(on_ground[:, None], torch.tensor([0.0, 1.0, 0.0], device=p.device), n)
    colour = torch.where(on_ground[:, None], checker(p), sc["albedo"][s])
    mirror = torch.where(on_ground, torch.full_like(sc["mirror"][s], GROUND_MIRROR), sc["mirror"][s])
    return n, colour, mirror


def trace(
    o: torch.Tensor, d: torch.Tensor, sc: dict[str, torch.Tensor], light: torch.Tensor, bounces: int, shininess: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Colour of every ray (P, 3) after up to `bounces` mirror reflections, plus a few statistics of the first hit."""
    colour = torch.zeros_like(d)
    weight = torch.ones(d.shape[0], 1, device=d.device)  # how much of the pixel the current ray still carries
    stats = {}
    for bounce in range(bounces + 1):
        t, k, hit = first_hit(o, d, sc)
        p = o + torch.where(hit, t, torch.zeros_like(t))[:, None] * d  # misses stay at o (they get weight 0 below)
        n, albedo, mirror = surface(p, k, sc)
        light_dir = normalize(light - p)  # (P, 3) unit vector l towards the light
        v = -d  # towards the viewer
        visible = lit(p, n, light, sc["centres"], sc["radii"]).float()[:, None]  # (P, 1): 1 lit, 0 in shadow
        direct = visible * (
            lambert(n, light_dir)[:, None] * albedo + SPECULAR * blinn_phong(n, light_dir, v, shininess)[:, None]
        )
        local = AMBIENT * albedo + direct  # (P, 3)

        m = mirror[:, None] if bounce < bounces else torch.zeros_like(mirror[:, None])  # last bounce: no mirror
        colour += weight * torch.where(hit[:, None], (1 - m) * local, sky(d))
        weight = weight * torch.where(hit[:, None], m, torch.zeros_like(m))
        if bounce == 0:
            ground = hit & (k == len(sc["radii"]))
            stats = {
                "spheres": (hit & ~(k == len(sc["radii"]))).float().mean().item(),
                "ground": ground.float().mean().item(),
                "sky": (~hit).float().mean().item(),
                "shadow": (1 - visible[ground, 0]).mean().item() if ground.any() else 0.0,
            }
        # The next batch: mirror rays from every hit point (rays that escaped to the sky just carry on, unweighted).
        o = torch.where(hit[:, None], p + EPS * n, o)
        d = torch.where(hit[:, None], reflect(d, n), d)
    return colour.clamp(0, 1), stats


def render(
    sc: dict[str, torch.Tensor],
    eyes: torch.Tensor,
    lights: torch.Tensor,
    target: torch.Tensor,
    fov: float,
    width: int,
    height: int,
    bounces: int,
    shininess: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Images for every frame: eyes, lights (F, 3) → (F, H, W, 3) and first-hit statistics per frame."""
    frames, stats = [], []
    for f in steps(len(eyes), "frames"):
        o, d = tensors.camera_rays(eyes[f], target, fov, width, height)
        rgb, s = trace(o, d, sc, lights[f], bounces, shininess)
        frames.append(rgb.reshape(height, width, 3))
        stats.append(s)
    return torch.stack(frames), stats
