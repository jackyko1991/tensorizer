r"""Rendering equation → tensor translation.

The light leaving a diffuse surface point x towards the eye is the emitted light plus all incoming light
reflected with the Lambertian BRDF f = ρ/π:

    L(x) = L_e(x) + ∫_{hemisphere} (ρ/π) L(x'(ω)) cos θ dω,

where x'(ω) is the first surface hit looking from x in direction ω. A path tracer estimates this integral by
Monte Carlo, for a batch of rays r at once. Sampling ω with the cosine-weighted pdf p(ω) = cos θ / π cancels
the cosine and the 1/π, so every bounce just multiplies the path throughput by the albedo:

    β_{k+1, rc} = β_{krc} ρ_{krc}          (c = colour channel)

Direct light from the area light (corner, edges e₁, e₂, area A, normal m) is added at every bounce by
sampling a point y on it (next-event estimation):

    D_{rc} = (ρ_{rc}/π) L^e_c (n_{ri} ω_{ri}) (−m_i ω_{ri}) A / |y − x|²_r · V_r       (i summed)

with V_r = 1 if nothing blocks the shadow ray. The pixel value averages L_{rc} = Σ_k β_{krc} D_{krc} (+ L_e
seen directly) over N samples, so its noise falls as 1/√N.
"""

import math

import torch

EPS = 1e-4


def intersect_planes(o: torch.Tensor, d: torch.Tensor, scene: dict) -> torch.Tensor:
    r"""t_{rp} = (d_p − n_{pi} o_{ri}) / (n_{pj} d_{rj})   (i, j summed), +∞ where the hit is behind or off the wall.

    o, d (R, 3) → (R, Pl)"""
    denom = torch.einsum("pj,rj->rp", scene["n"], d)
    t = (scene["d"] - torch.einsum("pi,ri->rp", scene["n"], o)) / denom
    x = o[:, None] + t[..., None] * d[:, None]  # (R, Pl, 3) hit points
    inside = ((x >= scene["lo"] - EPS) & (x <= scene["hi"] + EPS)).all(-1)
    return torch.where((t > EPS) & inside & (denom.abs() > 1e-9), t, math.inf)


def intersect_spheres(o: torch.Tensor, d: torch.Tensor, scene: dict) -> torch.Tensor:
    r"""|o + t d − c|² = r²  ⇒  t = −b ∓ √(b² − c), b = d_i (o − c)_i, c = (o − c)_i (o − c)_i − r².   → (R, S)"""
    oc = o[:, None] - scene["centre"]  # (R, S, 3)
    b = torch.einsum("ri,rsi->rs", d, oc)
    c = torch.einsum("rsi,rsi->rs", oc, oc) - scene["radius"] ** 2
    disc = b**2 - c
    root = disc.clamp(min=0).sqrt()
    near, far = -b - root, -b + root
    t = torch.where(near > EPS, near, far)  # from inside a sphere only the far root counts
    return torch.where((disc > 0) & (t > EPS), t, math.inf)


def nearest_hit(o: torch.Tensor, d: torch.Tensor, scene: dict):
    """Closest primitive along every ray: distance t (R,), primitive id (R,) (−1 = escaped) and the unit normal
    facing the incoming ray (R, 3)."""
    t_all = torch.cat([intersect_planes(o, d, scene), intersect_spheres(o, d, scene)], dim=1)  # (R, Pl + S)
    t, prim = t_all.min(dim=1)
    n_planes = scene["n"].shape[0]
    x = o + t.clamp(max=1e6)[:, None] * d
    plane_n = scene["n"][prim.clamp(max=n_planes - 1)]
    s = (prim - n_planes).clamp(min=0, max=max(0, len(scene["radius"]) - 1))
    sphere_n = (x - scene["centre"][s]) / scene["radius"][s, None] if len(scene["radius"]) else plane_n
    normal = torch.where((prim < n_planes)[:, None], plane_n, sphere_n)
    normal = torch.where((normal * d).sum(-1, keepdim=True) > 0, -normal, normal)  # face the ray
    return t, torch.where(torch.isfinite(t), prim, -1), normal


def cosine_sample(n: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r"""Directions with pdf p(ω) = cos θ / π about the normals n (R, 3) → (R, 3).

    Pick a uniform point in the unit disc (radius √u₁, angle 2π u₂) and lift it onto the hemisphere."""
    u = torch.rand(n.shape[0], 2, generator=gen, device=n.device)
    r, phi = u[:, 0].sqrt(), 2 * math.pi * u[:, 1]
    local = torch.stack([r * torch.cos(phi), r * torch.sin(phi), (1 - u[:, 0]).clamp(min=0).sqrt()], -1)
    helper = torch.where(n[:, :1].abs() > 0.9, n.new_tensor([0.0, 1, 0]), n.new_tensor([1.0, 0, 0]))
    t1 = torch.linalg.cross(helper, n)
    t1 = t1 / t1.norm(dim=1, keepdim=True)
    t2 = torch.linalg.cross(n, t1)
    frame = torch.stack([t1, t2, n], dim=1)  # (R, 3, 3): rows = tangent, bitangent, normal
    return torch.einsum("ra,rai->ri", local, frame)  # ω_i = local_a · frame_{ai}  (a summed)


def cosine_pdf(cos_theta: torch.Tensor) -> torch.Tensor:
    """p(ω) = cos θ / π on the upper hemisphere, 0 below."""
    return cos_theta.clamp(min=0) / math.pi


def direct_light(x: torch.Tensor, n: torch.Tensor, rho: torch.Tensor, scene: dict, gen: torch.Generator):
    r"""Next-event estimate D_{rc} = (ρ/π) L^e (n·ω)(−m·ω) A / |y − x|² · V with y uniform on the light.  → (R, 3)"""
    light = scene["light"]
    u = torch.rand(x.shape[0], 2, generator=gen, device=x.device)
    y = light["corner"] + u[:, :1] * light["e1"] + u[:, 1:] * light["e2"]  # (R, 3) point on the light
    w = y - x
    dist2 = (w * w).sum(-1)
    omega = w / dist2.sqrt()[:, None]
    cos_x = torch.einsum("ri,ri->r", n, omega).clamp(min=0)
    cos_y = -torch.einsum("i,ri->r", light["normal"], omega).clamp(max=0)  # the light faces −m
    area = torch.linalg.cross(light["e1"], light["e2"]).norm()
    t_block, _, _ = nearest_hit(x + EPS * n, omega, scene)
    visible = t_block >= dist2.sqrt() * (1 - 1e-3)  # the first thing the shadow ray meets is the light itself
    g = cos_x * cos_y * area / dist2 * visible
    return rho / math.pi * light["Le"] * g[:, None]
