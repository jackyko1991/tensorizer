r"""Optics of a ray tracer → tensor translation.

Every pixel sends one ray x(t) = o + t d with |d| = 1. All P rays form one (P, 3) tensor and all S spheres one (S, 3)
tensor, so the intersection test is a (P, S) table computed at once. Putting the ray into |x − c_s|² = r_s² gives
a quadratic in t (the leading coefficient is d·d = 1):

    t² + 2 b_{ps} t + q_{ps} = 0,    b_{ps} = (o_{pi} − c_{si}) d_{pi},    q_{ps} = (o − c)_{psi}(o − c)_{psi} − r_s²,

    t_{ps} = −b_{ps} − \sqrt{b_{ps}² − q_{ps}}          (i summed in b and q; no real root ⇒ the ray misses).

The nearest hit is the argmin over s. At the hit point with normal n, light direction l and view direction v:

    Lambert (diffuse)   max(0, n_i l_i)
    Blinn–Phong         max(0, n_i h_i)^α,   h = (l + v)/|l + v|
    mirror reflection   r_i = d_i − 2 (d_j n_j) n_i                  (j summed)

A shadow ray from the hit point towards the light is the same intersection test: if it hits a sphere before
reaching the light, the point gets no direct light.
"""

import torch

EPS = 1e-4  # start secondary rays a little off the surface so they do not hit it again


def normalize(v: torch.Tensor) -> torch.Tensor:
    """v / |v| along the last axis."""
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def sphere_hits(o: torch.Tensor, d: torch.Tensor, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""Distance t_{ps} from ray p to sphere s (inf if missed).   o, d: (P, 3), c: (S, 3), r: (S,) → (P, S)"""
    oc = o[:, None, :] - c[None, :, :]  # (P, S, 3)
    b = torch.einsum("psi,pi->ps", oc, d)  # i summed: b = (o − c)·d
    q = torch.einsum("psi,psi->ps", oc, oc) - r**2  # i summed: |o − c|² − r²
    disc = b**2 - q  # discriminant / 4
    root = disc.clamp(min=0).sqrt()
    t = torch.where(-b - root > EPS, -b - root, -b + root)  # near root, or far root if we start inside
    return torch.where((disc >= 0) & (t > EPS), t, torch.full_like(t, torch.inf))


def plane_hits(o: torch.Tensor, d: torch.Tensor, height: float = 0.0) -> torch.Tensor:
    r"""Distance to the ground plane y = height: t_p = (height − o_{py}) / d_{py} (inf if behind).   → (P,)"""
    t = (height - o[:, 1]) / d[:, 1]
    return torch.where(t > EPS, t, torch.full_like(t, torch.inf))


def lambert(n: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
    r"""Diffuse factor max(0, n_i l_i)   (i summed; l = light_dir).   (P, 3), (P, 3) → (P,)"""
    return torch.einsum("pi,pi->p", n, light_dir).clamp(min=0)


def blinn_phong(n: torch.Tensor, light_dir: torch.Tensor, v: torch.Tensor, shininess: float) -> torch.Tensor:
    r"""Specular factor max(0, n_i h_i)^α with the half vector h = (l + v)/|l + v|.   → (P,)"""
    h = normalize(light_dir + v)
    return torch.einsum("pi,pi->p", n, h).clamp(min=0) ** shininess


def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""Mirror direction r_i = d_i − 2 (d_j n_j) n_i   (j summed).   (P, 3) → (P, 3)"""
    return d - 2 * torch.einsum("pj,pj->p", d, n)[:, None] * n


def lit(p: torch.Tensor, n: torch.Tensor, light: torch.Tensor, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""Shadow rays: True where nothing blocks the segment from p to the light.   p, n: (P, 3), light: (3,) → (P,)"""
    to_light = light - p  # (P, 3)
    dist = torch.linalg.vector_norm(to_light, dim=-1)  # (P,)
    t = sphere_hits(p + EPS * n, to_light / dist[:, None], c, r)  # (P, S), a second batch of rays
    return t.min(dim=1).values > dist  # the ground cannot block a light above it, so only spheres are tested
