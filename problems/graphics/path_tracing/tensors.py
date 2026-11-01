"""Build the concrete tensors: the scene (bounded planes, spheres, materials, an area light) and the camera rays.

Planes are stored as n_{pi} x_i = d_p with an axis-aligned bounding box [lo, hi] (so a plane is a wall, not an
infinite sheet); spheres as centre and radius. Primitive ids run over the planes first, then the spheres.
"""

import torch

WHITE, RED, GREEN = [0.75, 0.75, 0.75], [0.70, 0.12, 0.10], [0.15, 0.60, 0.15]
BIG = 10.0  # "unbounded" along the plane's own normal direction


def _plane(normal, d, lo, hi, albedo, emission=(0.0, 0.0, 0.0)):
    return normal, d, lo, hi, albedo, emission


def build(planes: list, spheres: list, light: dict | None, device: torch.device) -> dict:
    """Stack the primitive lists into tensors. spheres: (centre, radius, albedo)."""

    def t(rows):
        return torch.tensor(rows, dtype=torch.float32, device=device)

    scene = {
        "n": t([p[0] for p in planes]),  # (Pl, 3) unit normals
        "d": t([p[1] for p in planes]),  # (Pl,) offsets: n·x = d on the plane
        "lo": t([p[2] for p in planes]),  # (Pl, 3) bounding box of each wall
        "hi": t([p[3] for p in planes]),
        "centre": t([s[0] for s in spheres]).reshape(-1, 3),  # (S, 3)
        "radius": t([s[1] for s in spheres]).reshape(-1),  # (S,)
        "albedo": t([p[4] for p in planes] + [s[2] for s in spheres]),  # (Pl + S, 3) diffuse reflectance ρ
        "emission": t([p[5] for p in planes] + [[0.0, 0.0, 0.0] for _ in spheres]),  # (Pl + S, 3) emitted L_e
        "light": None,
    }
    if light is not None:
        scene["light"] = {k: t(v) for k, v in light.items()}
    return scene


def cornell_box(device: torch.device, light_power: float = 12.0) -> dict:
    """The room [−1, 1]³ open towards the camera (+z): white floor, ceiling and back wall, red left wall, green
    right wall, a 0.6 × 0.6 square light just below the ceiling and two white spheres."""
    e = BIG
    le = [light_power] * 3
    planes = [
        _plane([0, 1, 0], -1.0, [-1, -e, -1], [1, e, 1], WHITE),  # floor y = −1
        _plane([0, -1, 0], -1.0, [-1, -e, -1], [1, e, 1], WHITE),  # ceiling y = 1
        _plane([0, 0, 1], -1.0, [-1, -1, -e], [1, 1, e], WHITE),  # back wall z = −1
        _plane([1, 0, 0], -1.0, [-e, -1, -1], [e, 1, 1], RED),  # left wall x = −1
        _plane([-1, 0, 0], -1.0, [-e, -1, -1], [e, 1, 1], GREEN),  # right wall x = 1
        _plane([0, -1, 0], -0.99, [-0.3, -e, -0.3], [0.3, e, 0.3], [0, 0, 0], le),  # the area light at y = 0.99
    ]
    spheres = [([-0.42, -0.6, -0.35], 0.4, WHITE), ([0.45, -0.68, 0.3], 0.32, WHITE)]
    # The light for next-event estimation: a corner, two edge vectors, its normal and its radiance.
    light = {"corner": [-0.3, 0.99, -0.3], "e1": [0.6, 0, 0], "e2": [0, 0, 0.6], "normal": [0, -1, 0], "Le": le}
    return build(planes, spheres, light, device)


def furnace(device: torch.device, albedo: float = 0.5, emission: float = 1.0) -> dict:
    """A closed box whose six walls all emit L_e and reflect a fraction ρ: the radiance everywhere is L_e/(1 − ρ)."""
    e, a, le = BIG, [albedo] * 3, [emission] * 3
    planes = [
        _plane([0, 1, 0], -1.0, [-1, -e, -1], [1, e, 1], a, le),
        _plane([0, -1, 0], -1.0, [-1, -e, -1], [1, e, 1], a, le),
        _plane([0, 0, 1], -1.0, [-1, -1, -e], [1, 1, e], a, le),
        _plane([0, 0, -1], -1.0, [-1, -1, -e], [1, 1, e], a, le),
        _plane([1, 0, 0], -1.0, [-e, -1, -1], [e, 1, 1], a, le),
        _plane([-1, 0, 0], -1.0, [-e, -1, -1], [e, 1, 1], a, le),
    ]
    return build(planes, [], None, device)


def camera(res: int, eye: tuple = (0.0, 0.0, 3.4), focal: float = 2.3) -> dict:
    """Pinhole camera looking down −z; focal length in units of the half image width."""
    return {"eye": torch.tensor(eye), "focal": focal, "res": res}


def camera_rays(cam: dict, spp: int, gen: torch.Generator, device: torch.device):
    """spp jittered rays through every pixel: origins and unit directions of shape (spp·res·res, 3).

    Ray r = s·P + p (sample s, pixel p in row-major order from the top-left)."""
    res = cam["res"]
    s = torch.arange(res, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(s, s, indexing="ij")
    jitter = torch.rand(spp, res, res, 2, generator=gen, device=device)  # random point inside each pixel
    x = (2 * (xx + jitter[..., 0]) / res - 1) / cam["focal"]
    y = (1 - 2 * (yy + jitter[..., 1]) / res) / cam["focal"]
    d = torch.stack([x, y, -torch.ones_like(x)], -1).reshape(-1, 3)
    d = d / d.norm(dim=1, keepdim=True)
    return cam["eye"].to(device).expand_as(d), d


def pixel_of(cam: dict, point: list) -> tuple[int, int]:
    """(row, column) of the pixel that sees a world point (used to probe the image)."""
    p = torch.tensor(point) - cam["eye"]
    x, y = cam["focal"] * p[0] / -p[2], cam["focal"] * p[1] / -p[2]
    res = cam["res"]
    return int((1 - y) * res / 2), int((x + 1) * res / 2)
