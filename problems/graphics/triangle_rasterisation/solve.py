"""The rasteriser: cull, compute barycentrics for (pixels × triangles) in memory-bounded tiles, z-buffer, shade."""

import torch

from problems._common.progress import steps

from .equation import (
    affine_interp,
    barycentric,
    blinn_phong,
    checker,
    inverse_depth,
    perspective_interp,
    project,
    rotate,
    rotation,
    signed_area,
)

NEAR = 0.1
MESH_COLOUR = torch.tensor([0.30, 0.52, 0.95])
FLOOR_COLOUR = torch.tensor([0.80, 0.80, 0.78])
TEXTURE_DARK, TEXTURE_LIGHT = torch.tensor([0.12, 0.25, 0.60]), torch.tensor([0.98, 0.80, 0.35])


def face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted (un-normalised) face normals (b − a) × (c − a): (T, 3)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return torch.linalg.cross(b - a, c - a)


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Average of the normals of the faces around each vertex (needed for smooth shading): (N, 3)."""
    fn = face_normals(verts, faces)
    n = torch.zeros_like(verts).index_add_(0, faces.reshape(-1), fn.repeat_interleave(3, dim=0))
    return n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)


def zbuffer(q: torch.Tensor, tri: torch.Tensor, z: torch.Tensor):
    """Brute force for a set of pixels against a set of triangles: barycentrics (p, T, 3), then the min over T.

    q (p, 2), tri (T, 3, 2), z (T, 3) → index of the nearest covering triangle (p,), its λ (p, 3), its depth (p,)."""
    lam = barycentric(q, tri)  # (p, T, 3)
    inside = (lam >= 0).all(-1)  # (p, T)
    d = torch.where(inside, 1 / inverse_depth(lam, z), float("inf"))  # depth of every covering triangle
    d_min, t = d.min(dim=1)  # the z-buffer: nearest triangle per pixel
    return t, lam[torch.arange(len(t), device=q.device), t], d_min


def rasterise(screen: torch.Tensor, z: torch.Tensor, keep: torch.Tensor, q: torch.Tensor, tile: int = 16):
    """z-buffer every pixel, one screen tile of tile × tile pixels at a time, against only the triangles whose
    bounding box touches that tile (most triangles are small, so this skips almost all pixel-triangle pairs).

    screen (T, 3, 2), z (T, 3), keep (T,) bool, q (P, 2) → triangle id (P,) (−1 = background),
    barycentrics of that triangle (P, 3) and depth (P,)."""
    dev, n_pix = q.device, q.shape[0]
    tri_id = torch.full((n_pix,), -1, dtype=torch.long, device=dev)
    lam_win = torch.zeros(n_pix, 3, device=dev)
    depth = torch.full((n_pix,), float("inf"), device=dev)
    ids = keep.nonzero().squeeze(1)  # (T',) triangles that survive culling
    if len(ids) == 0:
        return tri_id, lam_win, depth
    tri, zt = screen[ids], z[ids]
    lo, hi = tri.min(dim=1).values, tri.max(dim=1).values  # bounding boxes (T', 2)

    cell = torch.floor(q / tile).long()  # tile (column, row) of every pixel, (P, 2)
    n_cols = int(cell[:, 0].max()) + 1
    tile_of_pixel = cell[:, 1] * n_cols + cell[:, 0]  # (P,)
    tiles = torch.unique(tile_of_pixel)
    corner = torch.stack([tiles % n_cols, tiles // n_cols], -1).float() * tile  # lower-left corner of each tile
    # Binning: (tiles × triangles) overlap of bounding boxes, all at once.
    touches = ((lo[None] <= corner[:, None] + tile) & (hi[None] >= corner[:, None])).all(-1)  # (n_tiles, T')
    for k, tid in enumerate(tiles.tolist()):
        cand = touches[k].nonzero().squeeze(1)
        if len(cand) == 0:
            continue
        pix = (tile_of_pixel == tid).nonzero().squeeze(1)
        t, lam, d = zbuffer(q[pix], tri[cand], zt[cand])
        hit = torch.isfinite(d)
        tri_id[pix] = torch.where(hit, ids[cand[t]], -1)
        lam_win[pix], depth[pix] = lam, d
    return tri_id, lam_win, depth


def render(scene: dict, cam: dict, q: torch.Tensor, to_light: torch.Tensor, theta: float, tilt: float) -> dict:
    """Render one frame with the mesh spun by θ. Returns images (res, res, 3) and per-frame statistics."""
    dev, res = q.device, cam["res"]
    R = rotation(theta, tilt, dev)
    verts = torch.where(scene["is_mesh_vertex"][:, None], rotate(R, scene["verts"]), scene["verts"])  # (N, 3)
    faces = scene["faces"]
    normals = vertex_normals(verts, faces)  # (N, 3)

    screen_v, z_v = project(verts, cam)  # (N, 2), (N,)
    screen, z = screen_v[faces], z_v[faces]  # (T, 3, 2), (T, 3)
    front = signed_area(screen) > 1e-9  # back-face culling: clockwise on screen ⇒ facing away
    keep = front & (z.min(1).values > NEAR)
    tri_id, lam, depth = rasterise(screen, z, keep, q)
    hit = tri_id >= 0
    t = tri_id.clamp(min=0)

    # Gather the winning triangle's corner attributes for every pixel: (P, 3, ·).
    zc = z[t]  # (P, 3)
    pos = perspective_interp(lam, zc, verts[faces[t]])  # world position of the visible surface point
    to_eye = cam["eye"] - pos
    to_eye = to_eye / to_eye.norm(dim=1, keepdim=True)
    base = torch.where(scene["is_mesh_face"][t, None], MESH_COLOUR.to(dev), FLOOR_COLOUR.to(dev))  # (P, 3)

    # Flat: one normal per face. Gouraud: light the corners, interpolate the brightness. Phong: interpolate normals.
    fn = face_normals(verts, faces)[t]
    fn = fn / fn.norm(dim=1, keepdim=True)
    centre = verts[faces[t]].mean(1)
    eye_c = (cam["eye"] - centre) / (cam["eye"] - centre).norm(dim=1, keepdim=True)
    flat = blinn_phong(fn, to_light, eye_c)
    corner_pos = verts[faces[t]]  # (P, 3, 3)
    corner_eye = cam["eye"] - corner_pos
    corner_eye = corner_eye / corner_eye.norm(dim=-1, keepdim=True)
    corner_light = blinn_phong(normals[faces[t]], to_light, corner_eye)  # (P, 3)
    gouraud = perspective_interp(lam, zc, corner_light[..., None])[:, 0]
    n_p = perspective_interp(lam, zc, normals[faces[t]])
    n_p = n_p / n_p.norm(dim=1, keepdim=True)
    phong = blinn_phong(n_p, to_light, to_eye)

    # Texture: checker looked up at perspective-correct versus affine texture coordinates.
    uv = scene["uv"][t]  # (P, 3, 2)
    uv_persp, uv_affine = perspective_interp(lam, zc, uv), affine_interp(lam, uv)
    diffuse = 0.45 + 0.55 * (n_p * to_light).sum(-1).clamp(min=0)

    def textured(uv_p):
        c = checker(uv_p)[:, None]
        return (c * TEXTURE_LIGHT.to(dev) + (1 - c) * TEXTURE_DARK.to(dev)) * diffuse[:, None]

    def image(rgb):
        rgb = torch.where(hit[:, None], rgb.clamp(0, 1), torch.ones_like(rgb))  # white background
        return rgb.reshape(res, res, 3)

    near, far = depth[hit].min(), depth[hit].max()
    grey = ((depth - near) / (far - near)).clamp(0, 1)[:, None].expand(-1, 3)
    floor_px = hit & ~scene["is_mesh_face"][t]
    return {
        "depth": image(0.15 + 0.8 * grey),
        "flat": image(flat[:, None] * base),
        "gouraud": image(gouraud[:, None] * base),
        "phong": image(phong[:, None] * base),
        "persp": image(textured(uv_persp)),
        "affine": image(textured(uv_affine)),
        "culled": int((~front).sum()),
        "triangles": len(faces),
        "covered": int(hit.sum()),
        "uv_error": float((uv_persp - uv_affine)[floor_px].norm(dim=1).max()) if floor_px.any() else 0.0,
    }


PANELS = ["depth", "flat", "gouraud", "phong", "persp", "affine"]


def animate_mesh(scene: dict, cam: dict, q, to_light, n_frames: int, tilt: float) -> tuple[torch.Tensor, list]:
    """Render n_frames of one full turn: images (F, 6, res, res, 3) as uint8 and the per-frame statistics."""
    frames, stats = [], []
    for f in steps(n_frames, "frames"):
        out = render(scene, cam, q, to_light, 2 * torch.pi * f / n_frames, tilt)
        frames.append(torch.stack([(out[k] * 255).round().to(torch.uint8) for k in PANELS]).cpu())
        stats.append({k: out[k] for k in ("culled", "triangles", "covered", "uv_error")})
    return torch.stack(frames), stats
