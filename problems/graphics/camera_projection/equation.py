r"""Pinhole camera → tensor translation.

A 3D point is a homogeneous 4-vector X = (x, y, z, 1); a direction (a point at infinity) is (d_x, d_y, d_z, 0).
Three 4×4 matrices carry every vertex from the object to the screen (OpenGL convention, camera looking along −z):

    model M (object → world),   view V (world → camera),   projection P (camera → clip space)

    x̃_{vi} = P_{ij} V_{jk} M_{kl} X_{vl}          (j, k, l summed; v runs over all vertices at once)

Perspective divide and viewport turn clip coordinates into pixels (image rows run downwards):

    x^ndc_{va} = x̃_{va} / x̃_{v4},    u_v = W (x^ndc_{v1} + 1)/2,    v_v = H (1 − x^ndc_{v2})/2

Computer vision writes the same camera as λ (u, v, 1) = K [R|t] X: K holds the focal length f (pixels) and the
principal point (c_x, c_y), [R|t] is the rigid motion world → camera. A direction (w = 0) drops t, so
parallel lines with direction d all meet at the vanishing point K R d.
"""

import math

import torch


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def model_matrix(yaw_deg: float, scale: float, lift: float, device: torch.device) -> torch.Tensor:
    """M = T(0, lift, 0) · R_y(yaw) · S(scale): place the object on the floor.   → (4, 4)"""
    a = math.radians(yaw_deg)
    m = torch.eye(4, device=device)
    m[:3, :3] = scale * torch.tensor(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], device=device
    )
    m[1, 3] = lift
    return m


def look_at(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """View matrix V: rows right, up, −forward of the camera, then translate the eye to the origin.   → (4, 4)"""
    f = normalize(target - eye)
    r = normalize(torch.linalg.cross(f, torch.tensor([0.0, 1.0, 0.0], device=eye.device)))
    u = torch.linalg.cross(r, f)
    v = torch.eye(4, device=eye.device)
    v[:3, :3] = torch.stack([r, u, -f])
    v[:3, 3] = -v[:3, :3] @ eye  # t = −R · eye
    return v


def perspective(fov_deg: float, aspect: float, near: float, far: float, device: torch.device) -> torch.Tensor:
    """Perspective projection P: x̃_4 = −z_cam, so dividing by it shrinks far objects.   → (4, 4)"""
    f = 1 / math.tan(math.radians(fov_deg) / 2)
    return torch.tensor(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        device=device,
    )


def orthographic(half_height: float, aspect: float, near: float, far: float, device: torch.device) -> torch.Tensor:
    """Orthographic projection: x̃_4 = 1 always, so size does not depend on distance.   → (4, 4)"""
    return torch.tensor(
        [
            [1 / (aspect * half_height), 0, 0, 0],
            [0, 1 / half_height, 0, 0],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1],
        ],
        device=device,
    )


def clip_coords(p: torch.Tensor, v: torch.Tensor, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x̃_{vi} = P_{ij} V_{jk} M_{kl} X_{vl}   (j, k, l summed).   X: (N, 4) → (N, 4)"""
    return torch.einsum("ij,jk,kl,vl->vi", p, v, m, x)


def to_pixels(clip: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Perspective divide x̃_a / x̃_4, then the viewport map to pixels (u right, v down).   (N, 4) → (N, 2)"""
    ndc = clip[:, :2] / clip[:, 3:4]
    return torch.stack([width * (ndc[:, 0] + 1) / 2, height * (1 - ndc[:, 1]) / 2], dim=-1)


def intrinsics(fov_deg: float, width: int, height: int, device: torch.device) -> torch.Tensor:
    """K: focal length f = (H/2)/tan(fov/2) in pixels and principal point (W/2, H/2).   → (3, 3)"""
    f = height / 2 / math.tan(math.radians(fov_deg) / 2)
    return torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], device=device)


def extrinsics(view: torch.Tensor) -> torch.Tensor:
    """[R|t] in the vision convention (x right, y down, z forward): flip the y and z rows of V.   → (3, 4)"""
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=view.device))
    return flip @ view[:3]


def pinhole(k: torch.Tensor, rt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""λ_v û_{va} = K_{ab} [R|t]_{bl} X_{vl}   (b, l summed), then divide by λ_v = depth.   (N, 4) → (N, 2)"""
    h = torch.einsum("ab,bl,vl->va", k, rt, x)  # (N, 3)
    return h[:, :2] / h[:, 2:3]


def vanishing_point(k: torch.Tensor, rt: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Image of the point at infinity (d, 0): where all lines with direction d meet.   (3,) → (2,)"""
    x = torch.cat([direction, direction.new_zeros(1)])[None]  # (1, 4) with w = 0
    return pinhole(k, rt, x)[0]
