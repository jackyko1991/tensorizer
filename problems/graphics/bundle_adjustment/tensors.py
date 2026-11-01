"""Build the concrete tensors: true 3D points and cameras, noisy observations, the perturbed start and the gauge."""

import math

import torch

FOCAL = 1000.0  # px
IMAGE = 640  # px, square image; principal point at the centre
RADIUS = 5.0  # camera distance from the origin


def true_points(n_points: int, seed: int) -> torch.Tensor:
    """X_{pj}: n_points random points inside a ball of radius 1 around the origin.   → (P, 3) float64."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_points, 3, generator=gen, dtype=torch.float64)
    r = torch.rand(n_points, 1, generator=gen, dtype=torch.float64) ** (1 / 3)
    return x / x.norm(dim=1, keepdim=True) * r


def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Log map of SO(3), valid for rotation angles below π: (C, 3, 3) → ω (C, 3) with |ω| = angle."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    scale = torch.where(angle > 1e-8, angle / (2 * torch.sin(angle).clamp(min=1e-12)), 0.5)
    return v * scale[:, None]


def true_cameras(n_cams: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n_cams cameras on an arc in front of the points (azimuth -50° … 50°, alternately 10° above/below), all looking
    at the origin. Returns axis-angle ω (C, 3) and translation t (C, 3) with x_cam = R(ω) X + t.

    Camera axes: x right, y down, z forward (depth), so camera 0 at azimuth -50° sees the points along +z rotated.
    """
    az = torch.linspace(-math.radians(50), math.radians(50), n_cams, dtype=torch.float64)
    el = torch.tensor([math.radians(10.0 if k % 2 == 0 else -10.0) for k in range(n_cams)], dtype=torch.float64)
    centre = RADIUS * torch.stack([az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()], dim=-1)  # (C, 3)
    forward = -centre / centre.norm(dim=-1, keepdim=True)
    world_down = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).expand_as(forward)
    right = torch.linalg.cross(world_down, forward)
    right = right / right.norm(dim=-1, keepdim=True)
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward], dim=1)  # (C, 3, 3) rows = camera axes in world coordinates
    t = -torch.einsum("cij,cj->ci", R, centre)
    return rotmat_to_axis_angle(R), t


def pack(omega: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """All unknowns in one vector, cameras first: [ω_0, t_0, ω_1, t_1, …, X_0, X_1, …]   → (6C + 3P,)."""
    return torch.cat([torch.cat([omega, t], dim=1).reshape(-1), X.reshape(-1)])


def unpack(full: torch.Tensor, n_cams: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = full[: 6 * n_cams].reshape(n_cams, 6)
    return cam[:, :3], cam[:, 3:], full[6 * n_cams :].reshape(-1, 3)


def gauge_mask(n_cams: int, n_points: int) -> torch.Tensor:
    """True for the free unknowns. Fixed: all 6 numbers of camera 0 (rotation + position of the world frame) and the
    x-translation of camera 1 (the overall scale): 7 gauge freedoms, the similarity group of 3D space."""
    free = torch.ones(6 * n_cams + 3 * n_points, dtype=torch.bool)
    free[:6] = False  # camera 0
    free[6 + 3] = False  # t_x of camera 1
    return free


def perturb(full: torch.Tensor, n_cams: int, rot_deg: float, trans: float, point: float, seed: int) -> torch.Tensor:
    """A rough start: random rotations of ~rot_deg degrees, translation and point noise (world units)."""
    gen = torch.Generator().manual_seed(seed + 1)
    omega, t, X = unpack(full.clone(), n_cams)
    omega = omega + math.radians(rot_deg) * torch.randn(omega.shape, generator=gen, dtype=full.dtype)
    t = t + trans * torch.randn(t.shape, generator=gen, dtype=full.dtype)
    X = X + point * torch.randn(X.shape, generator=gen, dtype=full.dtype)
    return pack(omega, t, X)
