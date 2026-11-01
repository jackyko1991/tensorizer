"""Build the concrete tensors: the scene (spheres + ground plane), the camera rays and the orbits."""

import math

import torch

from .equation import normalize


def scene(device: torch.device) -> dict[str, torch.Tensor]:
    """S spheres: centres c_{si} (S, 3), radii r_s (S,), colours (S, 3) and mirror fractions (S,)."""
    centres = torch.tensor(
        [[0.0, 1.0, 0.0], [-2.1, 0.6, 0.9], [1.9, 0.7, 1.0], [0.7, 0.35, 2.1], [-1.0, 0.45, -1.9]], device=device
    )
    radii = torch.tensor([1.0, 0.6, 0.7, 0.35, 0.45], device=device)
    albedo = torch.tensor(
        [[0.85, 0.85, 0.88], [0.86, 0.15, 0.15], [0.15, 0.39, 0.92], [0.09, 0.64, 0.29], [0.96, 0.62, 0.04]],
        device=device,
    )
    mirror = torch.tensor([0.7, 0.1, 0.25, 0.05, 0.15], device=device)  # 0 = matte, 1 = perfect mirror
    return {"centres": centres, "radii": radii, "albedo": albedo, "mirror": mirror}


def camera_basis(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rows: right, up, forward unit vectors of a camera at eye looking at target (no roll): (3, 3)."""
    forward = normalize(target - eye)
    right = normalize(torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0], device=eye.device)))
    up = torch.linalg.cross(right, forward)
    return torch.stack([right, up, forward])


def camera_rays(
    eye: torch.Tensor, target: torch.Tensor, fov_deg: float, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ray per pixel: origins o_{pi} and unit directions d_{pi}, both (H·W, 3), row p = i·W + j."""
    dev = eye.device
    half = math.tan(math.radians(fov_deg) / 2)  # half-height of the image plane at distance 1
    xs = ((torch.arange(width, device=dev) + 0.5) / width * 2 - 1) * half * width / height  # (W,)
    ys = (1 - (torch.arange(height, device=dev) + 0.5) / height * 2) * half  # (H,) image rows go down
    y, x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    screen = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)  # (P, 3) = (right, up, forward) weights
    d = normalize(torch.einsum("pk,ki->pi", screen, camera_basis(eye, target)))  # k summed over the 3 basis vectors
    return eye.expand_as(d), d


def orbit(n_frames: int, radius: float, height: float, phase: float, device: torch.device) -> torch.Tensor:
    """Positions on a horizontal circle, one per frame: (F, 3)."""
    angle = phase + torch.linspace(0, 2 * math.pi, n_frames + 1, device=device)[:-1]
    return torch.stack([radius * torch.sin(angle), torch.full_like(angle, height), radius * torch.cos(angle)], dim=-1)
