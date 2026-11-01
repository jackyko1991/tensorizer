"""Build the concrete tensors: the Laplacian kernel, the damping border and the two sources."""

import torch

N = 200  # cells per side
BORDER = 30  # absorbing border width (cells)
SOURCE_Y = 60.0  # height of the two sources
H = 1.0  # cell size
C = 1.0  # wave speed


def grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell centres X, Y of shape (n, n), indexed [a, b] = [x, y]."""
    s = torch.arange(N, device=device, dtype=torch.float32) * H
    return torch.meshgrid(s, s, indexing="ij")


def stencil(device: torch.device) -> torch.Tensor:
    """K_{αβ} of shape (1, 1, 3, 3): the five-point Laplacian (out/in channel dimensions for conv2d)."""
    K = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device)
    return K.view(1, 1, 3, 3)


def damping(device: torch.device, strength: float = 0.25) -> torch.Tensor:
    """γ_{ab} (n, n): 0 inside, growing like (depth into the border)² up to `strength` at the walls."""
    s = torch.arange(N, device=device, dtype=torch.float32)
    depth = torch.clamp(BORDER - torch.minimum(s, N - 1 - s), min=0) / BORDER  # (n,) 0 inside → 1 at the wall
    d2 = torch.maximum(depth[:, None], depth[None, :])
    return strength * d2**2


def sources(X: torch.Tensor, Y: torch.Tensor, separation: float, width: float = 1.5):
    """Two Gaussian blobs (n, n) summed into one tensor, and their centres."""
    x_mid = (N - 1) * H / 2
    s1, s2 = (x_mid - separation / 2, SOURCE_Y), (x_mid + separation / 2, SOURCE_Y)
    blob = sum(torch.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * width**2)) for sx, sy in (s1, s2))
    return blob, s1, s2
