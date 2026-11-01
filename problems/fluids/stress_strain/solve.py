"""Each load step: stress → strain → displacement field → strain from its gradient → stress → principal stresses."""

import torch

from .equation import principal_stresses, strain, stress


def compliance(C: torch.Tensor) -> torch.Tensor:
    """S_{ijkl} with ε = S : σ — the inverse of C on symmetric tensors (pseudo-inverse of C as a 9×9 matrix)."""
    return torch.linalg.pinv(C.reshape(9, 9)).reshape(3, 3, 3, 3)


def displacement(eps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Uniform strain, no rotation: u_i = ε_{ij} x_j (j summed).   eps: (3, 3), x: (..., 3) → (..., 3)"""
    return torch.einsum("ij,...j->...i", eps, x)


def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂_i u_j by finite differences on the point grid: u, x (nx, ny, nz, 3) → (nx, ny, nz, 3, 3)."""
    spacing = [x[:, 0, 0, 0], x[0, :, 0, 1], x[0, 0, :, 2]]  # coordinates along each grid axis
    grads = torch.gradient(u, spacing=spacing, dim=(0, 1, 2))  # 3 tensors ∂_i u, each (nx, ny, nz, 3)
    return torch.stack(grads, dim=-2)  # [..., i, j] = ∂_i u_j


def load_path(sigma_applied: torch.Tensor, C: torch.Tensor, x: torch.Tensor) -> dict:
    """Run the whole loading ramp. Returns per-step tensors (T, ...)."""
    S = compliance(C)
    eps = torch.einsum("ijkl,tkl->tij", S, sigma_applied)  # (T, 3, 3) strain from Hooke's law inverted
    u = torch.stack([displacement(e, x) for e in eps])  # (T, nx, ny, nz, 3)
    eps_field = torch.stack([strain(gradient(ut, x)) for ut in u])  # (T, nx, ny, nz, 3, 3) from the displacement
    sigma = stress(C, eps_field.mean(dim=(1, 2, 3)))  # (T, 3, 3) back to stress: σ = C : ε
    values, vectors = principal_stresses(sigma)  # (T, 3), (T, 3, 3)
    in_plane, in_plane_dirs = principal_stresses(sigma[:, :2, :2])  # (T, 2), (T, 2, 2) the x-y block only
    return {"eps": eps, "u": u, "sigma": sigma, "principal": values, "directions": vectors,
            "in_plane": in_plane, "in_plane_dirs": in_plane_dirs}  # fmt: skip
