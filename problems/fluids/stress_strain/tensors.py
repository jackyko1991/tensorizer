"""Build the concrete tensors: Kronecker delta, the block's material points and the loading ramp."""

import torch


def kronecker(device: torch.device) -> torch.Tensor:
    """δ_ij: shape (3, 3)."""
    return torch.eye(3, device=device, dtype=torch.float64)


def block_points(n: int, device: torch.device) -> torch.Tensor:
    """Material points of a unit square plate (a few points thick): shape (n, n, 3, 3) → x_i at [ix, iy, iz, i].

    x and y run from -0.5 to 0.5 (in units of the plate size); z = -0.05, 0, 0.05 (a thin plate).
    """
    s = torch.linspace(-0.5, 0.5, n, device=device, dtype=torch.float64)
    z = torch.tensor([-0.05, 0.0, 0.05], device=device, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(s, s, z, indexing="ij")  # (n, n, 3) each
    return torch.stack([xx, yy, zz], dim=-1)


def loading(n_steps: int, sigma_max: float, tau_max: float, device: torch.device) -> torch.Tensor:
    """Applied stress σ_ij for each load step: shape (T, 3, 3).

    First half: uniaxial tension σ_xx ramps to σ_max. Second half: a shear σ_xy = σ_yx ramps to τ_max on top.
    σ_zz = 0 everywhere (plane stress: the faces of a thin plate are free).
    """
    s = torch.linspace(0.0, 1.0, n_steps, device=device, dtype=torch.float64)
    sigma = torch.zeros(n_steps, 3, 3, device=device, dtype=torch.float64)
    sigma[:, 0, 0] = sigma_max * torch.clamp(2 * s, max=1.0)
    sigma[:, 0, 1] = sigma[:, 1, 0] = tau_max * torch.clamp(2 * s - 1, min=0.0)
    return sigma
