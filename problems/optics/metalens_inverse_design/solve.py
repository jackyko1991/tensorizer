"""Inverse design: maximise the focal-spot power with autograd + Adam, starting from a flat (do-nothing) surface."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import field_at_surface, focal_efficiency, kz, metasurface_phase, propagate


class Optics:
    """Everything fixed during the optimisation: grid, aperture, wavelengths, focal length, spot masks."""

    def __init__(self, wavelengths: torch.Tensor, lam0: float, f: float, diameter: float, device: torch.device):
        self.x, kx = tensors.grid(device)
        self.wavelengths = wavelengths.to(device)
        self.lam0, self.f, self.diameter = lam0, f, diameter
        self.aperture = tensors.aperture(self.x, diameter)
        self.kz = kz(kx, self.wavelengths)  # (L, Q)
        self.spot = tensors.spot_masks(self.x, self.wavelengths, f, diameter)  # (L, X)

    def surface_field(self, theta_atoms: torch.Tensor) -> torch.Tensor:
        theta = tensors.atoms_to_grid(theta_atoms, self.x, self.diameter)  # (X,)
        return field_at_surface(self.aperture, metasurface_phase(theta, self.wavelengths, self.lam0))  # (L, X)

    def efficiency(self, theta_atoms: torch.Tensor) -> torch.Tensor:
        """η_l for every wavelength (L,), differentiable with respect to theta_atoms."""
        E0 = self.surface_field(theta_atoms)
        z = torch.tensor([self.f], device=self.x.device)
        E_f = propagate(E0, self.kz, z)[:, 0]  # (L, X) field in the focal plane
        return focal_efficiency(E_f, E0, self.spot)

    @torch.no_grad()
    def field_map(self, theta_atoms: torch.Tensor, z: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """|E_l(x, z)| on the z grid and the x points selected by keep: (L, Z, X_kept)."""
        return propagate(self.surface_field(theta_atoms), self.kz, z)[..., keep].abs()


def optimise(optics: Optics, theta0: torch.Tensor, weights: torch.Tensor, n_iter: int, lr: float, n_snap: int = 60):
    """Adam on loss = -Σ_l weights_l η_l. θ is in waves, so lr = 0.02 moves a meta-atom by ~7° per step.

    Returns the efficiency history (n_iter + 1, L), the iterations of the snapshots and θ at each snapshot.
    """
    theta = theta0.clone().requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    # Snapshots closer together early on, where the design changes fastest.
    snap_iters = sorted({round(n_iter * (s / max(1, n_snap - 1)) ** 2) for s in range(n_snap)})
    history, snaps = [], []
    for it in steps(n_iter + 1, "Adam steps"):
        eta = optics.efficiency(theta)
        history.append(eta.detach().cpu())
        if it in snap_iters:
            snaps.append(theta.detach().clone())
        if it == n_iter:
            break
        loss = -(weights * eta).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.stack(history), snap_iters, torch.stack(snaps)


@torch.no_grad()
def on_axis_peak(optics: Optics, theta_atoms: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Distance z of the brightest point on the axis x = 0, for every wavelength: (L,)."""
    centre = optics.x.abs().argmin()
    intensity = propagate(optics.surface_field(theta_atoms), optics.kz, z)[..., centre].abs() ** 2  # (L, Z)
    return z[intensity.argmax(dim=-1)]
