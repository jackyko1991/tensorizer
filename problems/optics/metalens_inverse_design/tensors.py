"""Build the concrete tensors: the x grid, its wavenumbers, the aperture, focal-spot masks and the meta-atom map."""

import math

import torch

WIDTH = 102.4  # simulation window (μm); the field is periodic in it, so keep the light away from its edges
N_GRID = 2048  # grid points → Δx = 0.05 μm, well below λ/2
PERIOD = 0.25  # meta-atom period (μm) = 5 grid points: one design value per meta-atom


def grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """x (X,) centred on 0 and the matching FFT wavenumbers k_x (Q = X,), in μm and 1/μm."""
    dx = WIDTH / N_GRID
    x = (torch.arange(N_GRID, device=device) - N_GRID // 2) * dx
    kx = 2 * math.pi * torch.fft.fftfreq(N_GRID, d=dx).to(device)
    return x, kx


def aperture(x: torch.Tensor, diameter: float) -> torch.Tensor:
    """A(x) = 1 on the lens |x| < D/2, 0 elsewhere (an opaque screen around it): (X,)."""
    return (x.abs() < diameter / 2).float()


def n_atoms(diameter: float) -> int:
    return int(round(diameter / PERIOD))


def atoms_to_grid(theta_atoms: torch.Tensor, x: torch.Tensor, diameter: float) -> torch.Tensor:
    """Spread one design value per meta-atom over its grid points: (n_atoms,) → (X,) (zero outside the lens)."""
    per_atom = int(round(PERIOD / (WIDTH / N_GRID)))
    inside = theta_atoms.repeat_interleave(per_atom)  # (n_atoms · per_atom,)
    start = int(torch.searchsorted(x, torch.tensor(-diameter / 2, device=x.device)))
    out = torch.zeros_like(x, dtype=theta_atoms.dtype)
    return torch.cat([out[:start], inside, out[start + len(inside) :]])


def atom_centres(diameter: float, device: torch.device) -> torch.Tensor:
    """x position of the centre of every meta-atom (n_atoms,)."""
    n = n_atoms(diameter)
    return (torch.arange(n, device=device) + 0.5) * PERIOD - diameter / 2


def spot_masks(x: torch.Tensor, wavelengths: torch.Tensor, f: float, diameter: float) -> torch.Tensor:
    """1 inside the focal spot |x| < w_l = λ_l f / D (the first zero of a diffraction-limited focus): (L, X)."""
    w = wavelengths * f / diameter  # (L,)
    return (x[None, :].abs() < w[:, None]).float()
