"""Evaluate the attenuation law: projection images, depth profiles of each ray, contrast and beam hardening."""

import torch

from . import equation


def projections(
    mu: torch.Tensor, mu_chosen: torch.Tensor, masks: torch.Tensor, dl: torch.Tensor, spectrum: torch.Tensor
):
    """Return (L (M, X), monochromatic I/I0 at the table energies + the chosen one (E+1, X),
    polychromatic I/I0 (X,))."""
    lengths = equation.path_lengths(masks, dl)  # (M, X)
    mono = equation.transmission(torch.cat([mu, mu_chosen[None]]), lengths)  # (E+1, X)
    poly = equation.polychromatic(spectrum, mu, lengths)  # (X,)
    return lengths, mono, poly


def ray_profiles(mu_at_energy: torch.Tensor, masks: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    r"""I(y)/I_0 inside the body for every ray: exp(-\sum_{y' \le y} \mu(y', x) \Delta l_{y'}):  (Y, X)."""
    mu_map = torch.einsum("m,myx->yx", mu_at_energy, masks)  # μ(y, x) at one energy
    return torch.exp(-torch.cumsum(mu_map * dl[:, None], dim=0))


def contrast_vs_energy(mu: torch.Tensor, thickness: float = 1.0) -> torch.Tensor:
    """Contrast of a 1 cm insert of lung, bone and iodine contrast against soft tissue: (E, 3)."""
    soft = mu[:, 2:3]  # (E, 1)
    inserts = mu[:, [1, 3, 4]]  # lung, bone, iodine contrast
    return equation.contrast(inserts, soft, thickness)


def beam_hardening(mu_soft: torch.Tensor, spectrum: torch.Tensor, e: torch.Tensor, max_depth: float = 30.0):
    """A polychromatic beam through a growing slab of soft tissue.

    Returns (depth (D,), I/I0 (D,), μ_eff (D-1,), mean energy of the transmitted photons (D,)).
    """
    depth = torch.linspace(0.0, max_depth, 121, device=e.device)
    survived = spectrum[None, :] * torch.exp(-depth[:, None] * mu_soft[None, :])  # (D, E) photons left per bin
    intensity = survived.sum(1)
    mean_energy = torch.einsum("de,e->d", survived, e) / intensity
    return depth, intensity, equation.effective_mu(depth, intensity), mean_energy


def half_value_layer(mu: float) -> float:
    """Thickness that halves a monochromatic beam: HVL = ln 2 / μ."""
    return float(torch.log(torch.tensor(2.0)) / mu)
