"""Build the concrete tensors: the batch of beam energies and reference ranges."""

import torch

# NIST PSTAR CSDA ranges of protons in water (g/cm² = cm at ρ = 1), rounded.
PSTAR = {70.0: 4.08, 100.0: 7.72, 150.0: 15.77, 200.0: 25.96, 250.0: 37.94}


def beam_energies(e_min: float, e_max: float, n: int, device: torch.device) -> torch.Tensor:
    """E0_b evenly spread between e_min and e_max (MeV): shape (B,)."""
    if n == 1:
        return torch.tensor([0.5 * (e_min + e_max)], device=device)
    return torch.linspace(e_min, e_max, n, device=device)


def reference_ranges(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """PSTAR (energy (R,), CSDA range (R,)) for comparison."""
    return torch.tensor(list(PSTAR), device=device), torch.tensor(list(PSTAR.values()), device=device)


def layer_ranges(z_start: float, z_end: float, n_layers: int, device: torch.device) -> torch.Tensor:
    """Ranges of the SOBP energy layers, spread a little beyond the target on both sides: (K,)."""
    return torch.linspace(z_start - 0.5, z_end + 0.5, n_layers, device=device)


def layer_energies(ranges: torch.Tensor, alpha: float, p: float) -> torch.Tensor:
    """Invert Bragg–Kleeman R = α E^p: the energy that stops at each wanted depth: (K,)."""
    return (ranges / alpha) ** (1.0 / p)
