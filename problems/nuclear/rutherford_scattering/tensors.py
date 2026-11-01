"""Build the concrete tensors: coupling constant, impact parameters and the batched initial state."""

import math

import torch

ALPHA_MASS = 3727.379  # MeV / c²
E2 = 1.44  # e² / 4πε0 in MeV fm


def coupling(z_target: int) -> float:
    """k = Z_α Z e²/4πε0 in MeV fm (Z_α = 2)."""
    return 2 * z_target * E2


def display_impact_parameters(n_show: int, b_max: float) -> torch.Tensor:
    """Evenly spaced b from −b_max to +b_max for the trajectories we draw: shape (S,)."""
    return torch.linspace(-b_max, b_max, n_show, dtype=torch.float64)


def beam_impact_parameters(n_beam: int, b_max: float, seed: int = 0) -> torch.Tensor:
    """A uniform beam over a disc of radius b_max: b = b_max √u gives equal flux per unit area: shape (B,)."""
    gen = torch.Generator().manual_seed(seed)
    return b_max * torch.rand(n_beam, generator=gen, dtype=torch.float64).sqrt()


def initial_state(b: torch.Tensor, energy_mev: float, k: float, start: float, device: torch.device) -> torch.Tensor:
    """y0 of shape (B, 2, 2): every alpha starts on the left at distance `start` from the nucleus, moving along +x.

    The speed is set so the total energy is exactly E (part of it is already Coulomb energy k/start), and the
    lateral offset is stretched so the angular momentum is exactly m v∞ b — the beam is then the one that
    came in from infinity with energy E and impact parameter b.
    """
    b = b.to(device)
    v_inf = math.sqrt(2 * energy_mev / ALPHA_MASS)
    v_start = math.sqrt(2 * (energy_mev - k / start) / ALPHA_MASS)
    y = b * v_inf / v_start
    position = torch.stack([-torch.sqrt(start**2 - y**2), y], dim=-1)  # (B, 2), |x| = start
    velocity = torch.stack([torch.full_like(b, v_start), torch.zeros_like(b)], dim=-1)  # (B, 2)
    return torch.stack([position, velocity], dim=1)
