"""Build the concrete tensors: launch speeds, launch sites and the batched initial state.

Units: the planet's radius R = 1 and GM = 1, so the escape speed is √2 and one time unit is
√(R³/GM) (for the Earth about 13 minutes).
"""

import math

import torch

GM = 1.0
R = 1.0


def speed_fractions(n: int, f_min: float, f_max: float) -> torch.Tensor:
    """Launch speeds as fractions v / v_esc, (B,)."""
    return torch.linspace(f_min, f_max, n) if n > 1 else torch.tensor([f_min])


def initial_state(fractions: torch.Tensor, v_esc: float, tilt_deg: float, device: torch.device) -> torch.Tensor:
    """y0 of shape (B, 2, 2): rockets spread over the upper half of the planet, each launched outward.

    Rocket b sits on the surface at polar angle φ_b and is fired at speed f_b v_esc, tilted by tilt_deg
    from the local vertical (0 = straight up).
    """
    B = len(fractions)
    phi = torch.linspace(math.radians(160), math.radians(20), B) if B > 1 else torch.tensor([math.pi / 2])
    up = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # (B, 2) local vertical
    side = torch.stack([torch.sin(phi), -torch.cos(phi)], dim=-1)  # (B, 2) local horizontal
    tilt = math.radians(tilt_deg)
    direction = math.cos(tilt) * up + math.sin(tilt) * side
    position = R * up
    velocity = (fractions * v_esc)[:, None] * direction
    return torch.stack([position, velocity], dim=1).to(device)
