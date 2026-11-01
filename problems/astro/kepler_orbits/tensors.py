"""Build the concrete tensors: orbit sizes, eccentricities and the batched initial state.

Units: astronomical units (AU), years and solar masses, so GM_sun = 4π² AU³/yr² and an orbit with
a = 1 AU takes exactly one year.
"""

import math

import torch

GM_SUN = 4 * math.pi**2  # AU³ / yr²


def orbit_shapes(n: int, e_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-major axes a_b (AU) and eccentricities e_b, both (B,). Wider orbits are more eccentric."""
    a = torch.linspace(0.6, 2.2, n) if n > 1 else torch.tensor([1.0])
    e = torch.linspace(0.05, e_max, n) if n > 1 else torch.tensor([e_max])
    return a, e


def initial_state(a: torch.Tensor, e: torch.Tensor, GM: float, device: torch.device) -> torch.Tensor:
    r"""y0 of shape (B, 2, 2): every planet starts at perihelion on the +x axis, moving in +y.

    perihelion distance  r_p = a (1 - e)
    perihelion speed     v_p = \sqrt{(GM / a)(1 + e) / (1 - e)}     (vis-viva equation)
    """
    a, e = a.to(device), e.to(device)
    r_p = a * (1 - e)
    v_p = torch.sqrt(GM / a * (1 + e) / (1 - e))
    zero = torch.zeros_like(a)
    position = torch.stack([r_p, zero], dim=-1)  # (B, 2)
    velocity = torch.stack([zero, v_p], dim=-1)  # (B, 2)
    return torch.stack([position, velocity], dim=1)
