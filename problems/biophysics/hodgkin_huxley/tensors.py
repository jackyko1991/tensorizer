"""Build the concrete tensors: membrane constants, the batch of stimulus currents and the resting state.

Classic squid giant axon (Hodgkin & Huxley 1952) in the modern sign convention, resting potential −65 mV.
Units: V in mV, t in ms, currents in µA/cm², conductances in mS/cm², capacitance in µF/cm².
"""

import torch

from .equation import steady_state

C_M = 1.0  # membrane capacitance, µF/cm²
CHANNELS = [  # name, maximal conductance ḡ_k (mS/cm²), reversal potential E_k (mV)
    ("Na", 120.0, 50.0),
    ("K", 36.0, -77.0),
    ("leak", 0.3, -54.387),  # E_L chosen so that the resting potential is exactly −65 mV
]
V_REST = -65.0
N_CURRENTS = 41  # stimulus currents from 0 to I_max


def channels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """ḡ_k (3,) and E_k (3,) for k = Na, K, leak."""
    g = torch.tensor([gk for _, gk, _ in CHANNELS], device=device)
    E = torch.tensor([Ek for _, _, Ek in CHANNELS], device=device)
    return g, E


def stimulus_currents(I_max: float, device: torch.device) -> torch.Tensor:
    """I_b, shape (B,): one axon per current, evenly spaced from 0 to I_max."""
    return torch.linspace(0.0, I_max, N_CURRENTS, device=device)


def temperature_factor(temp: float) -> float:
    """φ = 3^((T − 6.3 °C)/10): the gates open and close 3× faster for every 10 °C (Q10 = 3)."""
    return 3.0 ** ((temp - 6.3) / 10.0)


def initial_state(n_axons: int, device: torch.device) -> torch.Tensor:
    """y0 of shape (4, B) = (V, m, h, n): every axon at rest, gates at their resting steady state."""
    V = torch.full((n_axons,), V_REST, device=device)
    return torch.cat([V[None], steady_state(V)])
