"""Build the concrete tensors: four processes from one start state, as (V, p) samples."""

import torch

from .equation import R

V1 = 10e-3  # start volume (m³) = 10 litres
GAMMA = {"monatomic": 5 / 3, "diatomic": 7 / 5}  # heat-capacity ratio γ = C_p / C_V
NAMES = ["isothermal", "isobaric", "isochoric", "adiabatic"]


def processes(n: float, T0: float, ratio: float, gamma: float, points: int = 200) -> torch.Tensor:
    """paths[c, k, :] = (V, p) of sample k on process c.  Shape (4, K, 2), float64 for accurate areas."""
    p1 = n * R * T0 / V1  # start pressure (Pa)
    s = torch.linspace(0, 1, points, dtype=torch.float64)  # (K,)  progress along the path
    V = V1 * (1 + (ratio - 1) * s)  # (K,)  expanding from V1 to ratio × V1

    isothermal = torch.stack([V, p1 * V1 / V], dim=-1)  # p = nRT / V
    isobaric = torch.stack([V, p1 * torch.ones_like(V)], dim=-1)  # p stays at p1
    isochoric = torch.stack([V1 * torch.ones_like(s), p1 * (1 + (ratio - 1) * s)], dim=-1)  # V stays at V1
    adiabatic = torch.stack([V, p1 * (V1 / V) ** gamma], dim=-1)  # p V^γ = const
    return torch.stack([isothermal, isobaric, isochoric, adiabatic])  # (4, K, 2)
